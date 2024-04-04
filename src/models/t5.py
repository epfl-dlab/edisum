import os
from typing import List, Any
from statistics import mean
from omegaconf import OmegaConf

import hydra
import torch
import transformers
import pandas as pd
import gzip
import jsonlines

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bert import BERTScore
from torchmetrics import BLEUScore


from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, LongT5Config, LongT5ForConditionalGeneration
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from src.tools.general_tools import rec_dict_update, chunk_elements
from src.tools.logger import get_pylogger
from src.tools.trainning_utils import label_smoothed_nll_loss, get_predictions_dir_path, _move_predictions_for_subprocesses, upload_outputs_to_wandb
from src.models.collators import DefaultCollator

log = get_pylogger(__name__)

class T5PL(LightningModule):
    def __init__(
        self,
        hparams_overrides=None,
        hf_config_overrides=None,
        from_pretrained=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "hparams_overrides",
                "hf_config_overrides",
                "datamodule",
                "collator"
            ],
        )
        
        if hparams_overrides is not None:
            self._override_checkpoint_hparams(hparams_overrides)
        
        # ~~~ Load the tokenizer ~~~
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.pretrained_model_name_or_path, extra_ids=0, additional_special_tokens=OmegaConf.to_object(self.hparams.additional_special_tokens))
        
        # ~~~ Get the HF config ~~~
        config_class = LongT5Config if "long" in self.hparams.pretrained_model_name_or_path else T5Config
        hf_config = config_class.from_pretrained(self.hparams.pretrained_model_name_or_path)
        # Override the HF config with values from the checkpoint (if loading from checkpoint)
        if self.hparams.get("hf_config", None):
            hf_config.update(self.hparams.hf_config.to_dict())
        # Override HF config parameters (if it applies)
        if hf_config_overrides is not None:
            hf_config.update(hf_config_overrides)
        # Update the hparams with the updated config
        self.hparams.hf_config = hf_config
        
        model_class = LongT5ForConditionalGeneration if "long" in self.hparams.pretrained_model_name_or_path else T5ForConditionalGeneration
        log.info(f"Use model class {model_class.__name__}")
        # ~~~ Load the model ~~~
        if from_pretrained:
            self.model = model_class.from_pretrained(
                self.hparams.pretrained_model_name_or_path, config=self.hparams.hf_config
            )
        else:
            self.model = model_class(config=self.hparams.hf_config)
        log.info("HF model config:")
        log.info(self.hparams.hf_config)
        
        # ~~~ Set collator ~~~
        self.collator = kwargs.get("collator", None)
        if self.collator is None:
            self.collator = DefaultCollator(tokenizer=self.tokenizer, **self.hparams.default_collator_parameters)
        else:
            self.collator.set_tokenizer(self.tokenizer)
            
        # ~~~ Initialize metrics ~~~
        self.bleu_score = BLEUScore(n_gram=4)
        self.bert_score = BERTScore(
            model_name_or_path='roberta-large',
            max_length=self.collator.params.get("max_input_length", 512),
            batch_size=self.hparams.batch_size)
        self.rouge_score = ROUGEScore()
        
    def process_batch(self, batch):
        # get the decoder input ids by shifting target ids
        batch["decoder_input_ids"] = self.model._shift_right(batch["tgt_input_ids"])

        return batch
        
    def _override_checkpoint_hparams(self, hparams_overrides: dict):
        """
        Overrides the hyperparameters of a checkpoint at an arbitrary depth
        :param hparams_overrides:
        :return:
        """
        rec_dict_update(self.hparams, hparams_overrides)
        log.info("Some values of the original hparams were overridden")
        log.info("Hyper-parameters:")
        log.info(self.hparams)
        
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None, **kwargs):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs,
        )

        return output
    
    def _compute_loss(self, batch):
        model_output = self(
            input_ids=batch["src_input_ids"],
            attention_mask=batch["src_attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["tgt_attention_mask"],
            use_cache=False,
        )

        logits = model_output.logits

        loss, nll_loss = label_smoothed_nll_loss(
            logits.log_softmax(dim=-1),
            batch["tgt_input_ids"],
            batch["tgt_attention_mask"],
            epsilon=self.hparams.eps,
            ignore_index=self.tokenizer.pad_token_id,
        )
        
        return loss, nll_loss
        
    def training_step(self, batch, batch_idx):
        batch = self.process_batch(batch)

        loss, nll_loss = self._compute_loss(batch)

        self.log("train/nll_loss", 
                 nll_loss.item(), 
                 on_step=True, 
                 on_epoch=False, 
                 prog_bar=True)

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self.process_batch(batch)

        loss, nll_loss = self._compute_loss(batch)
        self.log("val/nll_loss", nll_loss.item(), on_step=False, on_epoch=True, prog_bar=True)

        return {"val/nll_loss": nll_loss}
    
    def test_step(self, batch, batch_idx):
        raw_input = [sample["text"] for sample in batch["raw"]]
        raw_target = [sample["target"] for sample in batch["raw"]]
        ids = batch["id"]

        sample_output = self._get_predictions_for_batch(batch, raw_input)

        self._write_step_output(
            batch_idx=batch_idx, ids=ids, raw_input=raw_input, raw_target=raw_target, sample_output=sample_output
        )

        return_object = {
            "ids": ids,
            "inputs": raw_input,
            "targets": raw_target,
            "predictions": sample_output["grouped_decoded_sequences"],
        }
        return return_object
    
    def on_test_batch_end(self, outputs: List[Any], batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # Get the data in the format expected by the metrics
        predictions = outputs["predictions"]  
        assert len(predictions[0]) == 1, "currently only support num_return_sequences=1 in model generate"
        predictions = [x[0] for x in predictions]
        targets = outputs["targets"]

        # Update the metrics
        bleu_score = self.bleu_score(predictions, targets)
        # if len(predictions) > 1:
        #     bert_score_f1 = mean(self.bert_score(predictions, targets)['f1'])  # list of F1 scores (num={batch_size})
        # else:
        #     bert_score_f1 = self.bert_score(predictions, targets)['f1']
        rougeLsum_fmeasure_score = self.rouge_score(predictions, targets)['rougeLsum_fmeasure']

        # Log the loss
        self.log("test/bleu_score_step", bleu_score, on_step=True, on_epoch=False, prog_bar=True)
        # self.log("test/bert_score_f1_step", bert_score_f1, on_step=True, on_epoch=False, prog_bar=True)
        self.log("test/rougeLsum_fmeasure_score_step", rougeLsum_fmeasure_score, on_step=True, on_epoch=False, prog_bar=True)
        
    def on_test_epoch_end(self):
        if hasattr(torch.distributed, "is_initialized") and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Temporary solution to Hydra + PL + DDP issue
        # https://github.com/Lightning-AI/lightning/pull/11617#issuecomment-1245842064
        # https://github.com/ashleve/lightning-hydra-template/issues/393
        # problem should be resolved in PL version 1.8.3
        _move_predictions_for_subprocesses(
            get_predictions_dir_path(os.getcwd()),
            get_predictions_dir_path(self.output_dir),
        )

        upload_outputs_to_wandb(
            getattr(self, "hparams_to_log", {}),
            get_predictions_dir_path(self.output_dir),
            logger=self.logger,
        )
        
        """Outputs is a list of test_step outputs"""
        # Log metrics aggregated across steps and processes (in ddp)
        self.log("test/bleu_score", self.bleu_score.compute())
        # self.log("test/bert_score_f1", mean(self.bert_score.compute()['f1']))
        self.log("test/rougeLsum_fmeasure_score", self.rouge_score.compute()['rougeLsum_fmeasure'])

        return {
            "test/bleu_score": self.bleu_score.compute(),
            # "test/bert_score_f1": mean(self.bert_score.compute()['f1']),
            "test/rougeLsum_fmeasure_score": self.rouge_score.compute()['rougeLsum_fmeasure'],
        }
    
    def _get_predictions_for_batch(self, batch, raw_input):
        # ~~~ Prediction related ~~~
        # Generate predictions
        hf_generation_params = self.hparams.inference["hf_generation_params"].copy()
        hf_generation_params.update(
            {
                "return_generation_inputs": True,
                "return_generation_outputs": True,
                "output_scores": True,
            }
        )
        
        sample_output = self.sample(
            batch,
            **hf_generation_params,
        )

        return sample_output  
    
    def _write_step_output(
        self,
        batch_idx,
        ids,
        raw_input,
        raw_target,
        sample_output,
    ):
        # ~~~ Write prediction outputs to file ~~~
        num_return_sequences = len(sample_output["grouped_decoded_sequences"][0])
        sequences = sample_output["generation_outputs"].sequences
        assert isinstance(sequences, torch.Tensor)
        prediction_ids = chunk_elements(sequences.tolist(), num_return_sequences)

        prediction_outputs = {
            "id": ids,
            "input": raw_input,
            "target": raw_target,
            "prediction": sample_output["grouped_decoded_sequences"],
            "prediction_ids": str(prediction_ids),
        }

        prediction_outputs_path = os.path.join(
            get_predictions_dir_path(self.output_dir),
            f"testing_output_{self.global_rank}.prediction.jsonl.gz",
        )
        
        
        mode = "w" if not os.path.exists(prediction_outputs_path) else "a"
        with gzip.open(prediction_outputs_path, mode) as fp:
            json_writer = jsonlines.Writer(fp)
            json_writer.write_all([prediction_outputs])
        
    def sample(self, 
               batch, 
               seed=None,
               skip_special_tokens=True,
               return_generation_outputs=False,
               return_generation_inputs=False,
               **kwargs):
        if self.training:
            self.eval()
            
        hf_generation_params = self.hparams.inference["hf_generation_params"].copy()
        hf_generation_params.update(kwargs)
        hf_generation_params["return_dict_in_generate"] = True
        
        if seed is None:
            seed = self.hparams.inference.get("seed", None)
        if seed:
            transformers.trainer_utils.set_seed(seed)
            
        input_ids = batch["src_input_ids"].to(self.device)
        attention_mask = batch["src_attention_mask"].to(self.device)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **hf_generation_params,
        }
        generation_outputs = self.model.generate(**generate_kwargs)
        sequences = generation_outputs.sequences
        
        # Returns a list of `num_sentences` decoded (textual) sequences
        num_return_sequences = hf_generation_params.get("num_return_sequences", 1)
        
        # TODO do we need this?
        decoded_sequences = self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)
        grouped_decoded_sequences = chunk_elements(decoded_sequences, num_return_sequences)

        if self.training:
            self.train()

        results = {"grouped_decoded_sequences": grouped_decoded_sequences}
        if return_generation_inputs:
            results["generate_kwargs"] = generate_kwargs
        if return_generation_outputs:
            results["generation_outputs"] = generation_outputs

        return results
    
    def configure_optimizers(self):
        # Apply weight decay to all parameters except for the biases and the weight for Layer Normalization

        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        # Per-parameter optimization.
        # Each dict defines a parameter group and contains the list of parameters to be optimized in a key `params`
        # Other keys should match keyword arguments accepted by the optimizers and
        # will be used as optimization params for the parameter group
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.hparams.optimizer.weight_decay,
                "betas": (0.9, 0.999),
                "eps": self.hparams.optimizer.eps,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
                "betas": (0.9, 0.999),
                "eps": self.hparams.optimizer.eps,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.optimizer.lr,
            weight_decay=self.hparams.optimizer.weight_decay,
        )

        if self.hparams.scheduler.name == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.scheduler.warmup_updates,
                num_training_steps=self.hparams.scheduler.total_num_updates,
            )
        elif self.hparams.scheduler.name == "polynomial":
            scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.scheduler.warmup_updates,
                num_training_steps=self.hparams.scheduler.total_num_updates,
                lr_end=self.hparams.scheduler.lr_end,
            )
        elif self.hparams.scheduler.name is not None:
            raise ValueError("Unknown scheduler name {}".format(self.hparams.scheduler.name))

        lr_dict = {
            "scheduler": scheduler,  # scheduler instance
            "interval": "step",  # The unit of the scheduler's step size. 'step' or 'epoch
            "frequency": 1,  # corresponds to updating the learning rate after every `frequency` epoch/step
            "name": f"LearningRateScheduler-{self.hparams.scheduler.name}",  # used by the LearningRateMonitor callback
        }

        return [optimizer], [lr_dict]
