import torch
import os
import shutil
import hydra

from typing import List, Callable
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import Logger

from src.tools.logger import get_pylogger


log = get_pylogger(__name__)

def label_smoothed_nll_loss(
    lprobs: torch.Tensor,
    target: torch.Tensor,
    target_attention_mask: torch.Tensor,
    epsilon: float,
    ignore_index: int = None,
    reduce: bool = True,
):
    # target.shape -> batch_size x tgt_seq_length; lprobs.shape -> batch_size x tgt_seq_length x vocabulary_size
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)  # target.shape -> batch_size x tgt_seq_length x 1

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target.clamp_min_(0)

        nll_loss = -lprobs.gather(dim=-1, index=target)  # get the log prob terms corresponding to the target indices
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)  # calculations needed for the smoothed loss

        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    nll_loss = nll_loss.squeeze(-1)
    smooth_loss = smooth_loss.squeeze(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    # There are in total lprobs.size(-1) - 1 valid classes (the padding token is ignored)
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    # Normalize the loss by diving with the number of non-padding tokens
    num_tokens = target_attention_mask.sum()
    loss, nll_loss = loss / num_tokens, nll_loss / num_tokens

    return loss, nll_loss

def get_predictions_dir_path(output_dir, create_if_not_exists=True):
    if output_dir is not None:
        predictions_folder = os.path.join(output_dir, "predictions")
    else:
        predictions_folder = "predictions"

    if create_if_not_exists:
        Path(predictions_folder).mkdir(parents=True, exist_ok=True)

    return predictions_folder

@rank_zero_only
def _move_predictions_for_subprocesses(predictions_dir_src, predictions_dir_dst):
    if os.path.exists(predictions_dir_src):
        for f in os.listdir(predictions_dir_src):
            shutil.move(os.path.join(predictions_dir_src, f), os.path.join(predictions_dir_dst, f))
        shutil.rmtree(predictions_dir_src)
        
        
@rank_zero_only
def upload_outputs_to_wandb(hparams_to_log, output_dir, logger):
    loggers = [logger]

    for logger in loggers:
        if isinstance(logger, WandbLogger):
            output_files = os.listdir(output_dir)
            output_files = [os.path.relpath(os.path.join(output_dir, f)) for f in output_files]

            logger.experiment.save(f"{output_dir}/*", base_path=".", policy="now")
            logger.experiment.config["output_files"] = output_files
            logger.experiment.config.update(hparams_to_log, allow_val_change=True)
            

def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger

@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.
    Additionally, it saves:
    - Number of model parameters
    """
    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["hydra_config"] = cfg
    # Add number of model parameters to logged information
    hparams["params"] = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "non_trainable": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }

    for key in hparams:
        if isinstance(hparams[key], DictConfig):
            hparams[key] = OmegaConf.to_container(hparams[key], resolve=True)

    # send hparams to all loggers
    for logger in trainer.loggers:
        # ToDo: The config's nested structure is not preserved by WandB. Why? Is this a bug fixed in newer versions?
        logger.log_hyperparams(hparams)
