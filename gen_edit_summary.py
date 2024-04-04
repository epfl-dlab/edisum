import hydra
import csv
from src.tools.hydra_tools import get_config
from omegaconf import DictConfig, OmegaConf
import argparse

def simple_gen_summary(config_name, output_file, output_type='json', output_cols=[''], limit=-1, general_sample_revids=[], overrides=[]):
    """generate edit summary for pre-processed wikidump data
    """
    cfg = get_config(config_name=config_name, overrides=overrides)
    # Initialize the prompting strategy
    prompting = hydra.utils.instantiate(cfg.openai.prompting)
    # Initialize the model
    model = hydra.utils.instantiate(cfg.openai.model, prompting=prompting)
    # Initialize the dataset module
    dataset = hydra.utils.instantiate(cfg.openai.datamodule, limit=limit)
    
    dataloader = dataset.dataloader()
    model.predict_datamodule(dataloader, output_file, output_type, output_cols)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--cache_path", type=str)
    parser.add_argument("--demonstration_path", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    config_name="root_config"
    
    output_file = args.output_file 
    output_cols = ['page_id', 'rev_id', 'summary', 'model_completions', 'url', 'prev_texts', 'cur_texts']
    
    data_path = args.data_path
    cache_path = args.cache_path 
    extend_cache = True

    model = args.model # gpt-3.5-turbo or gpt-4
    prompting = "chat_prompting"
    formatter = "gen_summary_chat_formatter_system_only"
    gen_dataset = False
    no_gen = False  # If true, this method would only be used to format the raw wikidump data to the training set format
    
    """Demonstration instruction
    order changed even after sort: 1107801612, 1107804547, 1107807151, 1107808056
    
    """
    general_sample_revids = [1107804776, 1107801638] # [1107803417, 1107804776] # [1107804400, 1107804482]
    num_retrieve = 3  # -1 is all, number of demonstrations to retrieve
    demonstration_path = args.demonstration_path
    gen_limit = 500 # -1
    output_type = 'csv'
    dry_run = False  # set to True to check the prompts
    
    api_keys = args.api_key
    
    
    overrides = []
    overrides.append("openai.prompting.retriever.general_sample_revids={}".format(general_sample_revids))
    overrides.append("openai.prompting.retriever.num_retrieve={}".format(num_retrieve))
    overrides.append("openai.prompting.retriever.path_to_edit_samples={}".format(demonstration_path))
    overrides.append("openai.datamodule.demonstration_path={}".format(demonstration_path))
    overrides.append("openai.datamodule.data_path={}".format(data_path))
    overrides.append("openai.datamodule.cache_path={}".format(cache_path))
    overrides.append("openai.datamodule.extend_cache={}".format(extend_cache))
    overrides.append(f"openai/model={model}")
    overrides.append(f"openai.model.dry_run={dry_run}")
    overrides.append(f"openai.model.no_gen={no_gen}")
    overrides.append(f"openai.model.gen_dataset={gen_dataset}")
    overrides.append(f"openai.model.api_keys={api_keys}")
    overrides.append(f"openai/prompting={prompting}")
    overrides.append(f"openai/prompting/formatter={formatter}")

    
    simple_gen_summary(config_name=config_name,
                       output_file=output_file, 
                       output_type=output_type, 
                       output_cols=output_cols, 
                       limit=gen_limit,
                       general_sample_revids=general_sample_revids,
                       overrides = overrides)
    
            