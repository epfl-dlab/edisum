import collections
import os
import hydra
import time
import signal
import rich
import rich.syntax
import rich.tree
import warnings
import shutil

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from itertools import islice, zip_longest
from pytorch_lightning.utilities import rank_zero_only
from typing import Sequence
from typing import List, Callable
from importlib.util import find_spec

from src.tools.logger import get_pylogger

log = get_pylogger(__name__)



def batched(iterable, batch_size):
    """Batch data into lists of length batch_size. The last batch may be shorter.
    Ref-> https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks"""
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            return
        yield batch
        

def rec_dict_update(d, u):
    """Performs a multilevel overriding of the values in dictionary d with the values of dictionary u"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = rec_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def chunk_elements(elements, n_items_per_group):
    """
    Parameters
    ----------
    elements: elements to be grouped, [s1, s2, s3, s3, s5, s6]
    n_items_per_group: integer if all groups are of equal size, or a list of varying group lengths
    Returns
    -------
    A list of grouped sequences.
    Example 1:
    elements=[s1,s2,s3,s3,s5,s6]
    n_items_per_group=3
    returns: [[s1,s2,s3], [s4,s5,s6]]
    Example 2:
    elements=[s1,s2,s3,s3,s5,s6]
    n_items_per_group=[2,4]
    returns: [[s1,s2], [s3,s4,s5,s6]]
    """
    if isinstance(n_items_per_group, int):
        assert len(elements) % n_items_per_group == 0
        n_items_per_group = [n_items_per_group for _ in range(len(elements) // n_items_per_group)]

    grouped_sequences = []
    start_idx = 0
    for n_items in n_items_per_group:
        grouped_sequences.append(elements[start_idx : start_idx + n_items])
        start_idx += n_items

    return grouped_sequences


def get_absolute_path(path):
    """Get absolute path (relative to the original working directory) from a (potentially) relative path."""
    if not os.path.isabs(path):
        return os.path.join(hydra.utils.get_original_cwd(), path)
    return path

@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callback",
        "logger",
        "trainer",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        current_time_stamp = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
        with open(Path(cfg.output_dir, f"config_tree_{current_time_stamp}.log"), "w") as file:
            rich.print(tree, file=file)

def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.
    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings
    if cfg.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library
    if cfg.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)
        
@rank_zero_only
def save_string_to_file(path: str, content: str, append_mode=True) -> None:
    """Save string to file in rank zero mode (only on the master process in multi-GPU setup)."""
    mode = "a+" if append_mode else "w+"
    with open(path, mode) as file:
        file.write(content)
        
def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()
            
@rank_zero_only
def _clean_working_dir_from_subprocesses_output(work_dir, output_dir):
    # Move the hydra folder
    hydra_folder_src = os.path.join(work_dir, ".hydra")
    hydra_folder_target = os.path.join(output_dir, ".hydra_subprocesses")
    if os.path.exists(hydra_folder_src):
        shutil.move(hydra_folder_src, hydra_folder_target)

    # Move the logs
    files = [f for f in os.listdir(work_dir) if f.endswith(".log")]
    for f in files:
        shutil.move(os.path.join(work_dir, f), os.path.join(output_dir, f))
        
def run_task(cfg: DictConfig, run_func: Callable) -> None:
    # Applies optional utilities:
    # - disabling python warnings
    # - prints config
    extras(cfg)

    # execute the task
    try:
        start_time = time.time()
        run_func(cfg)
    except Exception as ex:
        log.exception("")  # save exception to `.log` file
        raise ex
    finally:
        #     ToDo log also:
        #     - Number of CPU cores
        #     - Type of CPUs
        #     - Number of GPU cores
        #     - Type of GPUs
        #     - Number of GPU hours
        current_time_stamp = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
        path = Path(cfg.output_dir, f"exec_time_{current_time_stamp}.log")
        content = (
            f"Execution time: {time.time() - start_time:.2f} seconds "
            f"-- {(time.time() - start_time) / 60:.2f} minutes "
            f"-- {(time.time() - start_time) / 3600:.2f} hours"
        )
        log.info(content)
        save_string_to_file(path, content)  # save task execution time (even if exception occurs)
        close_loggers()  # close loggers (even if exception occurs so multirun won't fail)
        log.info(f"Output dir: {cfg.output_dir}")

        # Temporary solution to Hydra + PL + DDP issue
        # https://github.com/Lightning-AI/lightning/pull/11617#issuecomment-1245842064
        # https://github.com/ashleve/lightning-hydra-template/issues/393
        # problem should be resolved in PL version 1.8.3
        _clean_working_dir_from_subprocesses_output(cfg.work_dir, cfg.output_dir)
