"""
process osaacj collected data

data link: https://analytics.wikimedia.org/published/datasets/one-off/isaacj/edit_summaries/edit_summaries_enwiki_2022.tsv.gz
Associated README: https://analytics.wikimedia.org/published/datasets/one-off/isaacj/edit_summaries/README.md 
Edit summary abbreviations: https://en.wikipedia.org/wiki/Wikipedia:Edit_summary_legend
"""

from transformers import BertTokenizer
from mwedittypes import StructuredEditTypes
from tqdm import tqdm
from src.tools.wiki_tools import get_wikitext_from_revid
import pandas as pd
import gzip

def process_isaacj_data(path, edittypes_path, out_path):
    df = pd.read_csv(path, sep="\t")
    print(f"Getting valid plaintext id sets from : {edittypes_path}")
    valid_revids_to_edittypes = get_valid_revids(edittypes_path)
    print(f"filter data in : {path}")
    df = filter_isaacj_data(df, plain_text_revid_to_edittypes=valid_revids_to_edittypes)
    df.to_csv(out_path, compression='gzip', sep='\t')

def filter_isaacj_data(df, plain_text_revid_to_edittypes=None):
    """Filter out unuseful data: revert/reverted, empty-summary
    TODO: add other filter rules (e.g., user edit frequency)

    Args:
        df (pandas.DataFrame): dataframe object or trunk of dataframe
    """
    
    df = df.loc[df["edit_summary"].notnull() & (~df["was_revert"]) & (~df["was_reverted"])]
    if plain_text_revid_to_edittypes:
        df = df.loc[df["rev_id"].isin(plain_text_revid_to_edittypes)]
        df["edit_types"] = df["rev_id"].map(lambda x: plain_text_revid_to_edittypes[x])
    return df

def get_valid_revids(file_path):
    """get valid pageids by:
    1. Discard all the samples that include anything that isn't change of plain text
    """
    valid_types = ["Section", "Paragraph", "Sentence", "Word", "Punctuation", "Whitespace", "ExternalLink"]
    plain_text_revid_to_edittypes = dict()
    keep_cnt = 0
    all_cnt = 0
    with gzip.open(file_path, 'rt') as fin:
        assert next(fin).strip().split('\t') == ['revision_id', 'et_to_str(edit_types)']
        for line in fin:
            all_cnt += 1
            rev_id_str, edit_types_str = line.strip().split('\t')
            rev_id = int(rev_id_str)
            contain_bad_types = False
            if edit_types_str != '""':
                for et in edit_types_str.split(';'):
                    node, action, count = et.split(',')
                    # discard row if edit type is not belong to plain_text_edit_types
                    if node not in valid_types:
                        contain_bad_types = True
                        break
                if contain_bad_types:
                    continue
                else:
                    plain_text_revid_to_edittypes[rev_id] = ",".join(set([x.split(',')[0] for x in edit_types_str.split(';')]))
                    keep_cnt += 1
    print(f"From {all_cnt} revisions, {all_cnt-keep_cnt} were filterred out, {keep_cnt} ({(keep_cnt/all_cnt)*100:.2f}%) were keeped")
    return plain_text_revid_to_edittypes
            
    
                    
if __name__ == '__main__':
    DATA_PATH = "../data/isaacj_data/edit_summaries_enwiki_2022.tsv.gz"
    EDIT_TYPES_PATH = "../data/isaacj_data/enwiki-2022-09-edittypes.tsv.gz"
    OUT_PATH = "../data/isaacj_data/edit_summaries_enwiki_2022_processed.tsv.gz"
    process_isaacj_data(DATA_PATH, EDIT_TYPES_PATH, OUT_PATH)