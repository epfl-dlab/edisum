from transformers import BertTokenizer
from mwedittypes import StructuredEditTypes
from tqdm import tqdm
from src.tools.wiki_tools import get_wikitext_from_revid

import os
import csv
import sys

def count_diff_tokens(df, out_csv, tokenizer=BertTokenizer.from_pretrained('bert-base-cased')):
    """count token number in sentence and paragraph level seperately for diffs (insert and remove) of a edit

    Args:
        df (pandas.DataFrame): dataframe contais a list of revision_id (cur revision) and revision_parent_id (old revision)
        out_csv (str): .csv path for the output recording file (if not exists will create one)
        tokenizer (PreTrainedTokenizer, optional): pretrained tokenizer from huggingface. Defaults to BertTokenizer.from_pretrained('bert-base-cased').

    Returns:
        pandas.DataFrame: contains information of 
    """
    if not os.path.exists(out_csv):
        csv_file = open(out_csv, "w")
        csv_writer = csv.writer(csv_file, delimiter='\t')
        # write the head
        csv_writer.writerow(["revision_id", "revision_parent_id", "sum_sent_insert_cnt", "sum_para_insert_cnt", "sum_sent_remove_cnt", "sum_para_remove_cnt", \
            "max_sent_insert_cnt", "max_sent_remove_cnt", "max_para_insert_cnt", "max_para_remove_cnt", "sum_node_insert_cnt", "sum_node_remove_cnt", \
            "max_node_insert_cnt", "max_node_remove_cnt", "error"])
    else:
        csv_file = open(out_csv, "a")
        csv_writer = csv.writer(csv_file, delimiter='\t')
    
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            old_text = get_wikitext_from_revid(rev_id=row["revision_parent_id"])
            new_text  = get_wikitext_from_revid(rev_id=row["revision_id"])
        except Exception as e:
            print(e)
            csv_writer.writerow([row["revision_id"], row["revision_parent_id"], 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 0, "get_wikitext_error"])
            continue
         
        et = StructuredEditTypes(old_text, new_text, lang='en', timeout=True)
        try:
            diff = et.get_diff()
        except Exception as e: 
            print("============old text============")
            print(old_text)
            print("============new text============")
            print(new_text)
            print("============error msg============")
            print(e)
            csv_writer.writerow([row["revision_id"], row["revision_parent_id"], 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 0, "diff_error"])
            continue
            
        sum_sent_insert_cnt = 0
        sum_para_insert_cnt = 0
        sum_sent_remove_cnt = 0
        sum_para_remove_cnt = 0
        max_sent_insert_cnt = 0
        max_sent_remove_cnt = 0
        max_para_insert_cnt = 0
        max_para_remove_cnt = 0
        if len(diff['text-edits']) > 0:
            for edit in diff['text-edits']:
                if edit.type == "Sentence":
                    token_cnt = len(tokenizer.tokenize(edit.text))
                    if edit.edittype == "insert":
                        sum_sent_insert_cnt += token_cnt
                        if token_cnt > max_sent_insert_cnt:
                            max_sent_insert_cnt = token_cnt
                    elif edit.edittype == "remove":
                        sum_sent_remove_cnt += token_cnt
                        if token_cnt > max_sent_remove_cnt:
                            max_sent_remove_cnt = token_cnt                        
                elif edit.type == "Paragraph":
                    token_cnt = len(tokenizer.tokenize(edit.text))
                    if edit.edittype == "insert":
                        sum_para_insert_cnt += token_cnt
                        if token_cnt > max_para_insert_cnt:
                            max_para_insert_cnt = token_cnt                            
                    elif edit.edittype == "remove":
                        sum_para_remove_cnt += token_cnt
                        if token_cnt > max_para_remove_cnt:
                            max_para_remove_cnt = token_cnt
        
        """Example of node edit changes
        [NodeEdit(type='Wikilink', edittype='insert', section='0: Lede', name='landscape painter',
                         changes=[('title', None, 'landscape painter')]),
        NodeEdit(type='Template', edittype='change', section='0: Lede', name='Short description',
                         changes=[('parameter', ('1', 'Austrian painter'), ('1', 'Austrian [[landscape painter]]'))]),
        NodeEdit(type='Template', edittype='insert', section='0: Lede', name='Short description', 
                         changes=[('name', None, 'Short description'), ('parameter', None, ('1', 'American labor relations specialist'))])]
        """
        sum_node_insert_cnt = 0
        sum_node_remove_cnt = 0
        max_node_insert_cnt = 0
        max_node_remove_cnt = 0
        if len(diff['node-edits']) > 0:
            for node in diff['node-edits']:
                for change in node.changes:
                    remove_text = change[1]
                    if isinstance(change[1], tuple):  # in case of changing parameter
                        remove_text = change[1][1]
                    insert_text = change[2]
                    if isinstance(change[2], tuple):  # in case of changing parameter
                        insert_text = change[2][1]
                    remove_cnt = len(tokenizer.tokenize(str(remove_text))) if remove_text else 0
                    insert_cnt = len(tokenizer.tokenize(str(insert_text))) if insert_text else 0
                    sum_node_insert_cnt += insert_cnt
                    sum_node_remove_cnt += remove_cnt
                    if insert_cnt > max_node_insert_cnt:
                        max_node_insert_cnt = insert_cnt
                    if remove_cnt > max_node_remove_cnt:
                        max_node_remove_cnt = remove_cnt
                    
        csv_writer.writerow([row["revision_id"], row["revision_parent_id"], sum_sent_insert_cnt, sum_para_insert_cnt, sum_sent_remove_cnt, sum_para_remove_cnt, \
            max_sent_insert_cnt, max_sent_remove_cnt, max_para_insert_cnt, max_para_remove_cnt, sum_node_insert_cnt, sum_node_remove_cnt, max_node_insert_cnt, max_node_remove_cnt, "null"])
        
    csv_file.close()  
    return df