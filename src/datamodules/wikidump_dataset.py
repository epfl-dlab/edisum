"""
Dataset for pepararing wiki dump data for openai data generation
"""
from src.datamodules.abstract import AbstractDataset
from torch.utils.data import DataLoader
from mwedittypes import StructuredEditTypes
from src.tools.wiki_tools import get_prev_cur_wikitext_from_revid_pageid
from mwedittypes.tokenizer import parse_change_text
from mwedittypes.utils import wikitext_to_plaintext
from src.tools.general_tools import batched
import timeout_decorator
from datasets import load_dataset
from tqdm import tqdm
import csv
import os
import ast

class WikiEditDataset(AbstractDataset):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.cache_path = kwargs['cache_path']
        #if not self.cache_path:
        self.data = self._load_processed_data(kwargs['data_path'],
                                    kwargs['demonstration_path'],
                                    kwargs['limit'])
        # else:
        #     if not os.path.exists(self.cache_path) or kwargs['extend_cache']:
        #         self._cache_data(kwargs['data_path'],
        #                          kwargs['cache_path'],
        #                          kwargs['extend_cache'],
        #                          kwargs['demonstration_path'],
        #                          kwargs['limit'],
        #                          kwargs["diff_level"],
        #                          kwargs["lang"])
        #     self.data = load_dataset("csv", data_files=self.cache_path)['train']
            
        self.dataloader_params = kwargs["dataloader_params"]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def _cache_data(self, data_path, cache_path, extend_cache=False, demonstration_path=None, limit=-1, diff_level='Sentence', lang='en'):
        """Cache data to csv before loading it as a Huggingface dataset
        
        Args:
            data_path: preprocessed csv file from wikidump that contains wikidump data, 
            cache_path: path for caching the data
            whose header contains at least 'page_id', 'rev_id' and 'edit_summary'
            demonstration_path (optional): path for demonstration file containing good samples for ChatGBT demonstration. 
                            if not None, samples from this file will not be chosen for generate the summary
        """
        print(f"Caching dataset into {cache_path} ... ")
        
        dms_revids = set()  # revision ids that belong to demonstrations, these should be ignored in the data (because already used as demonstrations)
        if demonstration_path:
            with open(demonstration_path, "r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                revid_idx = None
                for i, row in enumerate(csv_reader):
                    if i == 0:
                        revid_idx = row.index('rev_id')
                        continue # jump header
                    dms_revids.add(int(row[revid_idx]))
        existing_cache_revids = set()
        if os.path.exists(cache_path):
            print(f"extending existing cache {cache_path}")
            with open(cache_path, "r") as cache_file:
                csv_reader = csv.DictReader(cache_file, delimiter=',')
                next(csv_reader) # skips the first row of the CSV file.
                for row in csv_reader:
                    existing_cache_revids.add(int(row['rev_id']))
            print(f"finished scanning existing revids")
        with open(data_path) as f:
            total_rows = sum(1 for _ in f)
        with open(data_path, "r") as csv_file:
            cache_mode = "a" if extend_cache else "w"
            with open(cache_path, cache_mode) as cache_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                csv_writer = csv.DictWriter(cache_file, delimiter=',', fieldnames=["id", "page_id", "rev_id", "summary", "prev_texts", "cur_texts"])
                if cache_mode == 'w':
                    csv_writer.writeheader()
                pageid_idx, revid_idx, summary_idx = None, None, None
                cnt = 0
                cnt_skip = 0
                cnt_mw_skip = 0
                for i, row in enumerate(tqdm(csv_reader, total=total_rows)):
                    if i == 0:
                        pageid_idx = row.index('page_id')
                        revid_idx = row.index('rev_id')
                        summary_idx = row.index('edit_summary')
                        continue # jump header
                    pageid = int(row[pageid_idx])
                    revid = int(row[revid_idx])
                    if i <= len(existing_cache_revids):
                        continue
                    if (revid in dms_revids) or (revid in existing_cache_revids):
                        cnt_skip += 1
                        continue
                    try:
                        diff = self._get_diff(pageid, revid, diff_level, lang)
                    except:
                        print(f"time out querying revid {revid}, skip")
                        continue
                    # print(diff)
                    if not diff:
                        cnt_mw_skip += 1
                        continue
                    
                    # try to maintain the order using alphabet order
                    diff['prev_texts'] = sorted(diff['prev_texts'])
                    diff['cur_texts'] = sorted(diff['cur_texts'])
                    csv_writer.writerow({"id": i,
                                        "page_id": row[pageid_idx],
                                        "rev_id": row[revid_idx],
                                        "summary": row[summary_idx],
                                        "prev_texts": diff['prev_texts'],
                                        "cur_texts": diff['cur_texts']})
                    cnt += 1
                    if limit > 0 and cnt >= limit:
                        break
        print(f"Caching dataset into {cache_path} ... Finished!")
        print(f"skiped {cnt_skip} samples, because used as demonstration of prompt or already exist in cache")
        print(f"skiped {cnt_mw_skip} samples, mweditypes return no {diff_level}-level difference or they contain node edits")

    def _load_processed_data(self, data_path, demonstration_path=None, limit=-1):
        print("Preparing dataset ... ")
        data = []

        dms_revids = set()  # revision ids that belong to demonstrations, these should be ignored in the data (because already used as demonstrations)
        if demonstration_path:
            with open(demonstration_path, "r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                revid_idx = None
                for i, row in enumerate(csv_reader):
                    if i == 0:
                        revid_idx = row.index('rev_id')
                        continue  # jump header
                    dms_revids.add(row[revid_idx])

        with open(data_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            prev_texts_idx, cur_texts_idx, summary_idx, revid_idx, page = None, None, None, None, None
            cnt = 0
            for i, row in enumerate(csv_reader):
                if i == 0:
                    prev_texts_idx = row.index('prev_texts')
                    cur_texts_idx = row.index('cur_texts')
                    revid_idx = row.index("revision_id")
                    pageid_idx = row.index('page_id')
                    summary_idx = row.index('summary')
                    continue  # jump header
                revid = int(row[revid_idx])
                prev_texts = ast.literal_eval(row[prev_texts_idx])
                cur_texts = ast.literal_eval(row[cur_texts_idx])
                if revid in dms_revids:
                    print(f"skip data with revision id {revid}, because its used as demonstration of prompt")
                    continue

                # try to maintain the order using alphabet order
                prev_texts = sorted(prev_texts)
                cur_texts = sorted(cur_texts)
                data.append({"id": i,
                             "page_id": row[pageid_idx],
                             "rev_id": row[revid_idx],
                             "summary": row[summary_idx],
                             "prev_texts": prev_texts,
                             "cur_texts": cur_texts})
                cnt += 1
                if limit > 0 and cnt >= limit:
                    break
        print(f"Loaded {len(data)} datapoints")
        return data



    def _load_data(self, data_path, demonstration_path=None, limit=-1, diff_level='Sentence', lang='en'):
        """Load data from csv file

        Args:
            data_path: preprocessed csv file from wikidump that contains wikidump data, 
            whose header contains at least 'page_id', 'rev_id' and 'edit_summary'
            demonstration_path (optional): path for demonstration file containing good samples for ChatGBT demonstration. 
                            if not None, samples from this file will not be chosen for generate the summary
        """
        print("Preparing dataset ... ")
        data = []
        
        dms_revids = set()  # revision ids that belong to demonstrations, these should be ignored in the data (because already used as demonstrations)
        if demonstration_path:
            with open(demonstration_path, "r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                revid_idx = None
                for i, row in enumerate(csv_reader):
                    if i == 0:
                        revid_idx = row.index('rev_id')
                        continue # jump header
                    dms_revids.add(row[revid_idx])
                    
        with open(data_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            pageid_idx, revid_idx, summary_idx = None, None, None
            cnt = 0
            for i, row in enumerate(csv_reader):
                if i == 0:
                    pageid_idx = row.index('page_id')
                    revid_idx = row.index('rev_id')
                    summary_idx = row.index('edit_summary')
                    continue # jump header
                pageid = int(row[pageid_idx])
                revid = int(row[revid_idx])
                if revid in dms_revids:
                    print(f"skip data with revision id {revid}, because its used as demonstration of prompt")
                    continue
                try:
                    diff = self._get_diff(pageid, revid, diff_level, lang)
                except:
                    print(f"time out querying revid {revid}, skip")
                    continue
                if not diff:
                    print(f"skip data with revision id {revid}, mweditypes return no {diff_level}-level difference or it contains node edits")
                    continue
                
                # try to maintain the order using alphabet order
                diff['prev_texts'] = sorted(diff['prev_texts'])
                diff['cur_texts'] = sorted(diff['cur_texts'])
                data.append({"id": i,
                             "page_id": row[pageid_idx],
                             "rev_id": row[revid_idx],
                             "summary": row[summary_idx],
                             "prev_texts": diff['prev_texts'],
                             "cur_texts": diff['cur_texts']})
                cnt += 1
                if limit > 0 and cnt >= limit:
                    break
        return data
    
    @timeout_decorator.timeout(20)
    def _get_diff(self, pageid, revid, level="Sentence", lang="en"):
        diff = {"prev_texts": [],
                "cur_texts": []}
        prev_text, cur_text = get_prev_cur_wikitext_from_revid_pageid(revid=revid, pageid=pageid, lang=lang)
        # HACK this one can only deal with text edit, can not filter out node ones
        # text_changes = parse_change_text(prev_wikitext=wikitext_to_plaintext(prev_text), 
        #                     curr_wikitext=wikitext_to_plaintext(cur_text),
        #                     lang=lang, summarize=False)
        et = StructuredEditTypes(prev_text, cur_text, lang=lang)
        structure_diff = et.get_diff()
        
        # HACK we do not support node-edits for now
        if len(structure_diff['node-edits']) > 0:
            return None
        
        for item in structure_diff['text-edits']:
            if item.type != level:
                continue
            if item.edittype == 'remove':
                diff["prev_texts"].append(item.text)
            else:
                diff["cur_texts"].append(item.text)
                
        if len(diff["prev_texts"]) + len(diff["cur_texts"]) == 0:
            return None
            
        return diff
    
    def dataloader(self):
        return batched(self.data, batch_size=self.dataloader_params['batch_size'])