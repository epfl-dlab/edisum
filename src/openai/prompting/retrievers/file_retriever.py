"""
Retrieve demonstrations from a file
"""

from src.openai.prompting.retrievers.abstract import Retriever
from typing import List

from src.tools.wiki_tools import get_prev_cur_wikitext_from_revid_pageid
from mwedittypes.tokenizer import parse_change_text
from mwedittypes.utils import wikitext_to_plaintext

import csv
import random

class FileRetriever(Retriever):
    def __init__(self, path_to_edit_samples, num_retrieve=3, level="Sentence", lang="en", random=False, **kwargs):
        super().__init__()
        self.sample_pool = self._get_sample_pool(path_to_edit_samples)
        self.diffs = self._get_summary_diffs(num_retrieve, level=level, lang=lang, random=random, general_sample_revids=kwargs['general_sample_revids'], mannual_pick_revids=kwargs['mannual_pick_revids'])
        
    def _get_sample_pool(self, path_to_edit_samples):
        """Read the file (good samples potential for demonstrations) content into a list (serve for later choose demonstrations)
        """
        pool = []
        with open(path_to_edit_samples, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue # header: edit_summary, page_id, rev_id, url
                pool.append(row)
        return pool
        
    def _get_summary_diffs(self, num_retrieve, level="Sentence", lang="en", random=False, general_sample_revids=None, mannual_pick_revids=None):
        """generate demonstrations (prev_texts+cur_texts+summary) for prompt of openai models
        """
        # choose samples based on different strategy
        if mannual_pick_revids:
            mannual_pick_revids = set([str(x) for x in mannual_pick_revids])
            chosen_samples = [x for x in self.sample_pool if x[2] in mannual_pick_revids]
        elif random and num_retrieve:
            chosen_samples = random.choices(self.sample_pool, k=num_retrieve)
        else:
            chosen_samples = self.sample_pool[:num_retrieve]
        general_sample_revids = set([str(x) for x in general_sample_revids])
        if general_sample_revids:   # get general edit examples
            chosen_samples.extend([x for x in self.sample_pool if x[2] in general_sample_revids])
        
        # use mweditypes to get differences of edit texts from chosen samples
        diffs = []
        for row in chosen_samples: 
            diff = {
                "summary": row[0],
                "prev_texts": [],
                "cur_texts": []
            }
            pageid = int(row[1])
            revid = int(row[2])
            prev_text, cur_text = get_prev_cur_wikitext_from_revid_pageid(revid=revid, pageid=pageid, lang=lang)
            text_changes = parse_change_text(prev_wikitext=wikitext_to_plaintext(prev_text), 
                                curr_wikitext=wikitext_to_plaintext(cur_text),
                                lang=lang, summarize=False)
            
            for sentence, change_type in text_changes[level].items():
                if change_type < 0:
                    diff["prev_texts"].append(sentence)
                else:
                    diff["cur_texts"].append(sentence)
                    
            # try to maintain the order using alphabet order
            diff["cur_texts"] = sorted(diff["cur_texts"])
            diff["prev_texts"] = sorted(diff["prev_texts"])
            diffs.append(diff)
        return diffs
        
    def _process_demonstrations(self, batch_samples):
        """Match the id of demonstration with sample input data id
        # TODO support match different edit types with same type demonstrations edit samples
        
        use demonstrations[{id}] to get the demonstration of sample input with id={id}
        """
        return dict([(s["id"], self.diffs) for s in batch_samples])
    
    def __call__(self, batch_samples):
        return self._process_demonstrations(batch_samples)
    
if __name__ == "__main__":
    r = FileRetriever(path_to_edit_samples="!")