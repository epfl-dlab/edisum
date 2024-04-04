import csv
import os
from tqdm import tqdm
from mwedittypes import SimpleEditTypes

from src.tools.wiki_tools import get_prev_cur_wikitext_from_revid_pageid, get_pageid_from_revid

valid_types = ["Section", "Paragraph", "Sentence", "Word", "Punctuation", "Whitespace", "ExternalLink"]

def process_filtered_wikidump(paths, out_path):
    out_head = ['edit_summary','page_id','rev_id','edit_types','url']
    csv_writer = csv.DictWriter(open(out_path, 'w'), fieldnames=out_head)
        
    invalid_cnt = 0
    for path in paths:
        print(f"processing {path}")
        csv_reader = csv.DictReader(open(path, 'r'))
        for i, item in enumerate(tqdm(csv_reader)):
            if i == 0:
                csv_writer.writeheader()
                continue
            new_item = process_item(item)
            if not new_item:
                invalid_cnt += 1
                continue
            csv_writer.writerow(new_item)
    
def get_valid_edittype(revid, pageid):
    """Check if the revision contains invalid edit types
    """
    prev_wikitext, curr_wikitext = get_prev_cur_wikitext_from_revid_pageid(revid, pageid)
    et = SimpleEditTypes(prev_wikitext, curr_wikitext, lang='en')
    diff = et.get_diff()
    for edit_type in diff.keys():
        if edit_type not in valid_types:
            return None
    return ",".join(diff.keys())

def process_item(item):
    page_id = get_pageid_from_revid(item['revision_id'])
    
    edit_types = get_valid_edittype(revid=item['revision_id'], pageid=page_id)
    if not edit_types:
        return None
    
    new_item = {}
    new_item['edit_summary'] = item['event_comment']
    new_item['rev_id'] = item['revision_id']
    new_item['page_id'] = page_id
    new_item['url'] = "https://en.wikipedia.org/w/index.php?&diff=prev&oldid={}".format(new_item['rev_id'])
    new_item['edit_types'] = edit_types
    
    return new_item
        

if __name__ == '__main__':
    filtered_data_folder = "../data/wikidumps_filtered/"
    dates_to_be_processed = ['2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12']
    paths = [filtered_data_folder + f"2023-01.enwiki.{x}.csv" for x in dates_to_be_processed]
    
    out_path = "./data/wikidump_processed/2022_filter_plaintext_all.csv"
    process_filtered_wikidump(paths=paths, out_path=out_path)