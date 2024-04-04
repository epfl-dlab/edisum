"""
Filter useful fields (columns) of wikidump tsv files. Also do some statiscal counting if needed.
"""

import pandas as pd
import requests
import csv
import os
from bs4 import BeautifulSoup
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# set the ENV variables here
DATES = None  # if None, process all dates in the DATA_PATH
DATA_PATH = "/scratch/gfeng/Edit_summary_generation/data/wikidumps"
OUTPUT_PATH = "/scratch/gfeng/Edit_summary_generation/data/wikidumps_filtered"
STATISTIC_PATH = "/scratch/gfeng/Edit_summary_generation/data/wikidump_statistic.csv"  # if None, do not perform statistics when filtering data


if STATISTIC_PATH and not os.path.exists(STATISTIC_PATH):
    # write head
    with open(STATISTIC_PATH, "w") as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(["date_name", "edit_cnt", "summary_cnt", "revert_cnt", "reverted_cnt", "deleted_cnt"])

# extract the field description table from wiki
URL = "https://wikitech.wikimedia.org/wiki/Analytics/Data_Lake/Edits/Mediawiki_history_dumps#Technical_Documentation"
table_id = 'wikitable'
res = requests.get(URL).text
soup = BeautifulSoup(res, 'html.parser')
table = soup.find_all('table', attrs={'class': table_id})
df = pd.read_html(str(table[1]))


# process the data
data_folder_path = Path(DATA_PATH)
for date_data in data_folder_path.iterdir():
    date_name = date_data.name.replace(".tsv.bz2", "")
    if f'{date_name}.csv' in os.listdir(OUTPUT_PATH):
        print(f"Date already processed: {date_data.stem}, jump this one")
        continue
    print(f"process {date_data.stem}")
    if "wiki" not in date_data.stem:  # check file name
        continue
    if DATES and date_data.split(".")[2] not in DATES:
        continue
    
    # C engine may raise some error when parsing certain rows
    try:
        raw_data = pd.read_csv(str(date_data), names=df[0]['Field name'], sep='\t', dtype=str, error_bad_lines=False, warn_bad_lines=True)
    except:
        raw_data = pd.read_csv(str(date_data), names=df[0]['Field name'], sep='\t', dtype=str, error_bad_lines=False, warn_bad_lines=True, engine="python")
    
    # perform statistcs if needed
    if STATISTIC_PATH:
        edit_cnt = len(raw_data)
        summary_cnt = len(raw_data.loc[raw_data['event_comment'].str.len() > 2])
        revert_cnt = raw_data['revision_is_identity_revert'].dropna().map({'true':1, 'false':0}).sum()
        reverted_cnt = raw_data['revision_is_identity_reverted'].dropna().map({'true':1, 'false':0}).sum()
        deleted_cnt = raw_data['revision_is_deleted_by_page_deletion'].dropna().map({'true':1, 'false':0}).sum()
        with open(STATISTIC_PATH, "a+") as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow([date_name, edit_cnt, summary_cnt, revert_cnt, reverted_cnt, deleted_cnt])
    
    # filter useful data
    raw_data = raw_data.loc[(raw_data['event_entity'] == 'revision') & (raw_data['event_type'] == 'create')]
    raw_data['event_comment'] = raw_data['event_comment'].str.strip()
    raw_data = raw_data.loc[raw_data['event_comment'].str.len() > 2]
    raw_data = raw_data.loc[(raw_data['revision_is_identity_reverted'] == 'false') &
                            (raw_data['revision_is_identity_revert'] == 'false') &
                            (raw_data['revision_is_deleted_by_page_deletion'] == 'false')]

    # only keep old and new page id
    raw_data = raw_data[['event_timestamp', 'event_comment', 'event_user_id', 'revision_parent_id', 'revision_id']]
    raw_data.dropna(inplace=True)
    raw_data.to_csv(f'{OUTPUT_PATH}/{date_name}.csv', index=False)