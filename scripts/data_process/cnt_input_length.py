from src.tools.data_tools import count_diff_tokens
from tqdm import tqdm
import pandas as pd

filtered_wikidump_folder = "../data/wikidumps_filtered"
filtered_wikidump = ["2023-01.enwiki.2022-01.csv", "2023-01.enwiki.2022-02.csv", "2023-01.enwiki.2022-03.csv", \
                     "2023-01.enwiki.2022-04.csv", "2023-01.enwiki.2022-06.csv", \
                     "2023-01.enwiki.2022-07.csv", "2023-01.enwiki.2022-08.csv", \
                     "2023-01.enwiki.2022-10.csv", "2023-01.enwiki.2022-11.csv", "2023-01.enwiki.2022-12.csv",]
out_path = "/scratch/gfeng/Edit_summary_generation/data/statistics/input_length.csv"

for dump in filtered_wikidump:
    print(f"processing dump {dump}")
    dump_path = filtered_wikidump_folder + "/" + dump
    df = pd.read_csv(dump_path)
    count_diff_tokens(df, out_path)

