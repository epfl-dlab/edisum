import csv
import random
import os


# Define the file paths
input_file = "/scratch/gfeng/Edit_summary_generation/data/output/wikidump_dataset_cache/202209_filter_plaintext_all175k.csv" # "/scratch/gfeng/Edit_summary_generation/data/output/data_generation/gen_summary_turbo_100_samples_fix_dms.csv"
output_folder = "/scratch/gfeng/Edit_summary_generation/data/edit_summary_data/raw_summary_50k-t90-v5-t5"
limit=50000

train_file = os.path.join(output_folder, "train.csv")
val_file = os.path.join(output_folder, "val.csv")
test_file = os.path.join(output_folder, "test.csv")

# Define the split percentages
train_percent = 0.9
val_percent = 0.05
test_percent = 0.05

# Open the input file and create the output files
with open(input_file, 'r') as f_input, \
     open(train_file, 'w', newline='') as f_train, \
     open(val_file, 'w', newline='') as f_val, \
     open(test_file, 'w', newline='') as f_test:

    # Create the CSV reader and writer objects
    reader = csv.reader(f_input)
    writer_train = csv.writer(f_train)
    writer_val = csv.writer(f_val)
    writer_test = csv.writer(f_test)

    # Skip the header row
    header = next(reader)
    writer_train.writerow(header)
    writer_val.writerow(header)
    writer_test.writerow(header)

    # Split the data randomly into train, validation, and test sets
    rows = list(reader)[:limit]
    random.shuffle(rows)
    num_rows = len(rows)
    num_train = int(num_rows * train_percent)
    num_val = int(num_rows * val_percent)
    num_test = num_rows - num_train - num_val

    for i in range(num_train):
        writer_train.writerow(rows[i])

    for i in range(num_train, num_train + num_val):
        writer_val.writerow(rows[i])

    for i in range(num_train + num_val, num_rows):
        writer_test.writerow(rows[i])
