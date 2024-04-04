#!/bin/bash

url="https://huggingface.co/datasets/msakota/edisum_dataset/resolve/main"

# Unless a data_directory is passed as an argument, download the files in the `data` directory
path_to_data_dir=${1:-"data"}
echo "Downloading the data to '$path_to_data_dir'."

# Create it if it does not exist
mkdir -p $path_to_data_dir
cd $path_to_data_dir


###################
# Download Datasets
###################

# Synthetic data
dataset_dir="100_perc_synth_data"
mkdir $dataset_dir && cd $dataset_dir
for split in "val" "test" "train"
do
    wget "$url/$dataset_dir/$split.csv"
done
cd ..


dataset_dir="75_perc_synth_data"
mkdir $dataset_dir && cd $dataset_dir
for split in "val" "test" "train"
do
    wget "$url/$dataset_dir/$split.csv"
done
cd ..

dataset_dir="50_perc_synth_data"
mkdir $dataset_dir && cd $dataset_dir
for split in "val" "test" "train"
do
    wget "$url/$dataset_dir/$split.csv"
done
cd ..


dataset_dir="25_perc_synth_data"
mkdir $dataset_dir && cd $dataset_dir
for split in "val" "test" "train"
do
    wget "$url/$dataset_dir/$split.csv"
done
cd ..

# Wikipedia processed data
dataset_dir="filtered-min30-enwiki-08-2023-data"
mkdir $dataset_dir && cd $dataset_dir
for split in "val" "test" "train"
do
    wget "$url/$dataset_dir/$split.csv"
done
cd ..


echo "The data was download at '$path_to_data_dir'."