#!/bin/bash

url="https://huggingface.co/msakota/edisum/resolve/main"

# Unless a model_directory is passed as an argument, download the files in the `data/models` directory
path_to_models_dir=${1:-"data/models"}
echo "Downloading the models to '$path_to_models_dir'."

#################################
##### Download Pre-Trained Models
#################################
mkdir -p $path_to_models_dir
cd $path_to_models_dir

# Edisum models (trained on the Edisum datasets dataset)
echo "Downloading Edisum[100%]"
wget "$url/edisum_100.ckpt"  
echo "Downloading Edisum[75%]"
wget "$url/edisum_75.ckpt"  
echo "Downloading Edisum[50%]"
wget "$url/edisum_50.ckpt"  
echo "Downloading Edisum[25%]"
wget "$url/edisum_25.ckpt"  
echo "Downloading Edisum[0%]"
wget "$url/edisum_0.ckpt"  

echo "The model checkpoints were downloaded to '$path_to_models_dir'."