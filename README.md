# Edisum
This repository contains the PyTorch implementation for the models and experiments in the paper "Edisum: Summarizing and Explaining Wikipedia Edits at Scale"

## 1. Setup
Start by cloning the repository:
```bash
git clone https://github.com/epfl-dlab/edisum.git
```

We recommend creating a new [conda](https://docs.conda.io/en/latest/) virtual environment as follows:
```bash
conda env create -f environment.yml
```
This command also installs all the necessary packages.

## 2. Downloading data and models
The data is available on [huggingface](https://huggingface.co/datasets/msakota/edisum_dataset) and can be loaded with:
```bash
from datasets import load_dataset
dataset = load_dataset("msakota/edisum_dataset")
```
Alternatively, to download the collected data for the experiments, run:

```bash
bash ./download_data.sh
```

For downloading the trained models (available on [hugginface]()), run:

```bash
bash ./download_models.sh
```

## 3. Usage
### Training
To train a model from scratch on the desired data, run:
```bash
DATA_DIR="./data/100_perc_synth_data/" # specify a directory where training data is located
RUN_NAME="train_longt5_100_synth"
python run_train.py run_name=$RUN_NAME dir=$DATA_DIR +experiment=finetune_longt5
```
### Inference
To run inference on a trained model:

```bash
DATA_DIR="./data/100_perc_synth_data/" # specify a directory where training data is located
CHECKPOINT_PATH="./models/edisum_100.ckpt"
RUN_NAME="inference_longt5_100_synth"
python run_inference.py run_name=$RUN_NAME dir=$DATA_DIR checkpoint_path=$CHECKPOINT_PATH +experiment=inference_longt5
```

## License
This project is licensed under the terms of the MIT license.
