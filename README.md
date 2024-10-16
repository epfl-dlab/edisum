# Edisum
This repository contains the PyTorch implementation for the models and experiments in the paper [Edisum: Summarizing and Explaining Wikipedia Edits at Scale](https://arxiv.org/pdf/2404.03428.pdf)

```
@article{šakota2024edisum,
      title={Edisum: Summarizing and Explaining Wikipedia Edits at Scale}, 
      author={Marija Šakota and Isaac Johnson and Guosheng Feng and Robert West},
      journal={arXiv preprint arXiv:2404.03428}
      year={2024}
}
```
Please consider citing our work, if you found the provided resources useful.

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

For downloading the trained models (available on [huggingface](https://huggingface.co/msakota/edisum/tree/main)), run:

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
CHECKPOINT_PATH="./models/edisum_100.ckpt" # specify path to the trained model
RUN_NAME="inference_longt5_100_synth"
python run_inference.py run_name=$RUN_NAME dir=$DATA_DIR checkpoint_path=$CHECKPOINT_PATH +experiment=inference_longt5
```

## 4. Experimenting with custom inputs
### By providing an edit diff link
To test any of the trained models on an arbitrary edit diff link:
```bash
python run_model.py --model_name_or_path edisum_100 --diff_link "https://en.wikipedia.org/w/index.php?title=C/2023_A3_(Tsuchinshan–ATLAS)&diff=prev&oldid=1251441412"
```
Optionally, you can stop the generation in case there are any node changes (as the generated edit might not reflect the changes exhaustively) by adding ```-prohibit_node```. If no ```model_name_or_path``` is provided, the script defaults to ```edisum_100```. You can provide a path towards any .ckpt model, or specify one of the five models from the paper: ```[edisum_0, edisum_25, edisum_50, edisum_75, edisum_100]```, where the number represents percentage of synthetic data in the training dataset.

### By providing a custom input
To test any custom input, which might not necessarily be a real edit:
```bash
python run_model.py --model_name_or_path edisum_100 --input_text <your_input_text>
```
For an optimal performance, the input text should be formatted in the way training data was formatted:

1. Edit diff should be represented by collecting sentences that were altered, added or removed during the edit into two sets: *previous* (belonging to the previous revision of the page) and *current* sentences (belonging to the current revision of the page)
2. Previous sentences should contain each sentence that was removed from the previous revision, and versions of the sentences which were altered from the previous revision
3. New sentences should contain each sentence that was added to the new revision, and versions of the sentences which were altered in the new revision
4. Input is then made concatenating each sentence in *previous sentences*, separating them with ```<sent_sep>```, and adding a prefix ```<old_text>```. Similarly, sentences in *current sentences* are separated with the same ```<sent_sep>``` and prefix ```<new_text>``` is added. Final input is then dervied by concatenating these two repesentations.

Example:

<img width="1369" alt="Screenshot 2024-10-16 at 12 03 49" src="https://github.com/user-attachments/assets/7316a34a-65c0-4a78-b9ee-40233c2d6bc5">

### Jupyter notebook

We also provide a Jupyter notebook for experimentation with custom inputs: [playground.ipynb](https://github.com/epfl-dlab/edisum/blob/main/playground.ipynb)

## License
This project is licensed under the terms of the MIT license.
