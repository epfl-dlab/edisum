from src.models import T5PL
import argparse
from src.tools.wiki_tools import get_prev_cur_wikitext_from_revid_pageid, get_pageid_from_revid
import re
import os
from transformers import AutoTokenizer
from mwedittypes import StructuredEditTypes
from mwedittypes.tokenizer import parse_change_text
from mwedittypes.utils import wikitext_to_plaintext
import requests
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(script_dir, 'models')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="edisum_100", help="Model name (edisum_0, edisum_25, edisum_50 or edisum_100) which defaults to ./models/model_name/ or specific path where model is stored")
    parser.add_argument("--input_text", type=str, default=None, help="Optional exact input to the model, has to be formatted with <old_text>, <new_text> and <sent_sep>")
    parser.add_argument("--diff_link", type=str, default=None, help="Optional link towards edit diff")
    parser.add_argument("-prohibit_node", action='store_true', help="Whether to prohibit generation for edit that has any node edits")
    return parser.parse_args()


def contains_ckpt(dir_path):
    ckpt_files = [f for f in os.listdir(dir_path) if f.endswith('.ckpt')]
    if ckpt_files:
        return ckpt_files
    else:
        return None


def download_from_url(url, save_dir, filename=None):
    if not filename:
        filename = url.split('/')[-1]
    file_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Open the file in write-binary mode and save the content
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded and saved to {file_path}")
        return True
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return False


def get_model_path(model_name_or_path):
    try:
        if os.path.exists(model_name_or_path):
            if os.path.isdir(model_name_or_path):
                ckpt_name = contains_ckpt(model_name_or_path)
                if ckpt_name is not None:
                    return os.path.join(model_name_or_path, ckpt_name)
                else:
                    raise NotImplementedError
            elif model_name_or_path.endswith(".ckpt"):
                return model_name_or_path
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    except:
        if os.path.exists(os.path.join(models_dir, model_name_or_path)) and (contains_ckpt(os.path.join(models_dir, model_name_or_path) is not None)):
            return os.path.join(models_dir, model_name_or_path, f"{model_name_or_path}.ckpt")
        elif model_name_or_path in ["edisum_0", "edisum_25", "edisum_50", "edisum_75", "edisum_100"]:
            url = f"https://huggingface.co/msakota/edisum/resolve/main/{model_name_or_path}.ckpt"
            if download_from_url(url, os.path.join(models_dir, model_name_or_path)):
                return os.path.join(models_dir, model_name_or_path, f"{model_name_or_path}.ckpt")
        raise NotImplementedError("Model path or name provided not found, make sure to specify correct path or name")


def get_revid(url):
    match = re.search(r'oldid=(\d+)', url)
    if match:
        return match.group(1)
    else:
        raise Exception("Invalid link, revision id not found")


def get_input_text(diff_link, prohibit_node):
    rev_id = get_revid(diff_link)
    page_id = get_pageid_from_revid(rev_id)
    prev_text, curr_text = get_prev_cur_wikitext_from_revid_pageid(rev_id, page_id)
    et = StructuredEditTypes(prev_text, curr_text, lang='en')
    structure_diff = et.get_diff()
    level = "Sentence"
    diff = {"prev_texts":[], "cur_texts":[]}
    # we do not support node-edits for now
    if len(structure_diff['node-edits']) > 0 and prohibit_node:
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

    diff['prev_texts'] = sorted(diff['prev_texts'])
    diff['cur_texts'] = sorted(diff['cur_texts'])
    prev_texts_sep = "<sent_sep>".join(diff["prev_texts"])
    cur_texts_sep = "<sent_sep>".join(diff["cur_texts"])
    input_text = f"<old_text>{prev_texts_sep}<new_text>{cur_texts_sep}"
    return input_text


def main():
    args = parse_args()
    model_path = get_model_path(args.model_name_or_path)
    print(f"Model path resolved: {model_path}")
    assert (args.input_text is None) != (args.diff_link is None), "Exactly one of diff_link or input_text must be set (not None)"
    if args.diff_link is not None:
        input_text = get_input_text(args.diff_link, args.prohibit_node)
    else:
        input_text = args.input_text # TODO add formatting

    print(f"Input text resolved: {input_text}")

    if input_text is None:
        print("We do not support node changes yet, and the diff has no extractable textual difference.")
    else:
        model = T5PL.load_from_checkpoint(model_path)
        tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")
        tokenizer_output = tokenizer(
            [input_text],
            return_tensors="pt",
            return_attention_mask=True,
            padding='longest',
            max_length=512,
            truncation=True,
        )
        batch = {}
        for k, v in tokenizer_output.items():
            batch["{}_{}".format('src', k)] = v

        predictions = model._get_predictions_for_batch(batch, None)
        print(f"Model output: {predictions['grouped_decoded_sequences'][0][0]}")


if __name__ == "__main__":
    main()
