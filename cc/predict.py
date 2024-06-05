import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import argparse
from utils import get_timestamp, preprocess_text
import numpy as np
from model import ModelClassification 
import tqdm
import json

def get_args():
    #rewrite this 
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default='lightning_logs/version_2/checkpoints/last.ckpt')
    parser.add_argument("--input_file", type=str, default='NCKH/public_test_final.json')
    parser.add_argument("--output_file", type=str, default='NCKH/public_test.json')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--debug", action="store_true")
    '''
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--encoder_name", type=str, default="bert-base-uncased")
    '''
    return parser.parse_args()

args = get_args()
tokenizer = AutoTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli')

def tokenize(claim, sent):
    encoded = tokenizer.encode_plus(
        claim,sent,
        add_special_tokens=True,
        truncation=True,
        return_tensors='pt',
        padding='max_length'
    )
    encoded = {key: torch.tensor(tensor).to("cuda") for key, tensor in encoded.items()}
    return encoded

def main():
    submit_file = 'NCKH/public_test.json' #Change it
    label_lookup = {0: "REFUTED",
                    1: "SUPPORTED"}
    #Load model
    model = ModelClassification.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
    model.to("cuda")
    model.eval()
    #Format to json to submit 
    with open(args.input_file, encoding='utf-8') as f:
        data = json.load(f)
    #Predict
    claim_ids = [x for x in data]
    formatted = {claim: {} for claim in claim_ids}
    for id in claim_ids:
        element = data[id]
        if element['verdict'] == 'RELEVANT':
            tokenized = tokenize(preprocess_text(element['claim']), preprocess_text(element['evidence']))
            output = model.predict(tokenized)
            formatted_entry = {
                'verdict' : label_lookup[int(output)],
                'evidence' : element['evidence'] 
            }
        else:
            formatted_entry = {
                'verdict' : "NEI",
                'evidence' : "" 
            }
        formatted[id].update(formatted_entry)

    # Convert to jsonl.
    with open(submit_file, "w", encoding='utf-8') as f:
        json.dump(formatted, f, ensure_ascii=False)

if __name__ == '__main__':
    main()