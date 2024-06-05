import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import argparse
from utils import get_timestamp, preprocess_text
import numpy as np
from src.rationale_selection.model import ModelSelection
import json
import tqdm
import spacy
nlp = spacy.load('vi_core_news_lg')

def get_args():
    #rewrite this 
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default='lightning_logs/version_0/checkpoints/last.ckpt')
    parser.add_argument("--input_file", type=str, default='NCKH/data/ise-dsc01-train_1.json')
    parser.add_argument("--output_file", type=str, default='rationale_output.json')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    return parser.parse_args()

args = get_args()
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-large')

def segmentSentence(corpus):
	list_sent = nlp(corpus)
	list_sent = [str(sent).strip() for sent in list_sent.sents if str(sent).strip() != ""]
	return list_sent

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
    label_lookup = {0: "REFUTED",
                    1: "NEI",
                    2: "SUPPORTED"}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelSelection.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
    model.to(device)
    model.eval()
    #Format to json to submit 
    with open(args.input_file, encoding='utf-8') as f:
        data = json.load(f)
    #Predict
    claim_ids = [x for x in data]
    formatted = {claim: {} for claim in claim_ids}
    for id in claim_ids:
        element = data[id]
        res = []
        if element['verdict'] == "NEI" :
            list_sent = segmentSentence(element['context'])
            for sent in list_sent:
                tokenized = tokenize(preprocess_text(element['claim']), preprocess_text(sent))
                output = model.predict(tokenized)
                res.append(int(output[0]))
            context_idx = np.argsort(res)[-2:]
            context = np.take(list_sent, sorted(context_idx))
        formatted_entry = {
            'claim' : element['claim'],
            'verdict': element['verdict'],
            'evidence': element['evidence'] if element['verdict'] != "NEI" else list(context)
        }
        formatted[id].update(formatted_entry)

    # Convert to jsonl.
    submit_file_js = 'NCKH/sentence_extract.json'
    with open(submit_file_js, "w", encoding='utf-8') as f:
        json.dump(formatted, f, ensure_ascii=False)

if __name__ == '__main__':
    main()