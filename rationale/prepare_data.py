from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import torch
import pytorch_lightning as pl
from utils import preprocess_text
import json
from torch.utils.data import Dataset
import random
import spacy
nlp = spacy.load('vi_core_news_lg')

class UITRationaleDataset(Dataset):
	def __init__(self, param) -> None:
		super().__init__()
		self.tokenizer = AutoTokenizer.from_pretrained(param.encoder_name)
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.path = param.input_file
		self.data = self._load_data()

	def __len__(self):
		return len(self.data)
	
	def _get_label_(self):
		return [0,1]
	
	def __getitem__(self, item):
		data = self.data[item]
		return data
	
	def _load_data(self):
		label_lookup = {"REFUTED": 0,
                        "NEI": 1,
                        "SUPPORTED": 2}
		with open(self.path, encoding="utf-8") as f:
			data = json.load(f)
			res = [ele for ele in data]
			res1 = [data[ids] for ids in res]
		data_loader = []
		for element in res1:
			if(element["verdict"] == "NEI"):
				list_sent = self.segmentSentence(element['context'])
				nei_sents = random.sample(list_sent, k=2)
				for nei_sent in nei_sents:
					tokenized = self.tokenize(preprocess_text(element['claim']), preprocess_text(nei_sent))
					entry = {
						"tokenized": tokenized,
						"label" : 0
					}
					data_loader.append(entry)
			else:
				tokenized = self.tokenize(preprocess_text(element['claim']), preprocess_text(element['evidence']))
				entry = {
					"tokenized": tokenized,
					"label" : 1
				}
				data_loader.append(entry)
		return data_loader
	
	def segmentSentence(self, corpus):
		list_sent = nlp(corpus)
		list_sent = [str(sent).strip() for sent in list_sent.sents if str(sent).strip() != ""]
		return list_sent
	
	def tokenize(self, claim, sent):
		encoded = self.tokenizer.encode_plus(
			claim, sent,
			add_special_tokens=True,
			truncation=True,
			padding='max_length',
		)
		encoded = {key: torch.tensor(tensor).to(self.device) for key, tensor in encoded.items()}
		return encoded