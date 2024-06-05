from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import torch
import pytorch_lightning as pl
from utils import preprocess_text
import json
from torch.utils.data import Dataset

class UITRelevantClassiffier(Dataset):
	def __init__(self, param) -> None:
		super().__init__()
		self.tokenizer = AutoTokenizer.from_pretrained(param.encoder_name)
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.path = param.input_file
		self.data = self._load_data()

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, index):
		data = self.data[index]
		return data
	
	def _load_data(self):
		label_lookup = {"REFUTED": 0,
                        "SUPPORTED": 1}
		with open(self.path, encoding='utf-8') as f:
			data = json.load(f)
			res = [ele for ele in data]
			res1 = [data[ids] for ids in res]
		data_loader = []
		for element in res1:
			if element['verdict'] == 'NEI':
				for sent in element['evidence']:
					tokenized = self.tokenize(preprocess_text(element['claim']), preprocess_text(sent))
					entry = {"tokenized" : tokenized,
                      "verdict": 0}
					data_loader.append(entry)
			else:
				tokenized = self.tokenize(preprocess_text(element['claim']), preprocess_text(element['evidence']))
				entry = {"tokenized" : tokenized,
						"verdict": 1}
				data_loader.append(entry)
		
		return data_loader
	
	def tokenize(self, claim, sent):
		encoded = self.tokenizer.encode_plus(
			claim, sent,
			add_special_tokens=True,
			truncation=True,
			#return_tensors='pt',
			padding='max_length',
            max_length = 356
		)
		encoded = {key: torch.tensor(tensor).to(self.device) for key, tensor in encoded.items()}
		return encoded
