'''
This is a source code from NewbieML team
'''
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import lightning_getattr
from feedforward import FeedForward

AUTH_TOKEN = "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"

class ModelClassification(pl.LightningModule):
    def __init__(self, param, loss_weight, steps_per_epoch) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(param.encoder_name, use_auth_token=AUTH_TOKEN)
        self.model = AutoModel.from_pretrained(param.encoder_name, self.config, use_auth_token=AUTH_TOKEN)
        self.tokenizer = AutoTokenizer.from_pretrained(param.encoder_name, use_auth_token=AUTH_TOKEN)
        #Defnition MLP 
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        activations = [nn.GELU(), nn.Identity()]
        dropouts = [self.dropout.p, 0]
        self.label_classifier = FeedForward(
            input_dim=self.config.hidden_size,
            num_layers=2,
            hidden_dims=[self.config.hidden_size, 2],
            activations=activations,
            dropout=dropouts)
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = torch.nn.CrossEntropyLoss(weight=loss_weight)
        #Others
        self.lr = param.lr
        self.param = param
        self.steps_per_epoch = steps_per_epoch


    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        if self.param.debug_mode:
            return optimizer
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.lr,
            pct_start=0.05,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            anneal_strategy="linear",
        )

        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]

    #Expand purpose
    def expand_embeddings(self):
        if (
            len(self.tokenizer)
            != self.roberta.roberta.embeddings.word_embeddings.weight.shape[0]
        ):
            print("Expanding embedding size")
            self.roberta.resize_token_embeddings(len(self.tokenizer))

    def forward(self, tokenized):
        output = self.model(**tokenized)
        classifier = self.dropout(output.pooler_output)
        logits = self.label_classifier(classifier)
        res = self.softmax(logits)
        return res
        
    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        input, label = train_batch['tokenized'], train_batch['verdict']
        output = self.forward(input)
        loss = self.loss(output, label)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, valid_batch, batch_idx) -> STEP_OUTPUT | None:
        input, label = valid_batch['tokenized'], valid_batch['verdict']
        output = self.forward(input)
        loss = self.loss(output, label)
        accuracy = torch.sum(torch.argmax(output) == label).item() / (len(label) * 1.0)
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict(self, eval_batch):
        with torch.no_grad():
            res = self.forward(eval_batch)
            res = torch.argmax(res)
        return res
    

