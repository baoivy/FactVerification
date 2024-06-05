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
from feedforward import FeedForward

AUTH_TOKEN = "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"

class ModelSelection(pl.LightningModule):
    def __init__(self, param, steps_per_epoch, loss_fct) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(param.encoder_name, use_auth_token=AUTH_TOKEN)
        self.model = AutoModel.from_pretrained(param.encoder_name, output_hidden_states=True, use_auth_token=AUTH_TOKEN)
        self.tokenizer = AutoTokenizer.from_pretrained(param.encoder_name, use_auth_token=AUTH_TOKEN)
        #Defnition MLP 
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        activations = [nn.GELU(), nn.Identity()]
        dropouts = [self.dropout.p, 0]
        self.rationale_classifier = FeedForward(
            input_dim=4*self.config.hidden_size,
            num_layers=2,
            hidden_dims=[4*self.config.hidden_size, 1],
            activations=activations,
            dropout=dropouts)
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=loss_fct)
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
        outputs = self.model(**tokenized)
        cls_concat = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
        classifier = self.dropout(cls_concat)
        res = self.rationale_classifier(classifier)
        return res
    
    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        input, label = train_batch['tokenized'], train_batch['label']
        output = self.forward(input)
        loss = self.loss(output.view(-1), label.float())
        self.log("train_loss", loss.mean(), on_epoch=True, prog_bar=True, logger=True)
        return loss.mean()

    
    def validation_step(self, valid_batch, batch_idx) -> STEP_OUTPUT | None:
        input, label = valid_batch['tokenized'], valid_batch['label']
        output = self.forward(input)
        loss = self.loss(output.view(-1), label.float())
        rationale_probs = torch.sigmoid(output.view(-1)).detach()
        predicted_rationales = (rationale_probs >= 0.5).to(torch.int32)
        accuracy = torch.sum(predicted_rationales == label).item() / (len(label) * 1.0)
        self.log("valid_loss", loss.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True)
        return loss.mean()

    def _invoke_metrics(self, pred, batch, fold):
        """
        Invoke metrics for a single step of train / validation / test.
        `batch` is gold, `pred` is prediction, `fold` specifies the fold.
        """
        assert fold in ["train", "valid", "test"]
    
    def predict(self, eval_batch):
        output = []
        with torch.no_grad():
            res = self.forward(eval_batch)
            output.append(res.view(-1))
        return output
    