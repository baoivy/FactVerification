import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import callbacks
import argparse
from pytorch_lightning.strategies import DDPStrategy
from prepare_data import UITRationaleDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from utils import get_timestamp
from model import ModelSelection
import math


def get_parse():
    parser = argparse.ArgumentParser(description="Run training.")
    parser.add_argument("--input_file", type=str, default='data/ise-dsc01-warmup.json')
    parser.add_argument("--encoder_name", type=str, default="bert-base-uncased")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epoch", type=int, default=4)
    parser.add_argument("--frac_warmup", type=float, default=0.1,
                            help="The fraction of training to use for warmup.")
    parser.add_argument("--scheduler_total_epochs", default=None, type=int,
                            help="If given, pass as total # epochs to LR scheduler.")
    parser.add_argument("--gradient_accumulations", default=8, type=int)
    parser.add_argument("--accelerator", type=str, default="ddp")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--starting_checkpoint", type=str, default=None)
    parser.add_argument("--monitor", type=str, default="valid_accuracy")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--debug_mode", type=bool, default=True)
    args = parser.parse_args()
    args.timestamp = get_timestamp()
    
    return args


def main():
    pl.seed_everything(76)
    
    args = get_parse()
    #checkpoint_dir = args.checkpoint
    lr_callback = callbacks.LearningRateMonitor(logging_interval="step")
    gpu_callback = callbacks.DeviceStatsMonitor()
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor=args.monitor, 
        mode="max", 
        save_top_k=1, 
        save_last=True,
    )

    if args.accelerator == "ddp":
        plugins = DDPStrategy(find_unused_parameters=True)
    else:
        plugins = None

    all_data = UITRationaleDataset(args)
    train_ds, valid_ds = train_test_split(all_data, test_size=0.1) 
    train_data = DataLoader(
				train_ds,
				#num_workers=args.num_workers,
				batch_size=args.batch_size,
				#collate_fn=collate_fn,
				shuffle=True,
				#pin_memory=True, 
			)
    valid_data = DataLoader(
				valid_ds,
				#num_workers=args.num_workers,
				batch_size=1,
				#collate_fn=collate_fn,
				shuffle=False,
				#pin_memory=True,
			)

    labels = [x["label"] for x in train_ds]
    class_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=labels
    )
    class_weights=torch.tensor(class_weights,dtype=torch.float)

    steps_per_epoch = math.ceil(
        (len(train_ds) / args.batch_size) / args.gradient_accumulations
    )

    if args.starting_checkpoint is None:
        model = ModelSelection(param=args, steps_per_epoch=steps_per_epoch, loss_fct=class_weights[0]/class_weights[1])
    else:
        model = ModelSelection.load_from_checkpoint(checkpoint_path=args.starting_checkpoint, steps_per_epoch=steps_per_epoch)

    trainer = pl.Trainer(
        callbacks=[lr_callback, checkpoint_callback, gpu_callback], 
        precision="16",
        max_epochs=args.epoch,
        accumulate_grad_batches=args.gradient_accumulations,
        #plugins=plugins
    )
    
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=valid_data)

if __name__ == '__main__':
    main()