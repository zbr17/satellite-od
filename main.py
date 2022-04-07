from email.policy import default
import os
import shutil
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tqdm import tqdm 

from src.dataset import TimeSeries
from src.model import Transformer

import logger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--name", type=str, default="test")
args = parser.parse_args()

class config:
    # dataset
    train_idx_ratio = 0.9
    input_length = 100
    output_length = 10
    input_size = 7
    # model
    feature_size = 64
    num_layers = 20
    nhead = 8
    dropout = 0.1
    # optimizer
    lr = 0.001
    weight_decay = 1e-4
    # scheduler
    step_size = 10
    gamma = 0.5
    # general settings
    epochs = 100
    save_path = "./results/trial1"
    save_interval = 10
    batch_size = 256

config.lr = args.lr
config.num_layers = args.num_layers
config.save_path = os.path.join("./results", args.name)

if os.path.exists(config.save_path):
    shutil.rmtree(config.save_path)
logger.config_logger(output_dir=config.save_path, dist_rank=0, name="LOG")
device = torch.device("cuda:0")

# get dataset
sample_set = TimeSeries(data_path="./data/data.ckpt", input_size=config.input_length, output_size=config.output_length, data_mask=None)
train_idx_end = int(config.train_idx_ratio*len(sample_set))
train_idx_mask = torch.arange(train_idx_end)
val_idx_mask = torch.arange(train_idx_end+1, len(sample_set))
train_set = TimeSeries(data_path="./data/data.ckpt", input_size=config.input_length, output_size=config.output_length, data_mask=train_idx_mask)
val_set = TimeSeries(data_path="./data/data.ckpt", input_size=config.input_length, output_size=config.output_length, data_mask=val_idx_mask)

# get dataloader
train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=8)

# get model
model = Transformer(
    input_size=config.input_size,
    feature_size=config.feature_size,
    num_layers=config.num_layers,
    nhead=config.nhead,
    dropout=config.dropout
).to(device)

# get optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

# get criterion
criterion = torch.nn.MSELoss().to(device)

# Start to train
for epoch in range(config.epochs):
    logger.info(f"Epoch {epoch}")

    loss_list = []
    train_iter = tqdm(train_loader)
    for idx, (data, target) in enumerate(train_iter):
        # data to device
        data = data.to(device).float().permute(1,0,2)
        target = target.to(device).float().permute(1,0,2)
        # model forward
        output = model(data)
        # compute loss
        loss = criterion(output[-config.output_length:], target)
        loss_list.append(loss.item())
        if idx % 10 == 0:
            train_iter.set_description(f"Loss: {loss.item()}")

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    logger.info(f"Loss: {np.sum(loss_list)}")
    if epoch % config.save_interval == 0:
        to_save_dict = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch
        }
        torch.save(to_save_dict, os.path.join(config.save_path, f"model-{epoch}.ckpt"))