import os
import shutil
import torch
import numpy as np
from tqdm import tqdm 

from src.dataset import give_dataloader
from src.model import classifier

import logger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--name", type=str, default="test")
args = parser.parse_args()

class config:
    # dataset
    split_ratio = {
        "train": 0.9,
        "val": 0.05,
        "test": 0.05
    }
    # model
    size_list = [7, 64, 64, 2]
    # optimizer
    lr = 0.001
    weight_decay = 1e-4
    # scheduler
    step_size = 10
    gamma = 0.5
    # general settings
    epochs = 1
    save_path = "./results/test"
    save_interval = 10
    batch_size = 2048

config.lr = args.lr
config.save_path = os.path.join("./results", args.name)

if os.path.exists(config.save_path):
    shutil.rmtree(config.save_path)
logger.config_logger(output_dir=config.save_path, dist_rank=0, name="LOG")
device = torch.device("cuda:0")

# get dataset
train_loader, val_loader, test_loader = give_dataloader(config)

# get model
model = classifier(
    size_list=config.size_list
).to(device)

# get optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

# get criterion
criterion = torch.nn.CrossEntropyLoss().to(device)

def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, config):
    logger.info("Training phase:")
    train_iter = tqdm(train_loader)
    loss_list = []
    for idx, (data, target) in enumerate(train_iter):
        # data to device
        data = data.to(device)
        target = target.to(device)
        # model forward
        output = model(data)
        # compute loss
        loss = criterion(output, target)
        loss_list.append(loss.item())
        if idx % 10 == 0:
            train_iter.set_description(f"Loss: {loss.item()}")

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    logger.info(f"Loss: {np.sum(loss_list)}")

def validate(loader, model, criterion, config):
    logger.info("Validation phase:")
    data_iter = tqdm(loader)
    loss_list = []
    acc_list = []
    TP, FP, TN, FN = 0, 0, 0, 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_iter):
            # data to device
            data = data.to(device)
            target = target.to(device)
            # model forward
            output = model(data)
            # compute loss
            loss = criterion(output, target)
            loss_list.append(loss.item())
            # compute acc
            pred = torch.max(output, dim=-1)[1]
            acc_list.append((torch.sum(target==pred) / len(target)).item())
            # statistic the confusion matrix
            TP += torch.sum((pred==0) & (target==0))
            FP += torch.sum((pred==1) & (target==0))
            TN += torch.sum((pred==1) & (target==1))
            FN += torch.sum((pred==0) & (target==1))

    logger.info(f"Loss: {np.sum(loss_list)}, Acc: {np.mean(acc_list)}")
    logger.info(f"TP: {TP}/{TP+FP}, FP: {FP}/{TP+FP}, TN: {TN}/{TN+FN}, FN: {FN}/{TN+FN}")

# Start to train
for epoch in range(config.epochs):
    logger.info(f"Epoch {epoch}")

    # train one epoch
    train_one_epoch(train_loader, model, criterion, optimizer, scheduler, config)

    # validation
    validate(val_loader, model, criterion, config)
    
        
    if epoch % config.save_interval == 0:
        to_save_dict = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch
        }
        torch.save(to_save_dict, os.path.join(config.save_path, f"model-{epoch}.ckpt"))

# test
validate(test_loader, model, criterion, config)