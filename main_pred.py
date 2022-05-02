#%%
import os
import shutil
import torch
import numpy as np
from tqdm import tqdm 

from src.dataset_pred import give_dataloader
from src.model_pred import RNN

import logger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args(args=[])

#%%
class config:
    # dataset
    train_idx_ratio = 0.9
    input_size = 10
    output_size = 1
    sample_name = "Y" # sample / Y / out_25W / out_12W
    data_path = f"./data/{sample_name}.ckpt"
    # model
    input_dim_dict = {
        "sample": 7,
        "Y": 10,
        "out_25W": 5,
        "out_12W": 5
    }
    thresh_dict = {
        "sample": 8,
        "Y": 10,
        "out_25W": 5,
        "out_12W": 5
    }
    input_dim = input_dim_dict[sample_name]
    hidden_dim = 64
    model_layer = 2
    dete_thresh = thresh_dict[sample_name]
    # optimizer
    lr = 0.001
    weight_decay = 1e-4
    # scheduler
    step_size = 10
    gamma = 0.5
    # general settings
    epochs = 3
    save_path = f"./results/pred/{sample_name}"
    save_interval = 10
    batch_size = 2048

#%%
class LogMeter:
    def __init__(self):
        self.value_list = []
    
    def append(self, v):
        self.value_list.append(v)
    
    def extend(self, v_list):
        self.value_list.extend(v_list)
    
    def avg(self):
        return np.mean(self.value_list)

acc_meter = LogMeter()
loss_meter = LogMeter()

config.lr = args.lr

if os.path.exists(config.save_path):
    shutil.rmtree(config.save_path)
logger.config_logger(output_dir=config.save_path, dist_rank=0, name="LOG")
device = torch.device("cuda:0")

# get dataset
train_loader, test_loader = give_dataloader(config)

# get model
model = RNN(
    input_size=config.input_dim,
    hidden_size=config.hidden_dim,
    num_layers=config.model_layer,
    out_size=config.output_size
).to(device)

# get optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

# get criterion
criterion = torch.nn.MSELoss(reduction="none").to(device)

def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, config):
    logger.info("Training phase:")
    train_iter = tqdm(train_loader)
    loss_list = []
    for idx, (data, target, labels, timestamp) in enumerate(train_iter):
        # data to device
        data = data.to(device)
        target = target.to(device)
        # model forward
        output = model(data)
        # compute loss
        loss = criterion(output, target).mean()
        loss_list.append(loss.item())
        loss_meter.append(loss.item())
        if idx % 10 == 0:
            train_iter.set_description(f"Loss: {loss.item()}")

        if idx % 100 == 0:
            validate(test_loader, model, criterion, config)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    logger.info(f"Loss: {np.sum(loss_list)}")

def compute_loss(loader, model, criterion, config):
    data_iter = tqdm(loader)
    loss_list = []
    with torch.no_grad():
        for idx, (data, target, labels, timestamp) in enumerate(data_iter):
            # data to device
            data = data.to(device)
            target = target.to(device)
            # model forward
            output = model(data)
            # compute loss
            loss = criterion(output, target).sum(dim=-1).sum(dim=-1)
            loss_list.extend(loss.cpu().detach().numpy().tolist())
    return loss_list

def compute_thresh(loss_list):
    # compute threshold
    thresh = np.mean(sorted(loss_list)[-10000:]) * 1.1
    return thresh

def validate(loader, model, criterion, config):
    logger.info("Validation phase:")
    data_iter = tqdm(loader)
    loss_list = []
    acc_list = []
    TP, FP, TN, FN = 0, 0, 0, 0
    with torch.no_grad():
        for idx, (data, target, labels, timestamp) in enumerate(data_iter):
            # data to device
            data = data.to(device)
            target = target.to(device)
            labels = labels.to(device).view(-1)
            # model forward
            output = model(data)
            # compute loss
            loss = criterion(output, target).sum(dim=-1).sum(dim=-1)
            loss_list.append(loss.sum().item())
            # compute acc
            pred = (loss > config.dete_thresh).long()
            acc_list.append((torch.sum(labels==pred) / len(labels)).item())
            # statistic the confusion matrix
            TP += torch.sum((pred==0) & (labels==0))
            FP += torch.sum((pred==1) & (labels==0))
            TN += torch.sum((pred==1) & (labels==1))
            FN += torch.sum((pred==0) & (labels==1))

    acc_meter.append(np.mean(acc_list))
    logger.info(f"Loss: {np.sum(loss_list)}, Acc: {np.mean(acc_list)}")
    logger.info(f"TP: {TP}/{TP+FP}, FP: {FP}/{TP+FP}, TN: {TN}/{TN+FN}, FN: {FN}/{TN+FN}")

#%%
# Start to train
for epoch in range(config.epochs):
    logger.info(f"Epoch {epoch}")

    # train one epoch
    train_one_epoch(train_loader, model, criterion, optimizer, scheduler, config)
        
    if epoch % config.save_interval == 0:
        to_save_dict = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch
        }
        torch.save(to_save_dict, os.path.join(config.save_path, f"model-{epoch}.ckpt"))

#%%
# Compute the threshold
train_loss_list = compute_loss(train_loader, model, criterion, config)
print("dete_thresh", compute_thresh(train_loss_list))

#%%
# Get time
time_list = []
def get_err_time(loader, model, criterion, config):
    data_iter = tqdm(loader)
    loss_list = []
    with torch.no_grad():
        for idx, (data, target, labels, timestamp) in enumerate(data_iter):
            # data to device
            data = data.to(device)
            target = target.to(device)
            labels = labels.to(device).view(-1)
            timestamp = timestamp.view(-1)
            # model forward
            output = model(data)
            # compute loss
            loss = criterion(output, target).sum(dim=-1).sum(dim=-1)
            loss_list.append(loss.sum().item())
            # compute acc
            pred = (loss > config.dete_thresh).long()
            time_list.extend(timestamp[pred==1].cpu().numpy().tolist())

get_err_time(train_loader, model, criterion, config)
get_err_time(test_loader, model, criterion, config)

import pandas as pd
time_list = sorted(time_list)
for i in range(10):
    print(pd.to_datetime(time_list[i]*1000000))

with open(f"{os.path.join(config.save_path, 'time_info.txt')}", mode="w", encoding="utf-8") as f:
    time_str = []
    for i in range(len(time_list)):
        time_str.append(str(pd.to_datetime(time_list[i]*1000000)) + "\n")
    f.writelines(time_str)

#%%
# Test set
validate(test_loader, model, criterion, config)

#%%
# Draw figures
import matplotlib.pyplot as plt 
print(acc_meter.value_list)
plt.figure()
plt.grid()
plt.plot(np.arange(len(acc_meter.value_list)), np.array(acc_meter.value_list))
plt.xlabel("Iteration")
plt.ylim(0, 1.1)
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.savefig(f"{os.path.join(config.save_path, 'acc.png')}")

plt.figure()
plt.grid()
plt.plot(np.arange(len(loss_meter.value_list[:200])), loss_meter.value_list[:200])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.savefig(f"{os.path.join(config.save_path, 'loss.png')}")