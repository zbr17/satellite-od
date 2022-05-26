#%%
from cProfile import label
import os
import shutil
from turtle import color
import torch
import numpy as np
from tqdm import tqdm 

from src.dataset_pred import give_dataloader
from src.model_pred import RNN

import logger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--name", type=str, default="sample")
args = parser.parse_args()

#%%
class config:
    # dataset
    train_idx_ratio = 0.9
    input_size = 10
    output_size = 1
    sample_name = args.name # sample / Y / out_25W / out_12W
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
    model_layer = 3
    dete_thresh = thresh_dict[sample_name]
    # optimizer
    lr = 0.001
    weight_decay = 1e-4
    # scheduler
    step_size = 10
    gamma = 0.5
    # general settings
    epochs = 1
    save_path = f"./results/pred/{sample_name}"
    save_interval = 10
    batch_size = 2048
    # PT_curve
    pt_dict = {
        "out_12W": ("2021-10-01 08:00:00", "2021-10-01 18:14:26"),
        "out_25W": ("2022-01-01 08:00:00", "2022-01-01 17:50:55"),
        "sample": ("2021-07-29 08:50:00", "2021-07-29 09:55:00"),
        "Y": ("2020-04-14 07:56:40", "2020-4-30 8:00:00")
    }
print(config.sample_name)

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
    output_list = []
    target_list = []
    label_list = []
    timestamp_list = []
    TP, FP, TN, FN = 0, 0, 0, 0
    with torch.no_grad():
        for idx, (data, target, labels, timestamp) in enumerate(data_iter):
            # data to device
            data = data.to(device)
            target = target.to(device)
            labels = labels.to(device).view(-1)
            # model forward
            output = model(data)
            output_list.append(output.cpu())
            target_list.append(target.cpu())
            label_list.append(labels.cpu())
            timestamp_list.append(timestamp)
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
    return (
        torch.cat(output_list, dim=0),
        torch.cat(target_list, dim=0),
        torch.cat(label_list, dim=0),
        torch.cat(timestamp_list, dim=0)
    )

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
out, trg, labels, timestamp = validate(test_loader, model, criterion, config)
sep_idx = torch.where(labels==0)[0][-1]
num_vis = 1000
import matplotlib.pyplot as plt

for i in range(config.input_dim):
    plt.figure()
    plt.grid()
    s, e = sep_idx - num_vis, sep_idx
    plt.plot(np.arange(s, e), out[s:e, :, i].flatten().numpy(), color="g", alpha=0.5, linestyle='--', label="pred")
    plt.plot(np.arange(s, e), trg[s:e, :, i].flatten().numpy(), color="r", alpha=0.5, linestyle='--', label="truth")
    s, e = sep_idx + 1, sep_idx + num_vis
    plt.plot(np.arange(s, e), out[s:e, :, i].flatten().numpy(), color="g", label="pred-abnormal")
    plt.plot(np.arange(s, e), trg[s:e, :, i].flatten().numpy(), color="r", label="truth-abnormal")
    plt.legend()
    plt.title(f"params{i} prediction")
    plt.savefig(os.path.join(config.save_path, f"params_pred_{i}.png"))

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

#%%
# Compute PT-curve and save timestamp
data_iter = tqdm(test_loader)
pred_list = []
trg_list = []
time_list = []
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
        # compute acc
        pred = (loss > config.dete_thresh).long()
        pred_list.append(pred)
        trg_list.append(labels)
        time_list.append(timestamp)
pred_list = torch.cat(pred_list).cpu()
trg_list = torch.cat(trg_list).cpu()
time_list = torch.cat(time_list).cpu()

sort_idx = torch.sort(time_list)[1]
pred_list = pred_list[sort_idx]
trg_list = trg_list[sort_idx]
time_list = time_list[sort_idx]

#%%
import pandas as pd
sn = config.sample_name
def str2date(input):
    df = pd.DataFrame({'date': [input]})
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].astype('int64')
    return float(df.values / 1000000)
start_time = str2date(config.pt_dict[sn][0])
end_time = str2date(config.pt_dict[sn][1])
time_mask = (time_list > start_time) & (time_list < end_time)

time_list = time_list[time_mask]
trg_list = trg_list[time_mask]
pred_list = pred_list[time_mask]

#%%
import matplotlib.pyplot as plt
acc_list = (pred_list == trg_list).float()
acc_list = torch.cumsum(acc_list, dim=0) / (1+torch.arange(len(acc_list))).float()
save_path = f"./data/raw_data/_pt_info/pred/{sn}"
print(save_path)
os.makedirs(save_path, exist_ok=True)
def date2str(input):
    return str(pd.to_datetime(1000000 * input))
xtick_list = np.array([date2str(item)[:-7] for item in time_list.numpy().tolist()])

step = len(time_list)
if step > 6:
    step = int(step / 6)
print(f"Num: {step}")
select_tick = np.arange(0, len(time_list), step)
plt.figure()
plt.grid()
plt.plot(time_list.numpy(), acc_list.numpy())
plt.xlabel("Timestamp")
plt.ylabel("Precision")
plt.xticks(time_list[select_tick], xtick_list[select_tick])
plt.xticks(rotation=30)
plt.title("Precision-Timestamp Curve")
plt.savefig(os.path.join(save_path, "pt_curve.png"), bbox_inches='tight')

error_time_list = time_list[pred_list != trg_list]
print(len(error_time_list))
with open(f"{os.path.join(save_path, 'time_info.txt')}", mode="w", encoding="utf-8") as f:
    time_str = []
    for i in range(len(error_time_list)):
        time_str.append(str(pd.to_datetime(int(error_time_list[i]*1000000))) + "\n")
    f.writelines(time_str)