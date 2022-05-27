#%%
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt

from src.dataset_pred import give_dataloader
from src.model_pred import RNN
from src.utils import LogMeter, plot_pred_result

import logger
import argparse

#%%
class CONFIG:
    def __init__(self, sample_name: str, lr: float):
        self.sample_name = sample_name
        self.lr = lr
        # dataset
        self.train_idx_ratio, self.input_size, self.output_size = 0.9, 10, 1
        self.data_path = f"./data/{self.sample_name}.ckpt"
        # model
        input_dim_dict = {
            "sample": 7,
            "Y": 10,
            "out_25W": 5,
            "out_12W": 5,
            "gf": 5
        }
        self.input_dim = input_dim_dict[self.sample_name]
        self.hidden_dim, self.model_layer = 64, 3
        thresh_dict = {
            "sample": 8,
            "Y": 10,
            "out_25W": 5,
            "out_12W": 5
        }
        self.dete_thresh = thresh_dict[self.sample_name]
        # optimizer
        self.weight_decay = 1e-4
        # scheduler
        self.step_size = 10
        self.gamma = 0.5
        # general settings
        self.epochs = 1
        self.save_path = f"./results/pred/{self.sample_name}"
        self.save_interval = 10
        self.batch_size = 2048
        # PT_curve
        self.pt_dict = {
            "out_12W": ("2021-10-01 08:00:00", "2021-10-01 18:14:26"),
            "out_25W": ("2022-01-01 08:00:00", "2022-01-01 17:50:55"),
            "sample": ("2021-07-29 08:50:00", "2021-07-29 09:55:00"),
            "Y": ("2020-04-14 07:56:40", "2020-4-30 8:00:00"),
            "gf": ("2022-02-01 00:00:00", "2022-02-03 12:00:00") # FIXME
        }

def initiate(args):
    config = CONFIG(args.sample_name, args.lr)

    logger.config_logger(output_dir=config.save_path, dist_rank=0, name="LOG")
    device = torch.device("cuda:0")
    config.device = device

    # get dataset
    loaders = give_dataloader(config)

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

    return config, loaders, model, optimizer, scheduler, criterion

def train_one_epoch(loaders, model, criterion, optimizer, scheduler, loss_meter, acc_meter, config):
    logger.info("Training phase:")
    train_iter = tqdm(loaders["train"])
    loss_list = []
    for idx, (data, target, labels, timestamp) in enumerate(train_iter):
        # data to device
        data = data.to(config.device)
        target = target.to(config.device)
        # model forward
        output = model(data)
        # compute loss
        loss = criterion(output, target).mean()
        loss_list.append(loss.item())
        loss_meter.append(loss.item())
        if idx % 10 == 0:
            train_iter.set_description(f"Loss: {loss.item()}")

        if idx % 100 == 0:
            validate(loaders["test"], model, criterion, acc_meter, config)

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
            data = data.to(config.device)
            target = target.to(config.device)
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

def validate(loader, model, criterion, acc_meter, config):
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
            data = data.to(config.device)
            target = target.to(config.device)
            labels = labels.to(config.device).view(-1)
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

def run(config, loaders, model, optimizer, scheduler, criterion):
    acc_meter = LogMeter()
    loss_meter = LogMeter()
    # Start to train
    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch}")

        # train one epoch
        train_one_epoch(loaders, model, criterion, optimizer, scheduler, loss_meter, acc_meter, config)

        if epoch % config.save_interval == 0:
            to_save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer,
                "scheduler": scheduler,
                "epoch": epoch
            }
            torch.save(to_save_dict, os.path.join(config.save_path, f"model-{epoch}.ckpt"))
    # Plot
    plot_pred_result(acc_meter, loss_meter, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--name", type=str, default="out_12W", help="")
    args = parser.parse_args()
    
    # Initiate
    config, loaders, model, optimizer, scheduler, criterion = initiate(args)
    # Train and test
    run(config, loaders, model, optimizer, scheduler, criterion)

    ### Generate results
    # get error time
    from src.utils import get_pred_err_time, save_err_time
    time_list = []
    get_pred_err_time(loaders["train"], model, criterion, time_list, config)
    get_pred_err_time(loaders["test"], model, criterion, time_list, config)
    save_err_time(time_list, config)
    # get test performance
    from src.utils import plot_err_pred
    out, trg, labels, timestamp = validate(loaders["test"], model, criterion, LogMeter(), config)
    plot_err_pred(out, trg, labels, config)
    # get pt-curve
    from src.utils import plot_pt_curve
    plot_pt_curve("pred", model, criterion, loaders, config)