#%%
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt
import datetime
import json
from typing import Union

from src.dataset_pred import give_dataloader
from src.model_pred import give_model
from src.utils import LogMeter, plot_pred_result

import logger
import tbwriter
import argparse

#%%
class CONFIG:
    def __init__(self, args: dict):
        self.args = args
        for k, v in args.items():
            setattr(self, k, v)

        self.num_params: int = ...
        self.pt_range: list = ...
        self.choose_params_list: Union[str, list] = ...
        self.abnormal_flag: Union[int, list] = ...

    def update(self, args: dict):
        for k, v in args.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # dataset
        self.train_idx_ratio, self.input_size, self.output_size = 0.8, 10, 1
        self.data_path = f"./data/{self.sample_name}.ckpt"
        # model
        if self.choose_params_list == "all":
            self.input_dim = self.num_params
        else:
            assert isinstance(self.choose_params_list, list)
            self.input_dim = len(self.choose_params_list)

        self.dete_thresh = 0.2
        # optimizer
        self.weight_decay = 0.0001
        # scheduler
        self.step_size = 10
        self.gamma = 0.5
        # general settings
        self.epochs = 5
        if not self.search:
            self.save_path = f"./results/pred/{self.model_name}/{self.sample_name}"
        else:
            self.save_path = f"./results_serach/pred/{self.model_name}/{self.sample_name}-{datetime.datetime.now().strftime('%Y%m%d%H%M')}"
        self.save_interval = 10
        self.batch_size = 2048
        # PT_curve
        self.pt_range = self.pt_range

def initiate(args):
    config = CONFIG(args)
    with open(os.path.join(args["raw_data"], args["sample_name"], "config.json"), mode="r", encoding="utf-8") as f:
        config_data = json.load(f)
        config.update(config_data)

    logger.config_logger(output_dir=config.save_path, dist_rank=0, name="LOG")
    tbwriter.config(output_dir=config.save_path, dist_rank=0)
    device = torch.device("cuda:0")
    config.device = device

    # get dataset
    loaders = give_dataloader(config)

    # get model
    model = give_model(config).to(device)

    # get optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    # get criterion
    criterion = torch.nn.MSELoss(reduction="none").to(device)

    return config, loaders, model, optimizer, scheduler, criterion

def train_one_epoch(loaders, model, criterion, optimizer, scheduler, loss_meter, acc_meter, config):
    model.train()
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
            # compute thresh
            thresh_loss_list, labels = compute_loss(loaders["test"], model, criterion, config)
            config.dete_thresh = compute_thresh(thresh_loss_list, labels)
            validate(loaders["test"], model, criterion, acc_meter, config)
            model.train()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    logger.info(f"Loss: {np.sum(loss_list)}")

def compute_loss(loader, model, criterion, config):
    data_iter = tqdm(loader)
    loss_list, labels_list = [], []
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
            labels_list.extend(labels.cpu().squeeze().numpy().tolist())
    return torch.tensor(loss_list), torch.tensor(labels_list)

def compute_thresh(loss_list, labels):
    loss_sorted, idx_sorted = torch.sort(loss_list)
    label_sorted = labels[idx_sorted]
    pos_cum = torch.cumsum(label_sorted == 0, dim=0)
    neg_cum = torch.cumsum((label_sorted == 1).flip(0), dim=0).flip(0)
    cum = pos_cum + neg_cum
    thresh = loss_sorted[cum.argmax()]
    bestacc = cum.max() / len(labels)
    logger.info(f"thresh: {thresh.item()}, bestacc: {bestacc.item()}")

    return thresh.item()

def validate(loader, model, criterion, acc_meter, config):
    model.eval()
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

    # Tensorboard
    tbwriter.add_hparams(
        hparam_dict=config.args,
        metric_dict={"hparam/acc": max(acc_meter.value_list)}
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sample_name", type=str, default="gf", help="")
    parser.add_argument("--raw_data", type=str, default="./data/raw_data")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--model_name", type=str, default="transformer")
    parser.add_argument("--model_layer", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--search", action="store_true", default=False)
    args = parser.parse_args()

    config_args = {}
    for k in dir(args):
        if not k.startswith("_") and not k.endswith("__"):
            config_args[k] = getattr(args, k)
    
    # Initiate
    config, loaders, model, optimizer, scheduler, criterion = initiate(config_args)
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