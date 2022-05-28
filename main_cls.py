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

from src.dataset_cls import give_dataloader
from src.model_cls import classifier
from src.utils import LogMeter, plot_cls_result

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
        
    def update(self, args: dict):
        for k, v in args.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # dataset
        self.split_ratio = {"train": 0.9,"val": 0.05,"test": 0.05}
        self.data_path = f"./data/{self.sample_name}.ckpt"
        # model
        if self.choose_params_list == "all":
            self.input_dim = self.num_params
        else:
            assert isinstance(self.choose_params_list, list)
            self.input_dim = len(self.choose_params_list)
        
        self.size_list = [self.input_dim, 64, 64, 2]
        # optimizer
        self.weight_decay = 1e-4
        # scheduler
        self.step_size = 10
        self.gamma = 0.5
        # general settings
        self.epochs = 1
        if not self.search:
            self.save_path = f"./results/cls/{self.sample_name}"
        else:
            self.save_path = f"./results_serach/cls/{self.sample_name}-{datetime.datetime.now().strftime('%Y%m%d%H%M')}"
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
    model = classifier(
        size_list=config.size_list
    ).to(device)

    # get optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    # get criterion
    criterion = torch.nn.CrossEntropyLoss().to(device)

    return config, loaders, model, optimizer, scheduler, criterion

def train_one_epoch(loaders, model, criterion, optimizer, scheduler, loss_meter, acc_meter, config):
    model.train()
    logger.info("Training phase:")
    train_iter = tqdm(loaders["train"])
    loss_list = []
    for idx, (data, target, timestamp) in enumerate(train_iter):
        # data to device
        data = data.to(config.device)
        target = target.to(config.device)
        # model forward
        output = model(data)
        # compute loss
        loss = criterion(output, target)
        loss_list.append(loss.item())
        loss_meter.append(loss.item())
        if idx % 10 == 0:
            train_iter.set_description(f"Loss: {loss.item()}")

        if idx % 100 == 0:
            validate(loaders["val"], model, criterion, acc_meter, config)
            model.train()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    logger.info(f"Loss: {np.sum(loss_list)}")

def validate(loader, model, criterion, acc_meter, config):
    model.eval()
    logger.info("Validation phase:")
    data_iter = tqdm(loader)
    loss_list = []
    acc_list = []
    TP, FP, TN, FN = 0, 0, 0, 0
    with torch.no_grad():
        for idx, (data, target, timestamp) in enumerate(data_iter):
            # data to device
            data = data.to(config.device)
            target = target.to(config.device)
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

    acc_meter.append(np.mean(acc_list))
    logger.info(f"Loss: {np.sum(loss_list)}, Acc: {np.mean(acc_list)}")
    logger.info(f"TP: {TP}/{TP+FP}, FP: {FP}/{TP+FP}, TN: {TN}/{TN+FN}, FN: {FN}/{TN+FN}")

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
    plot_cls_result(acc_meter, loss_meter, config)

    # Tensorboard
    tbwriter.add_hparams(
        hparam_dict=config.args,
        metric_dict={"hparam/acc": max(acc_meter.value_list)}
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sample_name", type=str, default="Y", help="")
    parser.add_argument("--raw_data", type=str, default="./data/raw_data")
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
    from src.utils import get_cls_err_time, save_err_time
    time_list = []
    get_cls_err_time(loaders["train"], model, criterion, time_list, config)
    get_cls_err_time(loaders["val"], model, criterion, time_list, config)
    get_cls_err_time(loaders["test"], model, criterion, time_list, config)
    save_err_time(time_list, config)
    # get test performance
    validate(loaders["test"], model, criterion, LogMeter(), config)
    # get pt-curve
    from src.utils import plot_pt_curve
    plot_pt_curve("cls", model, criterion, loaders, config)