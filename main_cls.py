#%%
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt

from src.dataset_cls import give_dataloader
from src.model_cls import classifier
from src.utils import LogMeter, plot_cls_result

import logger
import argparse

#%%
class CONFIG:
    def __init__(self, sample_name: str, lr: float):
        self.sample_name = sample_name
        self.lr = lr
        # dataset
        self.split_ratio = {"train": 0.9,"val": 0.05,"test": 0.05}
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
        self.size_list = [self.input_dim, 64, 64, 2]
        # optimizer
        self.weight_decay = 1e-4
        # scheduler
        self.step_size = 10
        self.gamma = 0.5
        # general settings
        self.epochs = 1
        self.save_path = f"./results/cls/{self.sample_name}"
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
    model = classifier(
        size_list=config.size_list
    ).to(device)

    # get optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    # get criterion
    criterion = torch.nn.CrossEntropyLoss().to(device)

    return config, loaders, model, optimizer, scheduler, criterion

def train_one_epoch(loaders, model, criterion, optimizer, scheduler, loss_meter, acc_meter, config):
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

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    logger.info(f"Loss: {np.sum(loss_list)}")

def validate(loader, model, criterion, acc_meter, config):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sample_name", type=str, default="gf", help="")
    args = parser.parse_args()

    # Initiate
    config, loaders, model, optimizer, scheduler, criterion = initiate(args)
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