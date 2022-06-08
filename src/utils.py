import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt 

import torch

class LogMeter:
    def __init__(self):
        self.value_list = []
    
    def append(self, v):
        self.value_list.append(v)
    
    def extend(self, v_list):
        self.value_list.extend(v_list)
    
    def avg(self):
        return np.mean(self.value_list)

def get_cls_err_time(loader, model, criterion, time_list, config):
    data_iter = tqdm(loader)
    loss_list = []
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
            time_list.extend(timestamp[pred==1].cpu().numpy().tolist())

def get_pred_err_time(loader, model, criterion, time_list, config):
    data_iter = tqdm(loader)
    loss_list = []
    with torch.no_grad():
        for idx, (data, target, labels, timestamp) in enumerate(data_iter):
            # data to device
            data = data.to(config.device)
            target = target.to(config.device)
            labels = labels.to(config.device).view(-1)
            timestamp = timestamp.view(-1)
            # model forward
            output = model(data)
            # compute loss
            loss = criterion(output, target).sum(dim=-1).sum(dim=-1)
            loss_list.append(loss.sum().item())
            # compute acc
            pred = (loss > config.dete_thresh).long()
            time_list.extend(timestamp[pred==1].cpu().numpy().tolist())
    return time_list

def save_err_time(time_list, config):
    time_list = sorted(time_list)
    for i in range(10):
        print(pd.to_datetime(time_list[i]*1000000))

    with open(f"{os.path.join(config.save_path, 'time_info.txt')}", mode="w", encoding="utf-8") as f:
        time_str = []
        for i in range(len(time_list)):
            time_str.append(str(pd.to_datetime(time_list[i]*1000000)) + "\n")
        f.writelines(time_str)

def plot_err_pred(out, trg, labels, config):
    sep_idx = torch.where(labels==1)[0][0] - 1
    neg_len = len(torch.where(labels==1)[0])
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000

    for i in range(config.input_dim):
        plt.figure()
        plt.grid()
        s, e = max(0, sep_idx-neg_len), sep_idx
        plt.plot(np.arange(s, e), out[s:e, :, i].flatten().numpy(), color="g", alpha=0.5, linestyle='--', label="pred")
        plt.plot(np.arange(s, e), trg[s:e, :, i].flatten().numpy(), color="r", alpha=0.5, linestyle='--', label="truth")
        s, e = sep_idx + 1, sep_idx + neg_len
        plt.plot(np.arange(s, e), out[s:e, :, i].flatten().numpy(), color="g", label="pred-abnormal")
        plt.plot(np.arange(s, e), trg[s:e, :, i].flatten().numpy(), color="r", label="truth-abnormal")
        plt.ylim([-1,1])
        plt.legend()
        plt.title(f"params{i} prediction")
        plt.savefig(os.path.join(config.save_path, f"params_pred_{i}.pdf"), dpi=300, format="pdf")

def plot_cls_result(acc_meter, loss_meter, config):
    plt.figure()
    plt.grid()
    plt.plot(acc_meter.value_list[:100])
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.savefig(f"{os.path.join(config.save_path, 'acc.pdf')}", dpi=300, format="pdf")

    plt.figure()
    plt.grid()
    plt.plot(loss_meter.value_list[:200])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.savefig(f"{os.path.join(config.save_path, 'loss.pdf')}", dpi=300, format="pdf")

def plot_pred_result(acc_meter, loss_meter, config):
    plt.figure()
    plt.grid()
    plt.plot(np.arange(len(acc_meter.value_list)), np.array(acc_meter.value_list))
    plt.xlabel("Iteration")
    plt.ylim(0, 1.1)
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.savefig(f"{os.path.join(config.save_path, 'acc.pdf')}", dpi=300, format="pdf")

    plt.figure()
    plt.grid()
    plt.plot(np.arange(len(loss_meter.value_list[:200])), loss_meter.value_list[:200])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.savefig(f"{os.path.join(config.save_path, 'loss.pdf')}", dpi=300, format="pdf")

def plot_pt_curve(type: str, model, criterion, loaders, config):
    device = config.device
    pred_list = []
    trg_list = []
    time_list = []

    if type == "cls":
        data_iter = tqdm(loaders["train"])
        with torch.no_grad():
            for idx, (data, target, timestamp) in enumerate(data_iter):
                # data to device
                data = data.to(device)
                target = target.to(device)
                # model forward
                output = model(data)
                # compute acc
                pred = torch.max(output, dim=-1)[1]
                pred_list.append(pred)
                trg_list.append(target)
                time_list.append(timestamp)
    elif type == "pred":
        data_iter = tqdm(loaders["test"])
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
    else:
        raise KeyError()

    pred_list = torch.cat(pred_list).cpu()
    trg_list = torch.cat(trg_list).cpu()
    time_list = torch.cat(time_list).cpu()

    sort_idx = torch.sort(time_list)[1]
    pred_list = pred_list[sort_idx]
    trg_list = trg_list[sort_idx]
    time_list = time_list[sort_idx]

    def str2date(input):
        df = pd.DataFrame({'date': [input]})
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].astype('int64')
        return float(df.values / 1000000)
    start_time = str2date(config.pt_range[0])
    end_time = str2date(config.pt_range[1])
    time_mask = (time_list > start_time) & (time_list < end_time)

    time_list = time_list[time_mask]
    trg_list = trg_list[time_mask]
    pred_list = pred_list[time_mask]

    acc_list = (pred_list == trg_list).float()
    acc_list = torch.cumsum(acc_list, dim=0) / (1+torch.arange(len(acc_list))).float()
    save_path = os.path.join(config.save_path, "_pt_info")
    os.makedirs(save_path, exist_ok=True)
    def date2str(input):
        return str(pd.to_datetime(1000000 * input))
    xtick_list = np.array([date2str(item)[:-7] for item in time_list.numpy().tolist()])

    step = len(time_list)
    if step > 6:
        step = int(step / 6)
    print(f"Num: {step}")
    select_tick = np.arange(0, len(time_list), step)
    time_list = torch.abs(time_list - time_list.max())
    plt.figure()
    plt.grid()
    plt.plot(time_list.numpy(), acc_list.numpy())
    plt.xlabel("Timestamp")
    plt.ylabel("Precision")
    # plt.xticks(time_list[select_tick], xtick_list[select_tick])
    # plt.xticks(rotation=30)
    plt.title("Precision-Timestamp Curve")
    plt.savefig(os.path.join(save_path, "pt_curve.pdf"), bbox_inches='tight', dpi=300, format="pdf")

    error_time_list = time_list[pred_list != trg_list]
    print(len(error_time_list))
    with open(f"{os.path.join(save_path, 'time_info.txt')}", mode="w", encoding="utf-8") as f:
        time_str = []
        for i in range(len(error_time_list)):
            time_str.append(str(pd.to_datetime(int(error_time_list[i]*1000000))) + "\n")
        f.writelines(time_str)
