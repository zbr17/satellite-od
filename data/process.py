#%%
from typing import Tuple, Union
import pandas as pd 
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from tqdm import tqdm
import json
import argparse

class CONFIG:
    num_params: int = ...
    choose_params_list: Union[str, list] = ...
    abnormal_flag: Union[int, list] = ...
    is_gen_abnormal: bool = ...
    gen_abnormal_amp: float = ...
    data_name: str = ...
    non_nan_thresh: int = 4
    data_path: str = "./data/raw_data/"
    data_out: str = "./data/"

def str2date(input: str) -> int:
    """
    Args:
        input (str): Format yyyy-mm-dd hh:mm:ss.xxx
    """
    df = pd.DataFrame({"date": [input]})
    output = pd.to_datetime(df["date"]).astype("int64")
    return output

def date2str(input: Union[int, list]) -> str:
    if isinstance(input, int):
        return pd.to_datetime(input * 1000000)
    elif isinstance(input, list):
        return [date2str(item) for item in input]

def get_config(data_name: str) -> CONFIG:
    config = CONFIG()
    config.data_name = data_name
    with open(os.path.join(config.data_path, config.data_name, "config.json"), mode="r", encoding="utf-8") as f:
        config_data = json.load(f)
        for k, v in config_data.items():
            setattr(config, k, v)

    os.chdir(os.path.abspath(os.path.join(__file__, "../../")))
    print(f"Abnormal-Flag: {date2str(config.abnormal_flag)}")
    return config

def load_data(config: CONFIG) -> pd.DataFrame:
    """
    Load all the data
    """
    data_path = os.path.join(config.data_path, config.data_name)
    file_list = os.listdir(data_path)
    file_list = [item for item in file_list if ".csv" in item]
    # data_dict = {}
    # for file_name in tqdm(file_list):
    #     file_idx = int(file_name.split(".")[0])
    #     data_dict[file_idx] = pd.read_csv(os.path.join(data_path, file_name))
    # data_list = [data_dict[idx] for idx in sorted(list(data_dict.keys()), reverse=False)]
    # data = pd.concat(data_list, axis=0)
    # data = data.reset_index(drop=True)
    # data = data.drop(data.columns[0], axis=1)
    # print(data)

    data = pd.DataFrame()
    for file_name in tqdm(file_list):
        file_path = os.path.join(data_path, file_name)
        data = data.append(pd.read_csv(file_path))
    data.sort_values(by=data.columns[1], inplace=True, ascending=True)
    data = data.drop(data.columns[0], axis=1)
    data = data.reset_index(drop=True)
    print(data)

    if isinstance(config.choose_params_list, list):
        for i in range(config.num_params):
            if i not in config.choose_params_list:
                data = data.drop(f"param{i}", axis=1)
        config.num_params = len(config.choose_params_list)

    return data

def interp(config: CONFIG, data: pd.DataFrame) -> Tuple[np.ndarray]:
    """
    Interpolation
    """
    ## Compute the interpolation points
    non_nan_pos = ~np.isnan(data.values)
    non_nan_pos, = np.where(np.sum(non_nan_pos, axis=1) > config.non_nan_thresh)

    sub_data = data.iloc[non_nan_pos]
    time_idx = data['time'].values
    interp_idx = sub_data['time'].values

    ## Linear interpolation
    data_aligned = []
    params_idx = (range(config.num_params) 
        if config.choose_params_list == "all"
        else config.choose_params_list
    )
    for i in params_idx:
        non_nan_list = ~np.isnan(data[f"param{i}"])
        xp = time_idx[non_nan_list]
        fp = data[f"param{i}"].values[non_nan_list]
        data_aligned.append(np.interp(interp_idx, xp, fp))
    data_aligned = np.stack(data_aligned, axis=-1)
    return data_aligned, interp_idx

def get_label(config: CONFIG, interp_idx: np.ndarray) -> np.ndarray:
    """
    Generate the label
    """
    flag = config.abnormal_flag
    if isinstance(flag, int):
        label = np.zeros_like(interp_idx)
        label[interp_idx > flag] = 1
    elif isinstance(flag, list):
        assert len(flag) == 2
        label = np.zeros_like(interp_idx)
        label[(interp_idx > flag[0]) & (interp_idx < flag[1])] = 1
    return label

def postprocess(config: CONFIG, label: np.ndarray, interp_idx: np.ndarray, data_aligned: np.ndarray) -> np.ndarray:
    """
    Normalization and add noise
    """
    # Add noise (Optional)
    if config.is_gen_abnormal:
        data_aligned[label == 1] += config.gen_abnormal_amp * np.sqrt(
            interp_idx[label == 1] - config.abnormal_flag
        ).reshape(-1, 1)
    # Normalize
    mean_value = np.mean(data_aligned, axis=0, keepdims=True)
    std_value = np.std(data_aligned, axis=0, keepdims=True)
    data_normed = (data_aligned - mean_value) / (std_value + 1e-8)
    return data_normed

def save2ckpt(config: CONFIG, data_normed: np.ndarray, interp_idx: np.ndarray, label: np.ndarray) -> None:
    """
    Save to PyTorch.ckpt
    """
    data_to_save = {
        "data": torch.from_numpy(data_normed).float(),
        "time": torch.from_numpy(interp_idx).long(),
        "label": torch.from_numpy(label).long()
    }
    torch.save(data_to_save, f"{os.path.join(config.data_out, config.data_name)}.ckpt")

def plot_data(config: CONFIG, data: np.ndarray, title="") -> None:
    # Plot the data
    for i in range(config.num_params):
        plt.figure()
        plt.title(f"{title}-{i}")
        plt.plot(data[:,i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data preprocess")
    parser.add_argument("--data-name", type=str, default="gf")
    opt = parser.parse_args()

    os.chdir(os.path.abspath(os.path.join(__file__, "../../")))
    # Get config
    config = get_config(opt.data_name)
    # Load data
    data = load_data(config)
    # Interpolation
    data_aligned, interp_idx = interp(config, data)
    # Label
    label = get_label(config, interp_idx)
    # Normalize
    data_normed = postprocess(config, label, interp_idx, data_aligned)
    # Save ckpt
    save2ckpt(config, data_normed, interp_idx, label)
    # Plot
    plot_data(config, data_normed, title="norm")
