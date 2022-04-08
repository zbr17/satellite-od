#%%
from audioop import reverse
import pandas as pd 
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from tqdm import tqdm

#%%
# Configuration
class config:
    start_idx_ratio = 0.5
    num_idx = 595842
    num_params = 7

#%%
# Load all the data
# data_path = "./data/raw_data"
data_path = "./raw_data"
file_list = os.listdir(data_path)
data_dict = {}
for file_name in tqdm(file_list):
    file_idx = int(file_name.split(".")[0])
    data_dict[file_idx] = pd.read_csv(os.path.join(data_path, file_name))
data_list = [data_dict[idx] for idx in sorted(list(data_dict.keys()), reverse=False)]
data = pd.concat(data_list, axis=0)
## time to zero
config.time_idx_min = np.min(data['time'])
data['time'] = data['time'] - np.min(data['time'])

def time2idx(time):
    return time - config.time_idx_min

# # Plot the data
# print("total", len(data))
# for i in range(config.num_params):
#     print(f"param{i}", sum(~np.isnan(data[f"param{i}"])))

#%%
# Interpolation

## Compute the interpolation points
time_idx = data['time'].values
step = int(time_idx.max() / config.num_idx)
start = int(config.start_idx_ratio * step)
interp_idx = np.array(range(start, time_idx.max(), step))[:config.num_idx] # drop the last few samples

## Linear interpolation
data_aligned = []
for i in range(config.num_params):
    non_nan_list = ~np.isnan(data[f"param{i}"])
    xp = time_idx[non_nan_list]
    fp = data[f"param{i}"].values[non_nan_list]
    data_aligned.append(np.interp(interp_idx, xp, fp))
data_aligned = np.stack(data_aligned, axis=-1)

#%%
# Normalization
mean_value = np.mean(data_aligned, axis=0, keepdims=True)
std_value = np.std(data_aligned, axis=0, keepdims=True)
data_normed = (data_aligned - mean_value) / (std_value + 1e-8)

#%% 
# Save to PyTorch.ckpt

data_to_save = {
    "data": torch.from_numpy(data_normed),
}

torch.save(data_to_save, "data.ckpt")

#%%
# Plot the data
for i in range(7):
    plt.figure()
    plt.plot(data_normed[:,i])


#%%
pd.to_datetime(1627577388568*1000000)