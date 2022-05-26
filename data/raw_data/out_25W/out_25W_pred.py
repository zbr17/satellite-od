#%%
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
    num_idx = 1864941
    num_params = 5
    data_name = "out_25W"
    data_path = "./data/raw_data/"
    data_out = "./data/"
    abnormal_flag = 1641024000000
    increase = 1e-3
    normal_range = [
        [500, 1000],
        [25, 50],
        [-15, 55],
        [-25, 60],
        [0, 20]
    ]

print(pd.to_datetime(config.abnormal_flag*1000000))
df = pd.DataFrame({'date': ['2020-04-14 08:00:00']})
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].astype('int64')
print(df.values)

#%%
# Load all the data
os.chdir("/home/zbr/Workspace/proj/space")
data_path = os.path.join(config.data_path, config.data_name)
file_list = os.listdir(data_path)
file_list = [item for item in file_list if ".csv" in item]
data_dict = {}
for file_name in tqdm(file_list):
    file_idx = int(file_name.split(".")[0])
    data_dict[file_idx] = pd.read_csv(os.path.join(data_path, file_name))
data_list = [data_dict[idx] for idx in sorted(list(data_dict.keys()), reverse=False)]
data = pd.concat(data_list, axis=0)

# Check the data
print("total", len(data))
for i in range(config.num_params):
    print(f"param{i}", sum(~np.isnan(data[f"param{i}"])))

#%%
# Interpolation

## Compute the interpolation points
time_idx = data['time'].values
step = int((time_idx.max() - time_idx.min()) / config.num_idx)
start = int(config.start_idx_ratio * step) + time_idx.min()
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
# Generate the label
flag = config.abnormal_flag
label = np.zeros_like(interp_idx)
label[interp_idx > flag] = 1


#%%
# Normalization
data_aligned[label == 1] += config.increase * np.sqrt(interp_idx[label==1] - flag).reshape(-1, 1)
mean_value = np.mean(data_aligned, axis=0, keepdims=True)
std_value = np.std(data_aligned, axis=0, keepdims=True)
for i in range(config.num_params):
    plt.figure()
    plt.plot(data_aligned[:,i])
    plt.savefig(f"./data/raw_data/time_info/out_25W/params{i}.png")
data_normed = (data_aligned - mean_value) / (std_value + 1e-8)


#%% 
# Save to PyTorch.ckpt

data_to_save = {
    "data": torch.from_numpy(data_normed).float(),
    "time": torch.from_numpy(interp_idx).long(),
    "label": torch.from_numpy(label).long()
}

torch.save(data_to_save, f"{os.path.join(config.data_out, config.data_name)}.ckpt")

#%%
# Plot the data
for i in range(config.num_params):
    plt.figure()
    plt.plot(data_normed[:,i])
    plt.savefig(f"./results/pred/out_25W/params{i}.png")

# %%
# Save abnormal timestamps
for i in range(config.num_params):
    lower_bound = config.normal_range[i][0] + 10
    upper_bound = config.normal_range[i][1] + 10
    mask = (data_aligned[:, i] > upper_bound) | (data_aligned[:, i] < lower_bound)
    sub_timestamp = interp_idx[mask]
    with open(os.path.join("./data/raw_data/time_info/out_25W", f"time{i}.txt"), mode="w", encoding="utf-8") as f:
        time_str = []
        for i in tqdm(range(len(sub_timestamp))):
            time_str.append(str(pd.to_datetime(sub_timestamp[i]*1000000)) + "\n")
        f.writelines(time_str)

#%%