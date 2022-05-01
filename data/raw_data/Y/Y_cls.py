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
    num_params = 10
    non_nan_thresh = 4
    data_name = "Y"
    data_path = "./data/raw_data/"
    data_out = "./data/"
    abnormal_flag = 1586851000000

print(pd.to_datetime(config.abnormal_flag*1000000))
df = pd.DataFrame({'date': ['2020-04-14 08:00:00.000']})
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].astype('int64')
print(df)

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
print(data)

#%%
# Interpolation

## Compute the interpolation points
non_nan_pos = ~np.isnan(data.values)
non_nan_pos, = np.where(np.sum(non_nan_pos, axis=1) > config.non_nan_thresh)

sub_data = data.iloc[non_nan_pos]
time_idx = data['time'].values
interp_idx = sub_data['time'].values

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
# Generate the label
flag = config.abnormal_flag
label = np.zeros_like(interp_idx)
label[interp_idx > flag] = 1

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
for i in range(7):
    plt.figure()
    plt.plot(data_normed[:,i])

# %%