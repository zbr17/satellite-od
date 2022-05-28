from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch

class TimeSeries(Dataset):
    def __init__(
        self,
        data_path="./data/data.ckpt",
        input_size=100,
        output_size=1,
        data_mask=None,
    ):
        super().__init__()
        self.mean_delta: float = ...
        self.data_path = data_path
        self.input_size = input_size
        self.output_size = output_size
        self.data_mask = data_mask
        
        data = torch.load(self.data_path)
        self.data = data["data"]
        self.label = data["label"]
        self.time = data["time"]
        if self.data_mask is not None:
            self.data = self.data[self.data_mask]
            self.label = self.label[self.data_mask]
            self.time = self.time[self.data_mask]
        self.dim = self.data.size(-1)
        
        # Generate data slices
        offset_idx = torch.arange(len(self.data)-self.input_size-self.output_size+1).unsqueeze(-1)
        input_idx = torch.arange(self.input_size).unsqueeze(0) + offset_idx
        input_idx = input_idx.flatten()
        output_idx = torch.arange(self.output_size).unsqueeze(0) + offset_idx + self.input_size
        output_idx = output_idx.flatten()
        self.input_list = self.data[input_idx, :].reshape(-1, self.input_size, self.dim)
        self.input_time = self.time[input_idx].reshape(-1, self.input_size)
        self.output_list = self.data[output_idx, :].reshape(-1, self.output_size, self.dim)
        self.output_label = self.label[output_idx].reshape(-1, self.output_size)
        self.output_time = self.time[output_idx].reshape(-1, self.output_size)

        # Add time position
        zero_time = self.input_time[:, -1].unsqueeze(-1)
        input_time = (self.input_time - zero_time).float()
        output_time = (self.output_time - zero_time).float()
        self.mean_delta = torch.mean(torch.abs(input_time)) * 10
        # tanh function
        input_time = torch.tanh(input_time / self.mean_delta)
        output_time = torch.tanh(output_time / self.mean_delta)
        # concat data
        data_padding = torch.zeros(self.input_list.size(0), 1, self.input_list.size(-1))
        data_padding = torch.cat([data_padding, output_time.unsqueeze(-1)], dim=-1)
        self.input_list = torch.cat([self.input_list, input_time.unsqueeze(-1)], dim=-1)
        self.input_list = torch.cat([self.input_list, data_padding], dim=1)
    
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, index):
        return self.input_list[index], self.output_list[index], self.output_label[index], self.output_time[index]

def give_dataloader(config) -> dict:
    sample_set = TimeSeries(data_path=config.data_path, input_size=config.input_size, output_size=config.output_size, data_mask=None)

    label = sample_set.label
    pos_idx_len = len(torch.where(label == 0)[0])
    train_idx_end = int(config.train_idx_ratio * pos_idx_len)
    train_idx_mask = torch.arange(train_idx_end)
    test_idx_mask = torch.arange(train_idx_end, len(sample_set))

    train_set = TimeSeries(data_path=config.data_path, input_size=config.input_size, output_size=config.output_size, data_mask=train_idx_mask)
    test_set = TimeSeries(data_path=config.data_path, input_size=config.input_size, output_size=config.output_size, data_mask=test_idx_mask)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=8, drop_last=False)

    return {
        "train": train_loader, 
        "test": test_loader
    }

