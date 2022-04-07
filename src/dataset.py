from torch.utils.data.dataset import Dataset
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
        self.data_path = data_path
        self.input_size = input_size
        self.output_size = output_size
        self.data_mask = data_mask
        
        data = torch.load(self.data_path)
        self.data = data["data"]
        if self.data_mask is not None:
            self.data = self.data[self.data_mask]
        self.dim = self.data.size(-1)
        
        # Generate data slices
        offset_idx = torch.arange(len(self.data)-self.input_size-self.output_size+1).unsqueeze(-1)
        input_idx = torch.arange(self.input_size).unsqueeze(0) + offset_idx
        input_idx = input_idx.flatten()
        output_idx = torch.arange(self.output_size).unsqueeze(0) + offset_idx + self.input_size
        output_idx = output_idx.flatten()
        self.input_list = self.data[input_idx, :].reshape(-1, self.input_size, self.dim)
        self.output_list = self.data[output_idx, :].reshape(-1, self.output_size, self.dim)
    
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, index):
        return self.input_list[index], self.output_list[index]