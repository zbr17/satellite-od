from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch

class SampleSet(Dataset):
    def __init__(
        self,
        data_path="./data/data.ckpt",
        data_mask=None,
    ):
        super().__init__()
        self.data_path = data_path
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
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index], self.time[index]

def give_dataloader(config) -> dict:
    sample_set = SampleSet(data_path=config.data_path, data_mask=None)
    label = sample_set.label
    pos_idx = torch.where(label == 0)[0]
    neg_idx = torch.where(label == 1)[0]

    # get the split indices
    pos_split_num = [
        int(config.split_ratio["train"]*len(pos_idx)),
        int(config.split_ratio["val"]*len(pos_idx))
    ]
    pos_split_num.append(len(pos_idx) - sum(pos_split_num))

    neg_split_num = [
        int(config.split_ratio["train"]*len(neg_idx)),
        int(config.split_ratio["val"]*len(neg_idx))
    ]
    neg_split_num.append(len(neg_idx) - sum(neg_split_num))

    train_pos, val_pos, test_pos = torch.utils.data.random_split(pos_idx, pos_split_num)
    train_neg, val_neg, test_neg = torch.utils.data.random_split(neg_idx, neg_split_num)

    train_set = SampleSet(data_path=config.data_path, data_mask=torch.tensor(train_pos + train_neg).long())
    val_set = SampleSet(data_path=config.data_path, data_mask=torch.tensor(val_pos + val_neg).long())
    test_set = SampleSet(data_path=config.data_path, data_mask=torch.tensor(test_pos + test_neg).long())

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=8, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=8, drop_last=False)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

