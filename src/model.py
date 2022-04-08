import torch
import torch.nn as nn
import torch.nn.init as init

class classifier(nn.Module):
    def __init__(self, size_list=[7,64,64,2]):
        super().__init__()
        self.size_list = size_list
        self.num_layer = len(self.size_list) - 1

        self.layer_list = []
        for i in range(self.num_layer):
            self.layer_list.append(nn.Linear(self.size_list[i], self.size_list[i+1]))
            if i != self.num_layer-1:
                self.layer_list.append(nn.ReLU(inplace=True))
        self.layer_list = nn.ModuleList(self.layer_list)
        self.init_weights()

    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
                init.zeros_(module.bias)
    
    def forward(self, x):
        for module in self.layer_list:
            x = module(x)
        return x


if __name__ == "__main__":
    model = classifier()
    x = torch.randn(4, 7)
    y = model(x)
    pass