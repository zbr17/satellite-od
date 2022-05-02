import torch
import torch.nn as nn
import torch.nn.init as init

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, input_size)
        self.init_weights()
    
    def init_weights(self):
        init.kaiming_normal_(self.fc.weight)
        init.zeros_(self.fc.bias)
    
    def forward(self, x):
        device = x.device
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -self.out_size:, :]
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = RNN(7, 64, 10, 2)
    data = torch.randn(4, 5, 7)
    output = model(data)
    pass