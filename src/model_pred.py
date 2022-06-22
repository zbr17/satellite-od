import torch
import torch.nn as nn
import torch.nn.init as init
import math

def give_model(config):
    if config.model_name == "lstm":
        model = RNN(
            input_size=config.input_dim + 1,
            hidden_size=config.hidden_dim,
            num_layers=config.model_layer,
            out_size=config.output_size
        )
    elif config.model_name == "transformer":
        model = Transformer(
            input_size=config.input_dim + 1,
            hidden_size=config.hidden_dim,
            num_layers=config.model_layer,
            out_size=config.output_size,
            nhead=config.nhead
        )
    else:
        raise KeyError()
    return model
    
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

        return out[:, :, :-1]

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size, nhead):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers
        )

        self.fc_in = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, input_size)
    
    def init_weights(self):
        init.kaiming_normal_(self.fc_in.weight)
        init.kaiming_normal_(self.fc_out.weight)
        init.zeros_(self.fc_in.bias)
        init.zeros_(self.fc_out.bias)
    
    def forward(self, x):
        x = self.fc_in(x)
        x = self.transformer(x)
        x = x[:, -self.out_size:, :]
        x = self.fc_out(x)
        return x[:, :, :-1]

if __name__ == "__main__":
    model = Transformer(7, 64, 2, 2, 4)
    data = torch.randn(4, 5, 7)
    output = model(data)
    pass