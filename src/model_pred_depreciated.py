import torch
import torch.nn as nn
import torch.nn.init as init
import math

class PositionEncoder(nn.Module):
    def __init__(self, dim=7, max_len=10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        position_emb = torch.zeros(self.max_len, self.dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        even_div_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-math.log(10000.0) / self.dim))
        odd_div_term = torch.exp((torch.arange(1, self.dim, 2).float()-1) * (-math.log(10000.0) / self.dim))
        position_emb[:, 0::2] = torch.sin(position * even_div_term)
        position_emb[:, 1::2] = torch.cos(position * odd_div_term)
        position_emb = position_emb.unsqueeze(0).transpose(0,1)
        self.register_buffer("position_emb", position_emb)
    
    def forward(self, x):
        return x + self.position_emb[:x.size(0), :] 

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size, nhead=8, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.feature_size = hidden_size
        self.nhead = nhead
        self.out_size = out_size
        self.dropout = dropout
        self.src_mask = None
        
        self.input_embedder = nn.Linear(self.input_size, self.feature_size)
        self.position_encoder = PositionEncoder(dim=self.input_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=self.nhead, dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.decoder = nn.Linear(self.feature_size, self.input_size)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        init.kaiming_normal_(self.input_embedder.weight)
        init.zeros_(self.input_embedder.bias)
    
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.position_encoder(src)
        src = self.input_embedder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)

        output = output[:, -self.out_size:, :]
        output = output[:, :, :-1]
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

if __name__ == "__main__":
    model = Transformer()
    x = torch.randn(100, 32, 7)
    y = model(x)
    pass