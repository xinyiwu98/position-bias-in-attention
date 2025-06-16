import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import numpy as np
import math
from rope import RotaryPositionalEmbeddings

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, d_model) with zeros
        pe = torch.zeros(max_len, d_model)
        
        # Create a tensor of shape (max_len, 1) with positions (0 to max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create a tensor of shape (d_model // 2) with the divisors for sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices (2i) and cosine to odd indices (2i+1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add an extra dimension at the beginning for batch size and register pe as a buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add the positional encoding to the input tensor x
        x = x + self.pe[:, :x.size(1), :]
        return x







class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, mask_type='causal', gamma=1, num_prefixes=1, window_size=20, bias=False, pe=None, base=10000):
        super().__init__()
        self.pe = pe
        self.gamma = gamma
        self.num_prefixes = num_prefixes
        self.window_size = window_size
        self.out_channels = out_channels
        
        if self.pe == 'rope':
            self.pos_encoding = RotaryPositionalEmbeddings(dim=out_channels, base=base)

        
        self.query_projection = nn.Linear(in_channels, out_channels, bias=bias)
        self.key_projection = nn.Linear(in_channels, out_channels, bias=bias)
        
        self.value_projection = nn.Linear(in_channels, out_channels, bias=bias)
        self.mask_type = mask_type



    def forward(self, x):
        batch_size, seq_length, hidden_size = x.size()

        # Project inputs to query, key, and value
       
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        if self.pe == 'rope':
            query = self.pos_encoding(query)
            key = self.pos_encoding(key)

        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.out_channels)
        
        if self.mask_type == 'full':
            pass

        elif self.mask_type == 'causal':
            mask = torch.triu(torch.ones((1, seq_length, seq_length)).to(x.device)).transpose(-2, -1).reshape(1, seq_length, seq_length)
            mask = mask.repeat(batch_size, 1, 1)
            mask = torch.log(mask)
            scores = scores + mask


        elif self.mask_type == 'decay':   # from paper "Retentive Network: A Successor to Transformer for Large Language Models"
            mask = decay_mask(seq_length=seq_length, gamma=self.gamma).to(x.device)
            mask = mask.repeat(batch_size, 1, 1)
            mask = torch.log(mask)
            scores = scores + mask

        elif self.mask_type == 'prefix':  
            mask = prefix_mask(seq_length=seq_length, num_prefixes=self.num_prefixes).to(x.device)
            mask = mask.repeat(batch_size, 1, 1)
            mask = torch.log(mask)
            scores = scores + mask

        elif self.mask_type == 'window':  
            mask = window_mask(seq_length=seq_length, window_size=self.window_size).to(x.device)
            mask = mask.repeat(batch_size, 1, 1)
            mask = torch.log(mask)
            scores = scores + mask


        

        # Apply softmax to get attention probabilities
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to the values
        attention_output = torch.matmul(attention_weights, value)

        return attention_output, attention_weights

class SAN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mask_type='causal', gamma=1, num_prefixes=1, window_size=20,
                       hidden_channels=128, num_attn_layers=2, num_mlp_layers=3, bias=False, pe='rope', base=10000, residual=True):
        super().__init__()
        self.residual = residual
        self.attn_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.num_attn_layers = num_attn_layers
        self.num_mlp_layers = num_mlp_layers
        self.pe = pe
        if self.pe == 'sin':
            self.pos_encoding = PositionalEncoding(in_channels)

        self.attn_layers.append(SelfAttention(in_channels=in_channels,
                                              out_channels=hidden_channels, 
                                              bias=bias, 
                                              mask_type=mask_type,
                                              pe=pe,
                                              gamma=gamma,
                                              num_prefixes=num_prefixes,
                                              window_size=window_size,
                                              base=base
                                              ))

        for _ in range(self.num_attn_layers-1):
            self.attn_layers.append(SelfAttention(in_channels=hidden_channels,
                                                  out_channels=hidden_channels,
                                                  bias=bias, 
                                                  mask_type=mask_type,
                                                  pe=pe,
                                                  gamma=gamma,
                                                  num_prefixes=num_prefixes,
                                                  window_size=window_size
                                                  ))
        for _ in range(self.num_mlp_layers):
            self.mlp_layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.lin = torch.nn.Linear(hidden_channels, out_channels)



    def forward(self, x):
        if self.pe == 'sin':
            x = self.pos_encoding(x)

        for layer in self.attn_layers:
            update, _ = layer(x)
            if self.residual:
                # add the update to the input
                x = x + update
            else:
                x = update

        # take the query vector as the input to the mlp:
        x = x[:,-1,:]

        for layer in self.mlp_layers:
            x = layer(x)
            x = F.relu(x)

        x = self.lin(x)

        return x
