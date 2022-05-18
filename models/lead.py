import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import revtorch as rv
from datetime import datetime
import math

def apply_stochastic_depth(seq: nn.Sequential, max_p=1.0, min_p=0.5):
    return nn.Sequential(*[Stochastic(mod, p) for p, mod in zip(np.linspace(max_p, min_p, len(seq)), seq)])


class Stochastic(nn.Module):
    def __init__(self, module, p=0.5):
        super(Stochastic, self).__init__()
        self.probability = p
        self.module = module

    def forward(self, *args, **kwargs):
        if random.random() <= self.probability or self.training:
            return self.module(*args, **kwargs)
        else:
            return args


def string2activation(s):
    if s == 'gelu':
        return nn.GELU()
    if s == 'relu':
        return nn.ReLU()

# Dummy layer for rv.ReversibleSequence


class IrreversibleBlock(nn.Module):
    def __init__(self, f_block, g_block, split_along_dim=1):
        super(IrreversibleBlock, self).__init__()
        self.f_block, self.g_block = f_block, g_block
        self.split_along_dim = split_along_dim

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=self.split_along_dim)
        y1, y2 = None, None
        y1 = x1 + self.f_block(x2)
        y2 = x2 + self.g_block(y1)

        return torch.cat([y1, y2], dim=self.split_along_dim)

# two layer perceptron


class TwoLP(nn.Module):
    def __init__(self, d_model, hidden_dim, activation=nn.GELU()):
        super(TwoLP, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

# some module with gate
# module must be take [*, *, d_model] shape and returns [*, *, d_model].
# hint : parameter of gate works as summary of this expert module.


class Expert(nn.Module):
    def __init__(self, d_model, module, name=None):
        super(Expert, self).__init__()
        self.gate = nn.Linear(d_model, 1, bias=False)
        self.module = module
        if not name:
            self.name = f'Unnnamed expert module {hex(random.randint(0, 2**32))}'
        else:
            self.name=name

    def forward(self, x):
        return self.module(x)

# expert must have to have attribute: gate, module, name.


class MixtureOfExperts(nn.Module):
    def __init__(self, d_model, experts=[], num_available_experts=4, logger=nn.Identity()):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.d_model = d_model
        self.logger = logger  # logger function. required callable
        self.num_available_experts = num_available_experts

    # x: [*, *, d_model]
    def forward(self, x):
        # caluclate key of gate
        k = torch.sum(x, dim=[0, 1]) # [d_model]
        # weight of each gate
        # gate(k) : shape=[1]
        gw = torch.stack([e.gate(k) for e in self.experts]) # [number of experts]
        gw = gw.squeeze(1)
        gw, indexes = torch.topk(gw, min(self.num_available_experts, len(self.experts)))

        available_experts = [self.experts[i] for i in indexes]
        self.logger(f"selected experts: {', '.join([e.name for e in available_experts])}")

        gw = F.softmax(torch.stack([ expert.gate(x) for expert in available_experts], dim=2).squeeze(3), dim=2)

        # call available experts
      
        x = sum([expert(x) * weight.swapaxes(0,1).unsqueeze(-1) for expert, weight in zip(available_experts, gw.swapaxes(0,2))])
        return x
        
    def append(self, expert):
        self.experts.append(expert)


# Inputs [batch_size, seq_len, d_model]
# Outputs [batch_size, seq_len, d_model]

# if you need use it as self-attention like, for example,
# hint: set submodule == nn.Conv1d(), swap axes (1, 2), before and after of passing this module.
class WithLSHSort(nn.Module):
    def __init__(self,
            d_model=512,
            n_heads=8,
            submodule=nn.Identity(),
            eps=1e-4
            ):
        super(WithLSHSort, self).__init__()
        assert d_model % n_heads == 0, f"d_model must be able to devided by n_heads"
        self.hash = nn.ModuleList([nn.Linear(d_model // n_heads, 2) for _ in range(d_model // n_heads)])
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.mod = submodule
        self.eps = 1e-4

    def forward(self, x):
        # caluclate indexes

        projected = torch.cat([self.hash[n](head) for n, head in zip(range(self.n_heads), torch.split(x, self.d_head, dim=2))], dim=2)
        h_x, h_y = torch.split(projected, self.n_heads, dim=2) # [batch_size, seq_len, nheads] where h_x, h_y
        angles = torch.arctan(h_x / (h_y + self.eps)) # [batch_size, seq_len, n_heads] # calculate angle of vector
        indexes = torch.argsort(angles, 1) # [batch_size, seq_len, n_heads]
        indexes = torch.unsqueeze(indexes, dim=3).expand(-1, -1, -1, self.d_head) # [batch_size, seq_len, n_heads, d_head]
        
        # split heads
        x = x.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head)

        # sort heads
        x = torch.gather(x, 1, indexes)

        # concatenate heads
        x = x.reshape(x.shape[0], x.shape[1], self.d_model)
                
    
        # call module
        x = self.mod(x)
        
        # split heads
        x = x.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head)

        
        # scatter
        x = torch.scatter(torch.zeros_like(x) ,1, indexes, x)

        # concatenate heads
        x = x.reshape(x.shape[0], x.shape[1], self.d_model)
        return x

# convolution with swap axes.
class Conv1dForLSHSort(nn.Module):
    def __init__(self, d_model, kernel_size, stride, padding, padding_mode='circular', **kwargs):
        super(Conv1dForLSHSort, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, stride, padding, padding_mode=padding_mode, **kwargs)
    def forward(self, x):
        x = x.swapaxes(1,2)
        x = self.conv(x)
        x = x.swapaxes(1,2)
        return x


# convolution with swap axes.
class Conv1dForLSHSort(nn.Module):
    def __init__(self, d_model, kernel_size, stride, padding, padding_mode='circular', **kwargs):
        super(Conv1dForLSHSort, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, stride, padding, padding_mode=padding_mode, **kwargs)
    def forward(self, x):
        x = x.swapaxes(1,2)
        x = self.conv(x)
        x = x.swapaxes(1,2)
        return x

class MultiheadAttentionForLSHSort(nn.Module):
    def __init__(self, d_model, segment_size=4, n_heads=8, logger=nn.Identity()):
        super(MultiheadAttentionForLSHSort, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.seg_size = segment_size
        self.proj_qk = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.logger=logger
    def forward(self, x):
        # x shape = [batch_size, seq_len, d_model]
        # convert to [batch_size*seg_size, seq_len//seg_size, d_model]

        # pad
        pad_seq_len = self.seg_size - (x.shape[1] % self.seg_size)
        seq_len = x.shape[1]
        a = (seq_len + pad_seq_len) // self.seg_size
        x = torch.cat([x, x[:, 0:pad_seq_len, :]], dim=1)
        x = torch.cat(torch.chunk(x, a, dim=1), dim=0) # pack to batch
        self.logger(f"Splitted attention {a} blocks, {self.seg_size} tokens per block")
        mask = torch.diag(torch.BoolTensor(self.seg_size)).to(x.device)
        x, _ = self.attn(self.proj_qk(x), self.proj_qk(x), self.proj_v(x), attn_mask=mask)
        x = torch.cat(torch.chunk(x, a, dim=0), dim=1)
        x = x[:, 0:seq_len, :]
        return x

class LSHAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, segment_size=4, logger=nn.Identity()):
        super(LSHAttention, self).__init__()
        self.seq = WithLSHSort(d_model, n_heads, MultiheadAttentionForLSHSort(d_model, segment_size=segment_size, n_heads=n_heads, logger=logger))

    def forward(self, x):
        return self.seq(x)


# convolution sequece with LSH sort
# For example usage, replace Transformer's MultiHeadAttention to it
class LSHConv(nn.Module):
    def __init__(self, d_model, n_heads, kernel_size=3, stride=1, padding=1, padding_mode='circular', groups=None, bias=True):

        super(LSHConv, self).__init__()

        if not groups:
            groups = n_heads
        submodule = Conv1dForLSHSort(d_model, kernel_size, stride, padding, padding_mode, groups=groups, bias=bias)
        self.lsh_module = WithLSHSort(d_model, n_heads, submodule)

    def forward(self,x):
        return self.lsh_module(x)

    def forward(self,x):
        return self.lsh_module(x)

# LEAD: Lesser-required-computability Efficient Approximated Data transformer
class LEAD(nn.Module):
    def __init__(
            self,
            d_model=256,
            n_heads=8,
            n_layers=12,
            d_expert_ffn=256,
            n_experts=4,
            layer_drop_probability=0.5,
            spatial_mixer_class=LSHAttention,
            reversible=True,
            spatial_mixer_kwargs={
                'n_heads': 8,
                'segment_size' :4
                },
            logger=nn.Identity()
            ):
        block_class = rv.ReversibleBlock if reversible else IrreversibleBlock
        seq_init    = (lambda blocks: rv.ReversibleSequence(nn.ModuleList(blocks))) if reversible else (lambda blocks: nn.Sequential(*blocks))
        super(LEAD, self).__init__()
        
        self.logger = logger
        seq = []
        self.moes = []
        self.d_model = d_model
        for i, d_prob in zip(range(n_layers), np.linspace(1.0, layer_drop_probability)):
            moe = MixtureOfExperts(
                    d_model,
                    [Expert(
                        d_model,
                        TwoLP(
                            d_model,
                            d_expert_ffn),
                        name=f"FFN of Layer:{i} No.:{j}"
                        ) for j in range(n_experts)],
                    num_available_experts=n_experts, logger=logger)
            self.moes.append(moe)
            seq.append(
                block_class(
                    Stochastic(
                        nn.Sequential(# F block: spatial mixer
                            nn.LayerNorm(d_model),
                            spatial_mixer_class(d_model, logger=logger, **spatial_mixer_kwargs),
                            ),
                        p=d_prob
                        ),
                    Stochastic(
                        nn.Sequential( # G block: FeedForward
                            nn.LayerNorm(d_model),
                            moe,
                            ), p=d_prob
                        ),
                    split_along_dim=2
                    )
                )
            self.seq = seq_init(seq)
        
        self.num_available_experts_ = n_experts

    def forward(self, x):
        start_time = datetime.now()
        self.logger(f"start processing shape={x.shape}")
        x = torch.repeat_interleave(x, repeats=2, dim=2)
        x = self.seq(x)
        x1, x2 = torch.chunk(x, 2, dim=2)
        x = (x1 + x2) / 2
        delta_time = datetime.now() - start_time
        self.logger(f"finished processing at {delta_time.total_seconds()}seconds")
        return x

    @property
    def num_avairable_experts(self):
        return self.num_available_experts_

    @num_avairable_experts.setter
    def num_available_experts(self, num):
        self.num_available_experts_ = num
        for moe in self.moes:
            moe.num_available_experts = num
    
    def add_expert_mlp(self, name='Unnamed Expert', dim=None):
        d_model = self.d_model
        if not dim:
            dim = self.d_model
        for i, moe in enumerate(self.moes):
            moe.append(Expert(d_model, TwoLP(d_model, dim), name=f"{name} of Layer {i}"))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int,  max_len: int = 1000, logger=nn.Identity()):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_len = max_len
        self.d_model = d_model
        self.logger = logger

    def forward(self, x):
        if x.shape[1] > self.max_len:
            self.logger(f"PositionalEncoding: updated max length: {self.max_len} -> {x.shape[1]}")
            self.__init__(self.d_model, x.shape[1], logger=self.logger)
        x = x.to(self.pe.device)
        x = x + self.pe[:, :x.size(0)]
        return x

class LEADForSIA(nn.Module):
    def __init__(self, d_model, model):
        super(LEADForSIA, self).__init__()
        self.memory_embedding = torch.randn(d_model)
        self.model = model
        self.pe = PositionalEncoding(d_model)
        self.positional_encoding = self.pe
        self.d_model = d_model

    def forward(self, x, memory):
        out = self.model(torch.cat([x, memory], dim=1))
        out, mem = out[:, :x.shape[1], :], out[:, x.shape[1]:, :]
        return out, mem
    
    # allocate new memory
    def allocate(self, length):
        memory = torch.zeros(length, self.d_model)
        memory = self.pe(memory)
        self.memory_embedding = self.memory_embedding.to(memory.device)
        memory + self.memory_embedding

        return memory
