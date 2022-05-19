import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import revtorch as rv
from datetime import datetime
import math

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
    def __init__(self, d_model=None, hidden_dim=512, activation=nn.GELU()):
        super(TwoLP, self).__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.initialized = False
        if d_model:
            self.__lazy_init__(d_model, hidden_dim, activation)

    def __lazy_init__(self, d_model, hidden_dim, activation):
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            d_model = x.shape[-1]
            self.__lazy_init__(d_model, self.hidden_dim, self.activation)
            self.to(x.device)
        return self.fc2(self.activation(self.fc1(x)))

class ExpertMLP(nn.Module):
    def __init__(self, d_model=None, hidden_dim=None, activation=nn.GELU()):
        super(ExpertMLP, self).__init__()
        self.d_model = d_model
        self.hidden_dim = d_model
        self.activation = activation
        self.initialized = False

    
    def __lazy_init__(self, d_model, hidden_dim, activation):
        if not hidden_dim:
            self.hidden_dim = d_model
            self.gate = nn.Linear(d_model, 1)
            self.mod = TwoLP(d_model, hidden_dim, activation)
        
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            d_model = x.shape[-1]
            self.__lazy_init__(d_model, self.hidden_dim, self.activation)
        return self.mod(x)


# Mixture of experts
class MixtureOfExperts(nn.Module):
    def __init__(self, d_model=None, experts=[], num_available_experts=4, logger=nn.Identity()):
        super(MixtureOfExperts, self).__init__()
        self.d_model = d_model
        self.experts = experts
        self.num_available_experts = num_available_experts
        self.logger = logger
        self.initialized = False
        if d_model:
            self.__lazy_init__(d_model, experts, num_available_experts, logger)

    def __lazy_init__(self, d_model, experts, num_available_experts, logger):
        self.experts = nn.ModuleList(experts)
        self.d_model = d_model
        self.logger = logger  # logger function. required callable
        self.num_available_experts = num_available_experts
        self.initialized = True

    # x: [*, *, d_model]
    def forward(self, x):
        # lazy initialization
        if not self.initialized:
            d_model = x.shape[-1]
            self.__lazy_init__(d_model, self.experts, self.num_available_experts, self.logger)
            self.to(x.device)

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
class LSHAttention(nn.Module):
    def __init__(
            self,
            d_model=None,
            n_heads=8,
            bucket_size=4,
            shared_qk=True,
            bias_qk=True,
            bias_v=True,
            bias_out=True,
            eps = 1e-4,
            logger=nn.Identity(),
            ):
        super(LSHAttention, self).__init__()
        self.initialized=False
        self.n_heads = n_heads
        self.bucket_size = bucket_size
        self.shared_qk = shared_qk
        self.bias_qk = bias_qk
        self.bias_v = bias_v
        self.bias_out = bias_out
        self.initialized = False
        self.logger=logger
        self.eps = eps
        if d_model != None:
            self.__lazy_init__(d_model, n_heads, bucket_size, shared_qk, bias_qk, bias_v, bias_out)

    def __lazy_init__(self, d_model, n_heads, bucket_size, shared_qk, bias_qk, bias_v, bias_out):
        assert d_model % n_heads == 0, f"d_model({d_model}) must be divied by n_heads({n_heads})"
        self.hash = nn.Linear(d_model, n_heads*2, bias=False)
        self.proj_q = nn.Linear(d_model, d_model, biasl=bias_qk)
        self.proj_k = self.proj_q if shared_qk else nn.Linear(d_model, d_model, bias=bias_qk)
        self.proj_v = nn.Linear(d_model, d_model, bias=bias_v)
        self.proj_o = nn.Linear(d_model, d_model, bias=bias_out)
        self.n_heads = n_heads
        self.d_model = d_model
        self.bucket_size = bucket_size
        self.initialized = True

    def forward(self, x, padding_mask=None, attn_mask=None):
        if not self.initialized:
            d_model = x.shape[2]
            self.__lazy_init__(d_model, self.n_heads, self.bucket_size, self.shared_qk, self.bias_qk, self.bias_v, self.bias_out)
            self.to(x.device)

        # project to hash
        h = self.hash(x) # [N, L, 2*n_heads]

        # calucate angle
        h = torch.split(h, 2, dim=2)
        angles = torch.cat([a[:,:,0]/(a[:,:,1]+self.eps) for a in h],dim=2) # [N, L, n_heads]

        # caluculate indexes by angles
        indexes = torch.argsort(angles, dim=1) # [N, L, index]
        indexes_of_each_heads = torch.split(indexes, dim=2) # list of [N, L], length = n_heads
        
        # caluculate dimention of one head
        d_head = self.d_model // self.n_heads

        # project q,k,v
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

