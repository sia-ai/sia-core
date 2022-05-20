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
        self.proj_q = nn.Linear(d_model, d_model, bias=bias_qk)
        self.proj_k = self.proj_q if shared_qk else nn.Linear(d_model, d_model, bias=bias_qk)
        self.proj_v = nn.Linear(d_model, d_model, bias=bias_v)
        self.proj_o = nn.Linear(d_model, d_model, bias=bias_out)
        self.n_heads = n_heads
        self.d_model = d_model
        self.bucket_size = bucket_size
        self.initialized = True

    def forward(self, x, padding_mask=None, attn_mask=None):
        # lazy initialization
        if not self.initialized:
            d_model = x.shape[2]
            self.__lazy_init__(d_model, self.n_heads, self.bucket_size, self.shared_qk, self.bias_qk, self.bias_v, self.bias_out)
            self.to(x.device)

        # project to hash
        h = self.hash(x) # [N, L, 2*n_heads]

        # calucate angle
        h = torch.split(h, 2, dim=2)
        angles = torch.cat([(a[:,:,0]/(a[:,:,1]+self.eps)).unsqueeze(2) for a in h], dim=2) # [N, L, n_heads]

        # caluculate indexes by angles
        indexes = torch.argsort(angles, dim=1) # [N, L, index]
        indexes_of_each_heads = torch.split(indexes, 1, dim=2) # list of [N, L], length = n_heads
        
        # caluculate dimention of one head
        d_head = self.d_model // self.n_heads

        # project q,k,v
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        # sort head by indexes
        def sort_by_indexes(seq, h_indexes): # seq: [N, L, d_head], h_indexes: [N, L]
            return torch.gather(seq, 1, h_indexes.expand(-1, -1, d_head))

        # split q, k, v by heads
        q, k, v = [ torch.split(n, d_head, dim=2) for n in [q, k ,v] ]
        
        # sort
        q = [ sort_by_indexes(hseq, hi) for hseq, hi in zip(q, indexes_of_each_heads) ]
        k = [ sort_by_indexes(hseq, hi) for hseq, hi in zip(k, indexes_of_each_heads) ]
        v = [ sort_by_indexes(hseq, hi) for hseq, hi in zip(v, indexes_of_each_heads) ]
        # list of [N, L, 1], length = n_heads

        # generate mask if not given it
        if padding_mask == None:
            padding_mask = torch.zeros(x.shape[0], x.shape[1], dtype=bool)
        
        if attn_mask == None:
            attn_mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1], dtype=bool) # [N, Tgt, Src]

        # integrate masks
        padding_mask = padding_mask.unsqueeze(2).expand(-1, -1, x.shape[1]).swapaxes(1,2) # expand Tgt axis
        integrated_mask = torch.logical_or(padding_mask, attn_mask)
        
        # sort mask by LSH indexes each heads
        integrated_mask_each_head = integrated_mask.unsqueeze(3).expand(-1, -1, -1, self.n_heads) # [N, Tgt, Src, n_heads]
        integrated_mask_each_head = torch.split(integrated_mask_each_head, 1, dim=3) # split by heads. List of [N, Tgt Src, 1] length=n_heads
        integrated_mask_each_head = [ hm[:, :, :, 0] for hm in integrated_mask_each_head ] # squeeze
        
        # sort dim=1, dim=2
        # sort tgt
        integrated_mask_each_head = [ torch.gather(hm, 2, hi.expand(-1, -1, x.shape[1])) for hm, hi in zip(integrated_mask_each_head, indexes_of_each_heads)]
        # sort src
        integrated_mask_each_head = [ torch.gather(hm, 1, torch.swapaxes(hi.expand(-1, -1, x.shape[1]), 1, 2)) for hm, hi in zip(integrated_mask_each_head, indexes_of_each_heads)]

        # caluclate number of buckets
        n_bucket = (x.shape[1] // self.bucket_size) + 1

        # pad mask to n_bucket*bucket_size
        p_src = self.bucket_size - (integrated_mask_each_head[0].shape[2] % self.bucket_size) + self.bucket_size 
        p_tgt = self.bucket_size - (integrated_mask_each_head[0].shape[1] % self.bucket_size) 

        integrated_mask_each_head = [ torch.cat([hm, hm[:, :, :p_src]], dim=2) for hm in integrated_mask_each_head ] #src dim
        integrated_mask_each_head = [ torch.cat([hm, hm[:, :p_tgt, :]], dim=1) for hm in integrated_mask_each_head ] #tgt dim
        # Note: bucket(n) refers bucket(n) and bucket(n+1)

        # clip mask by buckets each heads
        submasks_each_heads = []
        for hmask in integrated_mask_each_head:
            submasks = []
            for n in range(n_bucket):
                smask = hmask[:, n*self.bucket_size:(n+1)*self.bucket_size, n*self.bucket_size:(n+2)*self.bucket_size]
                block_self_ref = torch.cat([torch.diag(torch.BoolTensor(self.bucket_size)), torch.zeros(self.bucket_size, self.bucket_size, dtype=bool)], dim=1) # [bsize, bsize*2]
                block_self_ref = block_self_ref.unsqueeze(0).expand(smask.shape[0], -1, -1)
                smask = torch.logical_or(smask, block_self_ref)
                submasks.append(smask)
            submasks = torch.cat(submasks, dim=0) # concatenate batch dimention  [batch_size * n_bucket, bucket_size, bucket_size*2]
            #print(submasks.shape)
            submasks_each_heads.append(submasks)
            
        # clip kv
        k_, v_ = [], []
        for hk, hv in zip(k, v): # loop each head
            # pad
            hk = torch.cat([hk, hk[:, :p_src]],dim=1)
            hv = torch.cat([hv, hv[:, :p_src]],dim=1)
            
            bk, bv = [], []
            for n in range(n_bucket):
                bk.append(hk[:, n*self.bucket_size:(n+2)*self.bucket_size, :])
                bv.append(hv[:, n*self.bucket_size:(n+2)*self.bucket_size, :])
            # reduce to batch dimention
            bk = torch.cat(bk, dim=0)
            bv = torch.cat(bv, dim=0)
            k_.append(bk)
            v_.append(bv)

        k, v = k_, v_
        
        # pad q
        q = [ torch.cat([hq, hq[:, :p_tgt]], dim=1) for hq in q]
        
        # clip q
        q = [ torch.cat(torch.split(hq, self.bucket_size, dim=1), dim=0) for hq in q ]
       
            
        # attention
        output_each_heads = []
        for hq, hk, hv, hm in zip(q, k, v, submasks_each_heads):
            # matmul
            attn_weight = torch.bmm(hq, torch.swapaxes(hk, 1,2)) / (self.d_model ** 0.5) # scaled dot product

            # attention masking
            hm = hm * torch.full(hm.shape, float('-inf'), device=(x.device))
            hm = torch.nan_to_num(hm, 0)
            attn_weight = attn_weight + hm
            attn_weight = F.softmax(attn_weight, dim=2)

            ho = torch.bmm(attn_weight, hv)
            output_each_heads.append(ho)
        
        # expand buckets to length ( inverse of clip q )
        output_each_heads = [torch.cat(torch.split(ho, x.shape[0], dim=0), dim=1)[:, :x.shape[1]] for ho in output_each_heads]

        # scatter
        output_each_heads = [torch.scatter(ho, 1, hi, ho) for ho, hi in zip(output_each_heads, indexes_of_each_heads) ]
        
        # output projection
        output_each_heads = torch.cat(output_each_heads, dim=2)

        output = self.proj_o(output_each_heads)

        return output


# test 
seq = torch.randn(27,83,512)
attn = LSHAttention()
attn(seq)

