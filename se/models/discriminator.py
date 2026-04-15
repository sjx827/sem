# References: https://github.com/yxlu-0102/MP-SENet/blob/main/models/discriminator.py

import torch
import torch.nn as nn
import numpy as np
from models.lsigmoid import LearnableSigmoid1D

def calculate_sisdr(reference, estimation):
    """计算 SI-SDR，适用于任何音频信号分离评价"""
    eps = 1e-8
    reference = reference - reference.mean()
    estimation = estimation - estimation.mean()
    
    # 计算缩放因子 alpha
    alpha = (torch.sum(reference * estimation) + eps) / (torch.sum(reference ** 2) + eps)
    target = alpha * reference
    noise = estimation - target
    
    sisdr = 10 * torch.log10((torch.sum(target ** 2) + eps) / (torch.sum(noise ** 2) + eps))
    return sisdr

def batch_sisdr(clean, noisy, cfg):
    """计算 batch 的 SI-SDR 并映射到 [0, 1] 供 Discriminator 学习"""
    scores = []
    for c, n in zip(clean, noisy):
        c_t = torch.from_numpy(c)
        n_t = torch.from_numpy(n)
        scores.append(calculate_sisdr(c_t, n_t))
    
    scores = torch.stack(scores)
    # 将 SI-SDR 映射到 [0, 1] 范围内。假设常规分离任务的 SDR 范围在 -10dB 到 20dB
    scores = torch.clamp((scores + 10) / 30.0, 0.0, 1.0)
    return scores

class MetricDiscriminator(nn.Module):
    def __init__(self, dim=16, in_channel=2):
        super(MetricDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*2, affine=True),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Conv2d(dim*2, dim*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*4, affine=True),
            nn.PReLU(dim*4),
            nn.utils.spectral_norm(nn.Conv2d(dim*4, dim*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*8, affine=True),
            nn.PReLU(dim*8),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim*8, dim*4)),
            nn.Dropout(0.3),
            nn.PReLU(dim*4),
            nn.utils.spectral_norm(nn.Linear(dim*4, 1)),
            LearnableSigmoid1D(1)
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)