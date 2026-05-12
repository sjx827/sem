import torch
import torch.nn as nn
from einops import rearrange
from .lsigmoid import LearnableSigmoid2D

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0] * dilation[0] - dilation[0]) / 2), 
            int((kernel_size[1] * dilation[1] - dilation[1]) / 2))

class DenseBlock(nn.Module):
    def __init__(self, cfg, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.cfg = cfg
        self.depth = depth
        self.dense_block = nn.ModuleList()
        self.hid_feature = cfg['model_cfg']['hid_feature']

        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(self.hid_feature * (i + 1), self.hid_feature, kernel_size, 
                          dilation=(dil, 1), padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(self.hid_feature, affine=True),
                nn.PReLU(self.hid_feature)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x

class DenseEncoder(nn.Module):
    def __init__(self, cfg):
        super(DenseEncoder, self).__init__()
        self.cfg = cfg
        self.input_channel = cfg['model_cfg']['input_channel']
        self.hid_feature = cfg['model_cfg']['hid_feature']

        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.hid_feature, (1, 1)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )
        self.dense_block = DenseBlock(cfg, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

    def forward(self, x):
        x = self.dense_conv_1(x)  
        x = self.dense_block(x)   
        x = self.dense_conv_2(x)  
        return x

# 修改后的 CRMDecoder 类
class CRMDecoder(nn.Module):
    def __init__(self, cfg):
        super(CRMDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = 2 
        self.beta = cfg['model_cfg']['beta'] # 建议 beta 设为 2.0 左右

        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.Conv2d(self.hid_feature, self.output_channel, (1, 1)),
            nn.InstanceNorm2d(self.output_channel, affine=True),
            nn.PReLU(self.output_channel),
            nn.Conv2d(self.output_channel, self.output_channel, (1, 1))
        )

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x) # [B, 2, T, F]

        mask_r = x[:, 0, :, :] 
        mask_i = x[:, 1, :, :] 

        # 使用 tanh 将 CRM 限制在 [-beta, beta] 范围内，这比 Sigmoid 更适合复数域
        mask_r = self.beta * torch.tanh(rearrange(mask_r, 'b t f -> b f t'))
        mask_i = self.beta * torch.tanh(rearrange(mask_i, 'b t f -> b f t'))

        # 恢复维度 [B, 1, T, F] 返回给 generator
        mask_r = rearrange(mask_r, 'b f t -> b t f').unsqueeze(1)
        mask_i = rearrange(mask_i, 'b f t -> b t f').unsqueeze(1)

        return mask_r, mask_i