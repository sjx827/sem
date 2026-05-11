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

class CRMDecoder(nn.Module):
    """
    CRMDecoder: 预测带有频域先验权重的复数比值掩蔽(CRM)实部和虚部
    """
    def __init__(self, cfg):
        super(CRMDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = 2  # 输出两路 (CRM_real, CRM_imag)
        self.n_fft = cfg['stft_cfg']['n_fft']
        self.beta = cfg['model_cfg']['beta']
        
        # ---------------- 核心参数调整区 ----------------
        # 假设你的采样率是 16000Hz (如果是其他，请在这里修改或从 cfg 读取)
        self.sr = 16000 
        self.split_freq = 1200       # Hz: 树蛙叫声大致开始的最低频率
        self.high_freq_weight = 2.0  # 高频区的强化倍数
        # -----------------------------------------------

        n_freqs = self.n_fft // 2 + 1
        # 计算 2000Hz 对应的特征维度索引
        self.split_idx = min(int((self.split_freq / (self.sr / 2)) * n_freqs), n_freqs - 1)

        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.Conv2d(self.hid_feature, self.output_channel, (1, 1)),
            nn.InstanceNorm2d(self.output_channel, affine=True),
            nn.PReLU(self.output_channel),
            nn.Conv2d(self.output_channel, self.output_channel, (1, 1))
        )
        
        # 为实部和虚部实例化带有频率权重的 Sigmoid
        self.lsigmoid_r = LearnableSigmoid2D(n_freqs, beta=self.beta, split_idx=self.split_idx, high_freq_weight=self.high_freq_weight)
        self.lsigmoid_i = LearnableSigmoid2D(n_freqs, beta=self.beta, split_idx=self.split_idx, high_freq_weight=self.high_freq_weight)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x) # [B, 2, T, F]

        mask_r = x[:, 0, :, :] 
        mask_i = x[:, 1, :, :] 

        mask_r = rearrange(mask_r, 'b t f -> b f t')
        mask_i = rearrange(mask_i, 'b t f -> b f t')

        # 此时 lsigmoid 的输出范围是 [0, beta * weight_mask]
        out_r = self.lsigmoid_r(mask_r)
        out_i = self.lsigmoid_i(mask_i)
        
        # 将区间平移，得到最终的复数掩蔽范围: [-beta*weight_mask, beta*weight_mask]
        limit_r = self.beta * self.lsigmoid_r.weight_mask
        limit_i = self.beta * self.lsigmoid_i.weight_mask
        
        mask_r = out_r * 2 - limit_r
        mask_i = out_i * 2 - limit_i

        # 恢复维度 [B, 1, T, F]
        mask_r = rearrange(mask_r, 'b f t -> b t f').unsqueeze(1)
        mask_i = rearrange(mask_i, 'b f t -> b t f').unsqueeze(1)

        return mask_r, mask_i