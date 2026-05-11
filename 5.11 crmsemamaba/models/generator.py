import torch
import torch.nn as nn
from einops import rearrange
from .mamba_block import TFMambaBlock
from .codec_module import DenseEncoder, CRMDecoder

class SEMamba(nn.Module):
    def __init__(self, cfg):
        super(SEMamba, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = cfg['model_cfg']['num_tfmamba'] if cfg['model_cfg']['num_tfmamba'] is not None else 4

        self.dense_encoder = DenseEncoder(cfg)
        self.TSMamba = nn.ModuleList([TFMambaBlock(cfg) for _ in range(self.num_tscblocks)])
        self.crm_decoder = CRMDecoder(cfg)

    def forward(self, noisy_mag, noisy_pha):
        noisy_mag_in = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  
        noisy_pha_in = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  

        x = torch.cat((noisy_mag_in, noisy_pha_in), dim=1)  # [B, 2, T, F]
        x = self.dense_encoder(x)

        for block in self.TSMamba:
            x = block(x)

        # 获取带权重的 CRM 实部和虚部
        mask_r, mask_i = self.crm_decoder(x)
        
        # 维度转换以对齐相乘 [B, F, T]
        mask_r = rearrange(mask_r, 'b c t f -> b f t c').squeeze(-1)
        mask_i = rearrange(mask_i, 'b c t f -> b f t c').squeeze(-1)

        # 构建原信号的复数实部与虚部
        noisy_real = noisy_mag * torch.cos(noisy_pha)
        noisy_imag = noisy_mag * torch.sin(noisy_pha)

# 复数乘法
        denoised_real = mask_r * noisy_real - mask_i * noisy_imag
        denoised_imag = mask_r * noisy_imag + mask_i * noisy_real

       # ================= 核心修复：开启 Float32 安全隔离舱 =================
        # 1. 在执行物理乘法前，将掩蔽矩阵和带噪信号全部提升为 float32
        mask_r_f32 = mask_r.float()
        mask_i_f32 = mask_i.float()
        
        noisy_real_f32 = (noisy_mag * torch.cos(noisy_pha)).float()
        noisy_imag_f32 = (noisy_mag * torch.sin(noisy_pha)).float()

        # 2. 在绝对安全的 32 位空间内执行 CRM 复数乘法
        denoised_real_f32 = mask_r_f32 * noisy_real_f32 - mask_i_f32 * noisy_imag_f32
        denoised_imag_f32 = mask_r_f32 * noisy_imag_f32 + mask_i_f32 * noisy_real_f32

        # 3. 梯度保护：使用 float32 下生效的有效 eps，防止网络在静音段/噪声压制段崩溃
        eps = 1e-7 
        
        # 加上 eps 防止 sqrt 处在 0 点导致反向传播梯度爆炸
        denoised_mag_f32 = torch.sqrt(denoised_real_f32**2 + denoised_imag_f32**2 + eps)
        
        # 给 atan2 的分母和分子加极小扰动，防止出现 atan2(0,0) 的 NaN 梯度
        denoised_pha_f32 = torch.atan2(denoised_imag_f32 + eps, denoised_real_f32 + eps)

        # 4. 降维打击：计算完毕后，安全退回网络原本的精度（如 float16），确保不破坏下游框架
        denoised_mag = denoised_mag_f32.type_as(noisy_mag)
        denoised_pha = denoised_pha_f32.type_as(noisy_pha)
        denoised_com = torch.stack((denoised_real_f32.type_as(noisy_mag), denoised_imag_f32.type_as(noisy_mag)), dim=-1)
        # ====================================================================

        return denoised_mag, denoised_pha, denoised_com