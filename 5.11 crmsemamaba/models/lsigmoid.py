import torch
import torch.nn as nn

class LearnableSigmoid1D(nn.Module):
    def __init__(self, in_features, beta=1):
        super(LearnableSigmoid1D, self).__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requires_grad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid2D(nn.Module):
    """
    带有频域权重先验的 Learnable Sigmoid。
    允许为高频段（如蛙叫声频段）分配更大的掩蔽上限（beta）和更陡峭的初始斜率（slope）。
    """
    def __init__(self, in_features, beta=1.0, split_idx=None, high_freq_weight=1.0):
        super(LearnableSigmoid2D, self).__init__()
        self.beta = beta
        
        # 初始化斜率 slope [in_features, 1]
        slope_init = torch.ones(in_features, 1)
        # 初始化频域幅度权重掩膜 [in_features, 1]
        weight_mask = torch.ones(in_features, 1)
        
        # 注入先验知识：如果指定了分界点，则强化高频区的掩蔽能力
        if split_idx is not None:
            slope_init[split_idx:, 0] *= high_freq_weight  # 高频初始更陡峭（类似手术刀）
            weight_mask[split_idx:, 0] *= high_freq_weight # 高频掩蔽上限更大
            
        self.slope = nn.Parameter(slope_init)
        self.slope.requires_grad = True
        
        # 使用 register_buffer 注册掩膜，使其随模型保存至GPU，但不作为网络参数更新
        self.register_buffer('weight_mask', weight_mask)

    def forward(self, x):
        # 实际的动态范围变成了 [0, beta * weight_mask]
        return (self.beta * self.weight_mask) * torch.sigmoid(self.slope * x)