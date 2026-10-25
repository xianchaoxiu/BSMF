import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
import math
import numpy as np


# channel_pruning_resnet.py - get_channel_L2() 用于计算通道重要性
def get_channel_L2(Conv_weight):
    channel_norms = torch.sum(Conv_weight.weight.data.abs(), dim=(1, 2, 3))  # 按输出通道维度聚合 ∑∣W∣
    den = channel_norms.max() - channel_norms.min()
    return (channel_norms - channel_norms.min()) / (den + 1e-12)###返回归一化分数
