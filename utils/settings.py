import random

import numpy as np
import torch

"""
    定义通用方法
"""


def same_seeds(seed):
    """
    固定随机种子，以确保在使用随机性的深度学习模型时，每次运行的结果都是可重复的
    :param seed: 随机种子
    :return: 无返回值
    """
    # 设置python内置的random的随机种子
    random.seed(seed)
    # 设置numpy的随机种子
    np.random.seed(seed)
    # 设置pytorch的随机种子
    torch.manual_seed(seed)
    # 如果GPU是可用的，设置GPU的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # CuDNN（cuda Deep Neural Network Library）是NVIDIA提供的用于深度学习任务的GPU加速库
    # 禁用CuDNN的自动优化功能，以确保每次运行结果一致
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    """
    Get device (if GPU is available, use GPU)
    :return: 无返回值
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'
