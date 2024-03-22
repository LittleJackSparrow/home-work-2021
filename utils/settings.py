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


def get_device(accelerator=None):
    # 检查是否可用CUDA，如果可用，则适用GPU，如果不可用，则使用CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if accelerator is None:
        return device
    return accelerator.device


# Documentation for the toolkit:  https://huggingface.co/docs/accelerate/
def get_accelerate(fp16_training=False):
    # 检查是否开启混合精度训练(FP16)，如果是，则使用Accelerator初始化加速器
    # Mixed Precision Training：混合精度训练，某些适用半精度，某些使用单精度，降低内存消耗，加速训练，并且尽量减少损失
    # fp16：使用半精度（16位浮点数）浮点数来存储模型参数和中间计算结果，以降低内存消耗，加速训练
    # fp32：单精度（132位浮点数）浮点数
    # 除了混合精度训练外，还有Mixed Precision Optimization：混合精度优化，在Gradient descent更新步骤时使用单精度计算梯度，并将单精度转换为半精度浮点数以减少计算开销
    if fp16_training:
        from accelerate import Accelerator
        # 初始化Accelerator，设置为启用fp16训练
        accelerator = Accelerator(mixed_precision='fp16')
        return accelerator


def prepare(accelerator=None, model=None, optimizer=None, train_loader=None):
    if accelerator is None:
        return model, optimizer, train_loader
    return accelerator.prepare(model, optimizer, train_loader)
