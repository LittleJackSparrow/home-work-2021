import csv

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup


def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    """ Generates a dataset, then is put into a dataloader. """
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)  # Construct dataset
    # shuffle为True表示在每个epoch开始前数据随机打乱；为False表示不打乱
    # drop_last为True，最后一个批次不满足batch_size时则丢弃；为False时，不丢弃
    # num_workers：使用多少个线程同时加载数据
    # pin_memory：是否将数据加载到CUDA固定的内存区域中，在使用GPU时，会将数据加载到固定的内存区域，提高传输速度
    dataloader = DataLoader(dataset, batch_size, shuffle=(mode == 'train'), drop_last=False, num_workers=n_jobs, pin_memory=True)  # Construct dataloader
    return dataloader


class COVID19Dataset(Dataset):
    """ Dataset for loading and preprocessing the COVID19 dataset """

    def __init__(self, path, mode='train', target_only=False):
        self.mode = mode
        # Read data into numpy arrays
        # 读取csv文件中数据，剔除第一行和第一列
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)
        # 整个文件除去第一列的便函之外包含94列，剔除最后一列是target之外，有93列，索引是0~92
        if not target_only:
            features = list(range(93))
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            features = list(range(40)) + [57, 75]
            # pass

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, features]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            # feats是个数字列表，切片时可以按照列表的索引进行切片
            # data的shape是2700 x 93
            data = data[:, features]

            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]

            # Convert data into PyTorch tensors-
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        # Normalization，将数据的特征值缩放到一个统一的范围内，以消除特征之间的量纲差异，提高模型的性能和收敛速度
        # dim用于指定沿着哪个维度计算均值，keepdim用于指定是否保持结果的维度不变
        # Batch Normalization的计算方式是(当前值 - 均值) / 标准差
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'.format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)
