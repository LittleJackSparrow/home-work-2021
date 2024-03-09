import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def split_data(path, val_ratio):
    print('Loading data ...')
    train = np.load(path + 'train_11.npy')
    train_label = np.load(path + 'train_label_11.npy')
    test = np.load(path + 'test_11.npy')
    print('Size of training data: {}'.format(train.shape))
    print('Size of testing data: {}'.format(test.shape))
    percent = int(train.shape[0] * (1 - val_ratio))
    train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
    print('Size of training set: {}'.format(train_x.shape))
    print('Size of validation set: {}'.format(val_x.shape))
    return train, train_label, test, train_x, train_y, val_x, val_y


def prep_dataloader(train_x, train_y, conf, shuffle):
    dataset = TIMITDataset(train_x, train_y)
    data_loader = DataLoader(dataset, batch_size=conf['batch_size'], shuffle=shuffle)
    return dataset, data_loader


class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.integer)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)
