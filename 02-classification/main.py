import gc

import utils.settings as setting
from dataset import split_data, prep_dataloader
from model import Classifier, train_val, test_model
import torch

if __name__ == '__main__':
    # Hint1：post-processing，后处理怎么处理？不懂
    # Hint2：模型修改，layers,dimension,activation function
    # Hint3: batch size?optimizer?lr?n_epochs?
    # Hint4: batch norm?dropout?regularization?
    # acc=0.74
    config = {
        "batch_size": 64,
        "n_epochs": 20,
        'optimizer': 'Adam',
        "optim_hparas": {
            "lr": 0.0001
        },
        "model_path": "models/model.pth"
    }

    path = 'D:/01-workspace/github/home-work-2021/dataset/02-phoneme/'
    train, train_label, test, train_x, train_y, val_x, val_y = split_data(path, 0.2)
    train_set, train_loader = prep_dataloader(train_x, train_y, config, True)  # only shuffle the training data
    val_set, val_loader = prep_dataloader(val_x, val_y, config, False)

    # 清理缓存
    del train, train_label, train_x, train_y, val_x, val_y
    gc.collect()

    device = setting.get_device()

    model = Classifier().to(device)
    train_val(config, model, train_set, val_set, train_loader, val_loader, device)

    # # create testing dataset
    # test_set, test_loader = prep_dataloader(test, None, config, False)  # only shuffle the training data
    # # create model and load weights from checkpoint
    # predict = test_model(config, device, test_loader)
    # with open('prediction.csv', 'w') as f:
    #     f.write('Id,Class\n')
    #     for i, y in enumerate(predict):
    #         f.write('{},{}\n'.format(i, y))
