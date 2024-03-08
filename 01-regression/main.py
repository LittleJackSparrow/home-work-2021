import csv
import os

import matplotlib.pyplot as plt
import torch
from matplotlib.pyplot import figure

from dataset import prep_dataloader
from model import NeuralNet, train, test
from utils.settings import get_device


def plot_learning_curve(loss_record, title=''):
    """Plot learning curve of your DNN (train & dev loss) """
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    """ Plot prediction of your DNN """
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 3000,  # maximum number of epochs
    'batch_size': 270,  # mini-batch size for dataloader
    'optimizer': 'AdamW',  # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {  # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,  # learning rate of SGD
        # 'momentum': 0.95,  # momentum for SGD Adam不需要单独设置动量
        'weight_decay': 0.001
    },
    'early_stop': 200,  # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}

if __name__ == '__main__':
    device = get_device()  # get the current available device ('cpu' or 'cuda')
    os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
    # target_only = False  # TODO: Using 40 states & 2 tested_positive features
    target_only = True
    tr_path = "D:/01-workspace/github/home-work-2021/dataset/01-covid/covid.train.csv"
    tt_path = "D:/01-workspace/github/home-work-2021/dataset/01-covid/covid.test.csv"
    tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
    dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
    tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)
    # Model的features为数据的features的长度
    model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device
    # 训练并验证
    model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

    plot_learning_curve(model_loss_record, title='deep model')

    del model
    model = NeuralNet(tr_set.dataset.dim).to(device)
    ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
    model.load_state_dict(ckpt)
    plot_pred(dv_set, model, device)  # Show prediction on the validation set
    #
    # preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
    # save_pred(preds, 'D:/01-workspace/github/home-work-2021/dataset/01-covid/pred.csv')  # save prediction file to pred.csv
