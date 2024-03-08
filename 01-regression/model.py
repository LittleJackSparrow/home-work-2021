import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup


class NeuralNet(nn.Module):
    """ A simple fully-connected deep neural network """

    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        # 定义模型的层级，先是经过一个Linear层，然后经过ReLU函数，最后输出到Linear，得到一个scalar
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        # 定义损失函数
        # reduction=none，返回给个样本的损失值
        # reduction=mean，返回所有样本损失值的均值
        # reduction=sum，返回所有样本损失值的加总
        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        """ Given input of size (batch_size x input_dim), compute output of the network """
        # squeeze(1)：如果张量的第二个维度为1，则去除此维度；squeeze(2)：如果张量的第三个维度为1，则去除此维度
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        """ Calculate loss """
        # TODO: you may implement L2 regularization here
        # L2正则化的目的是限制模型参数的大小，使其不能太大，从而避免模型过度拟合训练数据
        return self.criterion(pred, target)


def train(tr_set, dv_set, model, config, device):
    """ DNN training """

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    # getattr(torch.optim, config['optimizer'])：根据配置定义优化器
    # (model.parameters(), **config['optim_hparas'])：设置优化器的参数
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=160, num_training_steps=n_epochs * len(tr_set))
    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}  # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()  # 设置model为training模式 set model to training mode
        for x, y in tr_set:  # 通过迭代得到features和targets iterate through the dataloader
            optimizer.zero_grad()  # 设置梯度为0 set gradient to zero
            x, y = x.to(device), y.to(device)  # 将数据移动到可用的设备上 move data to device (cpu/cuda)
            pred = model(x)  # 执行forward forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # 计算损失 compute loss
            mse_loss.backward()  # 反向传播 compute gradient (backpropagation)
            optimizer.step()  # 更新参数 update model with optimizer
            scheduler.step()
            loss_record['train'].append(mse_loss.detach().cpu().item())  # detach：将损失值提取出来，不包含梯度；cpu：将数据放到cpu上；item：取值
        # print(f"train loss:{loss_record['train'][-1]}")

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        # 如果 验证的损失值小于上个epoch的损失值，则说明有改进，保存模型
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


def dev(dv_set, model, device):
    model.eval()  # 设置模型到验证模式 set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:  # 通过dataloder迭代出数据 iterate through the dataloader
        x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # 关闭梯度计算 disable gradient calculation
            pred = model(x)  # 前向传播 forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # 计算损失 compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # 每个损失的值 x 数据长度，然后加总 accumulate loss
    total_loss = total_loss / len(dv_set.dataset)  # 加总后的数据取平均 compute averaged loss

    return total_loss


def test(tt_set, model, device):
    model.eval()  # set model to evalutation mode
    preds = []
    for x in tt_set:  # iterate through the dataloader
        x = x.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            preds.append(pred.detach().cpu())  # collect prediction
    preds = torch.cat(preds, dim=0).numpy()  # concatenate all predictions and convert to a numpy array
    return preds
