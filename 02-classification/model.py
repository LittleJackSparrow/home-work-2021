import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(429, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 39)

        self.act_fn = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.25)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.layer1(x)
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.act_fn(x)

        x = self.out(x)

        return x

    def cal_loss(self, model, outputs, labels, device):
        loss = self.criterion(outputs, labels)
        lambda_reg = 0.00075
        # 计算L2正则化项
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)
        loss += lambda_reg * l2_reg
        return loss


def train_val(config, model, train_set, val_set, train_loader, val_loader, device):
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    best_acc = 0.0
    for epoch in range(config["n_epochs"]):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        model.train()  # set the model to training mode
        for i, data in enumerate(train_loader):
            optimizer.zero_grad() # 梯度置0
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # forward propagation
            print(f'outputs.shape={outputs.shape}')
            batch_loss = model.cal_loss(model, outputs, labels, device)  # compute the loss
            _, train_pred = torch.max(outputs, 1)  # 只是做了每行取最大值的操作，(values:[每一行最大概率值的元组], indices:[每一行最大概率索引的元组])
            batch_loss.backward()
            optimizer.step()

            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()

        # validation
        if len(val_set) > 0:
            model.eval()  # set the model to evaluation mode
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    batch_loss = model.cal_loss(model, outputs, labels, device)
                    _, val_pred = torch.max(outputs, 1)

                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                    val_loss += batch_loss.item()
                # :03d ->:格式化字符串的开始 03表示如果输出的位数不足3位数，则在左侧用0补充，以满足位数 d输出整数
                # :3.6f -> :格式化字符串的开始 3.6 3表示输出的最小宽度，如果不足3，在左侧用空格补充，6表示小数点最大的位，f表示输出浮点数
                # train_loss / len(train_loader):表示每个批次的平均损失
                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, config["n_epochs"], train_acc / len(train_set), train_loss / len(train_loader), val_acc / len(val_set), val_loss / len(val_loader)
                ))

                # if the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), config["model_path"])
                    print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, config["n_epochs"], train_acc / len(train_set), train_loss / len(train_loader)
            ))

    # if not validating, save the last epoch
    if len(val_set) == 0:
        torch.save(model.state_dict(), config["model_path"])
        print('saving model at last epoch')


def test_model(config, device, test_loader):
    model = Classifier().to(device)
    model.load_state_dict(torch.load(config["model_path"]))
    predict = []
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability

            for y in test_pred.cpu().numpy():
                predict.append(y)
    return predict
