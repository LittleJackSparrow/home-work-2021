import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.datasets import DatasetFolder

from dataset import prep_dataloader, image_loader
from model import Classifier, train_val, test
from utils.settings import get_device

# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # randon clip 8*8 pixel
    # transforms.RandomCrop(16, padding=None),
    # 50%的概率将图片水平翻转，也就是左边移动到右边，右边移动到左边
    transforms.RandomHorizontalFlip(p=0.5),
    # 对图像进行颜色增强，主要是调整图像的亮度，亮度的变化范围在 [-0.5, 0.5] 之间
    transforms.ColorJitter(brightness=0.5),
    # 随机将图像沿着中心进行最多5度的随机旋转
    transforms.RandomRotation(5),
    # 随机选择图像的四个角点，并在这些角点周围进行透视变换，从而产生一个畸变的效果，使得图像看起来具有透视感
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor()
])
# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

if __name__ == '__main__':
    config = {
        # Batch size for training, validation, and testing.
        # A greater batch size usually gives a more stable gradient.
        # But the GPU memory is limited, so please adjust it carefully.
        "batch_size": 128,
        "n_epochs": 80,
        'optimizer': 'Adam',
        "optim_hparas": {
            "lr": 0.0003,
            "weight_decay": 1e-5
        },
        "model_path": "models/model.pth"
    }
    path = "D:/01-workspace/github/home-work-2021/dataset/03-food-11/"
    # Construct datasets.
    # The argument "loader" tells how torchvision reads the data.
    train_set, train_loader = prep_dataloader(path + "training/labeled", config, train_tfm)
    valid_set, valid_loader = prep_dataloader(path + "validation", config, test_tfm)
    unlabeled_set, unlabeled_loader = prep_dataloader(path + "training/unlabeled", config, train_tfm)
    test_set = DatasetFolder(path + "testing", loader=image_loader, extensions=("jpg",), transform=test_tfm)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

    # "cuda" only when GPUs are available.
    device = get_device()
    # Initialize a model, and put it on the device specified.
    model = Classifier().to(device)
    model.device = device
    # 使用resnet18替换自定义的模型
    # resnet18 = models.resnet18(pretrained=False)
    # in_features = resnet18.fc.in_features
    # resnet18.fc = nn.Linear(in_features, 11)
    train_val(model, config, train_set, train_loader, unlabeled_loader, valid_loader, device)

    predictions = test(model, test_loader, device)
    # Save predictions into the file.
    with open("predict.csv", "w") as f:
        # The first row must be "Id, Category"
        f.write("Id,Category\n")
        # For the rest of the rows, each image id corresponds to a predicted class.
        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")
