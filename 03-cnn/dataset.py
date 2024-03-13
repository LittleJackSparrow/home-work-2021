from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder


# 定义加载图像的函数
def image_loader(image_path):
    return Image.open(image_path)


def prep_dataloader(path, config, tfm):
    # 类别是通过文件夹得到的
    data_set = DatasetFolder(path, loader=image_loader, extensions=("jpg",), transform=tfm)
    data_loader = DataLoader(data_set, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    return data_set, data_loader

    # valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions=("jpg",), transform=tfm)
    # unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    # test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
