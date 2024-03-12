from PIL import Image
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, DataLoader


def prep_dataloader(path, config, tfm):
    data_set = DatasetFolder(path, loader=lambda x: Image.open(x), extensions=("jpg",), transform=tfm)
    data_loader = DataLoader(data_set, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    return data_set, data_loader

    # valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions=("jpg",), transform=tfm)
    # unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    # test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
