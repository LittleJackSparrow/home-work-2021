import glob
import os

import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CrypkoDataset(Dataset):
    def __init__(self, file_names, transform):
        self.transform = transform
        self.file_names = file_names
        self.num_samples = len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        # 1. Load the image
        img = torchvision.io.read_image(file_name)
        # 2. Resize and normalize the images using torchvision.
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


def get_dataset(root):
    file_names = glob.glob(os.path.join(root, '*'))
    # 1. Resize the image to (64, 64)
    # 批量归一化之后的数据一般都在[0, 1]之间，做-0.5，然后除以0.5的操作，可以把数据移动到[-1, 1]之间
    # 2. Linearly map [0, 1] to [-1, 1]
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    return CrypkoDataset(file_names, transform)


if __name__ == '__main__':
    workspace_dir = "D:/01-workspace/github/dataset/06-anima_face"
    dataset = get_dataset(os.path.join(workspace_dir, 'faces'))

    # images = [dataset[i] for i in range(16)]
    # grid_img = torchvision.utils.make_grid(images, nrow=4)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(grid_img.permute(1, 2, 0))
    # plt.show()

    images = [(dataset[i] + 1) / 2 for i in range(16)]
    grid_img = torchvision.utils.make_grid(images, nrow=4)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
