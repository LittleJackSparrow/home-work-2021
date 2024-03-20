import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataset


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    """
    Input shape: (N, in_dim)
    Output shape: (N, 3, 64, 64)
    """

    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(input_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(input_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            # 最后一次经过反卷积层后不需要再进行批量归一化，是因为每个像素点都是独立的，并不需要通过批量归一化提升特征的表示能力
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            # Tanh激活函数的作用是将生成的向量值限制在[-1, 1]的范围内
            nn.Tanh()
        )
        # 随机初始化权重
        self.apply(weights_init)

    def forward(self, x):
        # input:64 * 100  output:64 * dim * 8 * 4 * 4
        y = self.l1(x)
        # input: 64 * dim * 8 * 4 * 4 output: 64 * 512 * 4 * 4
        y = y.view(y.size(0), -1, 4, 4)
        # input: 通道数512=dim*8
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):
    """
    Input shape: (N, 3, 64, 64)
    Output shape: (N, )
    """

    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, 5, 2, 2)),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )

        """ Medium: Remove the last sigmoid layer for WGAN. """
        self.ls = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_dim, dim, 5, 2, 2)),
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.utils.spectral_norm(nn.Conv2d(dim * 8, 1, 4)),
            # nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


z_dim = 100
workspace_dir = "D:/01-workspace/github/dataset/06-anima_face"
ckpt_dir = os.path.join(workspace_dir, 'checkpoints')


def train_base(device):
    # Training hyperparameters
    batch_size = 64
    # torch.randn：标准正太分布，均值位0，方差位1
    z_sample = Variable(torch.randn(100, z_dim)).to(device)
    lr = 1e-4

    """ Medium: WGAN, 50 epoch, n_critic=5, clip_value=0.01 """
    n_epoch = 50  # 50
    n_critic = 5  # 5
    clip_value = 0.01

    log_dir = os.path.join(workspace_dir, 'logs')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Model
    G = Generator(in_dim=z_dim).to(device)
    D = Discriminator(3).to(device)
    G.train()
    D.train()

    # Loss
    criterion = nn.BCELoss()

    """ Medium: Use RMSprop for WGAN. """
    # Adam：动量+自适应率学习率
    # RMSprop：自适应率学习率
    # Optimizer
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    # opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
    # opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)

    # DataLoader
    dataset = get_dataset(os.path.join(workspace_dir, 'faces'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader, G, opt_G, D, opt_D, criterion, n_epoch, n_critic, z_sample, log_dir, clip_value


lambda_gp = 1


# Calculate the gradient penalty
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(d_interpolates.size()).to(real_samples.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_loop(device):
    dataloader, G, opt_G, D, opt_D, criterion, n_epoch, n_critic, z_sample, log_dir, clip_value = train_base(device)
    steps = 0
    # epoch_index：索引 epoch：具体迭代的值
    for epoch_index, epoch in enumerate(range(n_epoch)):
        progress_bar = tqdm(dataloader)
        for index, data in enumerate(progress_bar):
            # [64, 3, 64, 64]
            imgs = data
            imgs = imgs.to(device)
            # 64
            batch_size = imgs.size(0)

            # ============================================
            #  Train D
            # ============================================
            z = Variable(torch.randn(batch_size, z_dim)).to(device)
            r_imgs = Variable(imgs).to(device)
            f_imgs = G(z)

            """ Medium: Use WGAN Loss. """
            # # Label
            # r_label = torch.ones((batch_size)).to(device)
            # f_label = torch.zeros((batch_size)).to(device)
            #
            # # Model forwarding
            # # detach()：分离后的tensor不再具有梯度信息
            # r_logit = D(r_imgs.detach())
            # f_logit = D(f_imgs.detach())
            #
            # # Compute the loss for the discriminator.
            # r_loss = criterion(r_logit, r_label)
            # f_loss = criterion(f_logit, f_label)
            # loss_D = (r_loss + f_loss) / 2
            gradient_penalty = compute_gradient_penalty(D, r_imgs.data, f_imgs.data)
            # WGAN Loss 加上gradient penalty才是真的WGAN
            loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs)) + lambda_gp * gradient_penalty

            # Model backwarding
            D.zero_grad()
            loss_D.backward()

            # Update the discriminator.
            opt_D.step()

            """ Medium: Clip weights of discriminator. """
            """ 最初的WGAN就是使用的裁剪 """
            # for p in D.parameters():
            #     p.data.clamp_(-clip_value, clip_value)

            # ============================================
            #  Train G
            # ============================================
            if steps % n_critic == 0:
                # Generate some fake images.
                z = Variable(torch.randn(batch_size, z_dim)).to(device)
                f_imgs = G(z)

                # Model forwarding
                # f_logit = D(f_imgs)

                """ Medium: Use WGAN Loss"""
                # Compute the loss for the generator.
                # loss_G = criterion(f_logit, r_label)
                # WGAN Loss
                loss_G = -torch.mean(D(f_imgs))

                # Model backwarding
                G.zero_grad()
                loss_G.backward()

                # Update the generator.
                opt_G.step()

            steps += 1

            # Set the info of the progress bar
            #   Note that the value of the GAN loss is not directly related to
            #   the quality of the generated images.
            progress_bar.set_postfix({
                'Loss_D': round(loss_D.item(), 4),
                'Loss_G': round(loss_G.item(), 4),
                'Epoch': epoch_index + 1,
                'Step': steps,
            })

        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(log_dir, f'Epoch_{epoch + 1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')

        # Show generated images in the jupyter notebook.
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        G.train()

        if (epoch_index + 1) % 5 == 0 or epoch_index == 0:
            # Save the checkpoints.
            torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))


def test_loop(device):
    G = Generator(z_dim)
    G.load_state_dict(torch.load(os.path.join(ckpt_dir, 'G.pth')))
    G.eval()
    G.to(device)
    # Generate 1000 images and make a grid to save them.
    n_output = 1000
    z_sample = Variable(torch.randn(n_output, z_dim)).to(device)
    imgs_sample = (G(z_sample).data + 1) / 2.0
    log_dir = os.path.join(workspace_dir, 'logs')
    filename = os.path.join(log_dir, 'result.jpg')
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)

    # Show 32 of the images.
    grid_img = torchvision.utils.make_grid(imgs_sample[:32].cpu(), nrow=10)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

    # Save the generated images.
    # os.makedirs('output', exist_ok=True)
    # for i in range(1000):
    #     torchvision.utils.save_image(imgs_sample[i], f'output/{i + 1}.jpg')
