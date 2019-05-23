import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self._model = nn.Sequential(nn.Linear(latent_dim, 128),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(128, 256),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(256, 512),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(512, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(1024, 784),
                                    nn.Tanh())

    def forward(self, z):

        return self._model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self._model = nn.Sequential(nn.Linear(784, 512),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(512, 256),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(256, 1),
                                    nn.Sigmoid())

    def forward(self, img):

        return self._model(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D,
          device):

    start = datetime.now().strftime("%Y-%m-%d_%H:%M")
    f_G = open('losses_gan/G_' + start + '.txt', 'w')
    f_D = open('losses_gan/D_' + start + '.txt', 'w')

    for epoch in range(args.n_epochs):

        if epoch == 1:
            grid = make_grid(out_G[:25].view(-1, 1, 28, 28), nrow=5, padding=0)
            save_image(grid, 'samples_gan/start_' + str(start) + '.png')

        if epoch == args.n_epochs / 2 - 1 or epoch == args.n_epochs / 2 - 0.5:
            grid = make_grid(out_G[:25].view(-1, 1, 28, 28), nrow=5, padding=0)
            save_image(grid, 'samples_gan/middle_' + str(start) + '.png')

        for i, (imgs, _) in enumerate(dataloader):

            imgs = imgs.view(-1, 784)
            batch_size = imgs.shape[0]

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # imgs.cuda()
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # Train Generator
            out_G = generator(torch.randn(batch_size, args.latent_dim))\
                    .to(device)
            loss_G = torch.sum(- torch.log(discriminator(out_G))).to(device)
            optimizer_G.zero_grad()
            loss_G.backward(retain_graph=True)
            optimizer_G.step()

            # Train Discriminator
            out_G = generator(torch.randn(batch_size, args.latent_dim))\
                    .to(device)
            out_D_f = discriminator(out_G).to(device)
            out_D_r = discriminator(imgs).to(device)
            loss_D = torch.sum(- (torch.log(out_D_r) + torch.log(1 - out_D_f)))\
                     .to(device)
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)

                print("[{}] Step {:06d}/{:06d}, Batch Size = {} \
                       Loss G = {:.4f}, Loss D = {:.4f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), batches_done,
                        args.n_epochs * len(dataloader), args.batch_size, \
                        loss_G, loss_D
                ))

                f_G.write(', ' + str(loss_G.item()))
                f_D.write(', ' + str(loss_D.item()))

    grid = make_grid(out_G[:25].view(-1, 1, 28, 28), nrow=5, padding=0)
    save_image(grid, 'samples_gan/end_' + str(start) + '.png')

    f_G.close()
    f_D.close()



def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator().to(args.device)
    discriminator = Discriminator().to(args.device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D,
          args.device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")

    args = parser.parse_args()

    main()
