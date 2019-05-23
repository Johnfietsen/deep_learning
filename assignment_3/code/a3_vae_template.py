import argparse
from datetime import datetime
from scipy.stats import norm

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.utils import make_grid, save_image

from datasets.bmnist import bmnist
import numpy as np


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, input_dim=784):
        super().__init__()

        self._mean = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, z_dim))

        self._std = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, z_dim),
                                  nn.ReLU())

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        mean, std = self._mean(input), self._std(input)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, output_dim=784):
        super().__init__()

        self._mean = nn.Sequential(nn.Linear(z_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, output_dim),
                                   nn.Sigmoid())

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self._mean(input)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, image_dim=784):
        super().__init__()

        self._z_dim = z_dim

        self._image_side = int(np.sqrt(image_dim))
        self._image_dim = image_dim
        self._encoder = Encoder(hidden_dim, z_dim, image_dim)
        self._decoder = Decoder(hidden_dim, z_dim, image_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        input = input.view(-1, self._image_dim)

        mean_enc, std_enc = self._encoder(input)
        mean_dec = self._decoder(mean_enc +
                                 torch.randn(std_enc.size()) * std_enc)

        loss_rec = (1 / 2) * (torch.sum(- torch.log(std_enc) + std_enc ** 2 +
                                        mean_enc ** 2, dim=1) - 1)
        loss_reg = torch.sum(input * torch.log(mean_dec) + (1 - input) *
                             torch.log(1 - mean_dec), dim=1)

        average_negative_elbo = torch.mean(loss_rec - loss_reg, dim=0)

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        sampled_ims = self._decoder(torch.randn((n_samples, self._z_dim)))\
                          .view(- 1, 1, self._image_side, self._image_side)

        im_means = torch.mean(sampled_ims)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """

    average_epoch_elbo = 0

    nr_data = len(data)
    for batch in data:
        loss = model(batch)

        if model.training:
            model.zero_grad()
            loss.backward()
            optimizer.step()

        average_epoch_elbo += loss.item() / nr_data

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    start = datetime.now().strftime("%Y-%m-%d_%H:%M")
    f_train = open('elbos_vae/train_' + start + '.txt', 'w')
    f_valid = open('elbos_vae/valid_' + start + '.txt', 'w')

    sample_images, _ = model.sample(16)
    grid = make_grid(sample_images.detach(), nrow=4, padding=0)
    save_image(grid, 'samples_vae/start_' + str(start) + '.png')

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos

        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")
        f_train.write(', ' + str(train_elbo))
        f_valid.write(', ' + str(val_elbo))

        if epoch == ARGS.epochs / 2 - 1 or epoch == ARGS.epochs / 2 - 0.5:
            sample_images, _ = model.sample(16)
            grid = make_grid(sample_images.detach(), nrow=4, padding=0)
            save_image(grid, 'samples_vae/middle_' + str(start) + '.png')

    if ARGS.zdim == 2:
        with torch.no_grad():
            grid = np.empty((420, 420))
            for i, y in enumerate(np.linspace(.05, .95, 15)):
                for j, x in enumerate(np.linspace(.05, .95, 15)):
                    # Use ppf to sample the correct spot that matches the 0.05 to 0.95 area
                    noise = torch.tensor(np.array([[norm.ppf(x), norm.ppf(y)]])\
                            .astype('float32'))
                    means = model._decoder(noise)
                    grid[(14-i)*28:(15-i)*28, j*28:(j+1)*28] = means\
                                            .reshape(28, 28).data.numpy()

            plt.figure(figsize=(8, 10))
            plt.imshow(grid, origin="upper", cmap="gray")
            plt.tight_layout()
            plt.savefig('manifold_' + str(start) + '.png')

    sample_images, _ = model.sample(16)
    grid = make_grid(sample_images.detach(), nrow=4, padding=0)
    save_image(grid, 'samples_vae/end_' + str(start) + '.png')
    # save_grid(sample_images.detach(), 'samples/end_' + str(start))

    f_train.close()
    f_valid.close()
    # save_elbo_plot(train_curve, val_curve, 'elbo.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
