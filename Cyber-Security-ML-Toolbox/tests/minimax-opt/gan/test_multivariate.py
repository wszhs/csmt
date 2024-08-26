import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)

import os
import torch

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from csmt.zoopt.vegans.GAN import WassersteinGAN
from csmt.zoopt.vegans.utils import plot_losses

z_dim = 32  # noise vector size
N = 20  # size of sampled vectors

cov_mat = np.zeros(shape=(N, N))
cov_mat[0:10, 0:10] = 0.8
np.fill_diagonal(cov_mat, 1.)

plt.imshow(cov_mat)
plt.title('Data generating process covariance matrix')
plt.colorbar();
plt.show()

X_train = np.random.multivariate_normal(mean=np.array([0. for _ in range(N)]), cov=cov_mat, size=10000)

# re-cover matrix
cov_mat_recovered = np.cov(X_train.T)

# plt.imshow(cov_mat_recovered)
# plt.title('Sample cov matrix from multivariate Gaussian');
# plt.colorbar();
# plt.show()

# plt.hist(X_train.flatten(), bins=50)
# plt.title('Histogram of (flattened) training set samples');
# plt.show()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.hidden_part = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.LeakyReLU(0.1),
        )
        self.output = nn.Linear(32, N)

    def forward(self, x):
        x = self.hidden_part(x)
        x = self.output(x)
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.hidden_part = nn.Sequential(
            nn.Linear(N, 10),
            nn.LeakyReLU(0.1),
            nn.Linear(10, 1)
        )
        self.output = nn.Identity()

    def forward(self, x):
        x = self.hidden_part(x)
        x = self.output(x)
        return x
    
critic = Critic()
generator = Generator()

def plot_generator_cov(generator, n=10000):
    """ Try the generator on a noise mini-batch and compute sample covariance matrix
    """
    z = torch.from_numpy(np.random.randn(n, z_dim)).float()
    samples = generator(z)
    cov_mat_est = np.cov(samples.detach().numpy().T)
    plt.imshow(cov_mat_est)
    plt.title('sample covariance matrix from generator network');
    plt.colorbar();
    
plot_generator_cov(generator)

optimizer = {
    "Generator": torch.optim.Adam,
    "Adversary": torch.optim.Adam
}
optimizer_kwargs = {
    "Generator": {"lr": 0.0001, "betas": (0.5, 0.999)},
    "Adversary": {"lr": 0.0001, "betas": (0.5, 0.999)},
}

gan = WassersteinGAN(
    generator, critic, z_dim=z_dim, x_dim=N, 
    optim=optimizer, optim_kwargs=optimizer_kwargs, device="cpu", folder=None
)

gan.fit(X_train, epochs=125, print_every="1e", enable_tensorboard=False)

samples, losses = gan.get_training_results()

plt.hist(samples.flatten(), bins=50)

fig, axs = plot_losses(losses)

# plot_generator_cov(generator)