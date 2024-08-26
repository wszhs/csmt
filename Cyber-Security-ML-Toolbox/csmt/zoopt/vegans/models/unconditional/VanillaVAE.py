"""
VanillaVAE
----------
Implements the Variational Autoencoder[1].

Trains on Kullback-Leibler loss and mean squared error reconstruction loss.

Losses:
    - Encoder: Kullback-Leibler
    - Decoder: L2 (Mean Squared Error)
Default optimizer:
    - torch.optim.Adam
Custom parameter:
    - lambda_KL: Weight for the encoder loss computing the Kullback-Leibler divergence in the latent space.

References
----------
.. [1] https://arxiv.org/pdf/1906.02691.pdf
"""

import torch

import numpy as np
import torch.nn as nn

from torch.nn import MSELoss
from csmt.zoopt.vegans.utils.layers import LayerReshape
from csmt.zoopt.vegans.utils.networks import Encoder, Decoder, Autoencoder
from csmt.zoopt.vegans.models.unconditional.AbstractGenerativeModel import AbstractGenerativeModel

class VanillaVAE(AbstractGenerativeModel):
    """
    Parameters
    ----------
    encoder: nn.Module
        Encoder architecture. Produces predictions in the latent space.
    decoder: nn.Module
        Decoder architecture. Produces output in the real space.
    x_dim : list, tuple
        Number of the output dimensions of the generator and input dimension of the discriminator / critic.
        In the case of images this will be [nr_channels, nr_height_pixels, nr_width_pixels].
    z_dim : int, list, tuple
        Number of the latent dimensions for the generator input. Might have dimensions of an image.
    optim : dict or torch.optim
        Optimizer used for each network. Could be either an optimizer from torch.optim or a dictionary with network
        name keys and torch.optim as value, i.e. {"Generator": torch.optim.Adam}.
    optim_kwargs : dict
        Optimizer keyword arguments used for each network. Must be a dictionary with network
        name keys and dictionary with keyword arguments as value, i.e. {"Generator": {"lr": 0.0001}}.
    lambda_KL: float
        Weight for the encoder loss computing the Kullback-Leibler divergence in the latent space.
    fixed_noise_size : int
        Number of images shown when logging. The fixed noise is used to produce the images in the folder/images
        subdirectory, the tensorboard images tab and the samples in get_training_results().
    device : string
        Device used while training the model. Either "cpu" or "cuda".
    ngpu : int
        Number of gpus used during training if device == "cuda".
    folder : string
        Creates a folder in the current working directory with this name. All relevant files like summary, images, models and
        tensorboard output are written there. Existing folders are never overwritten or deleted. If a folder with the same name
        already exists a time stamp is appended to make it unique.
    """

    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            encoder,
            decoder,
            x_dim,
            z_dim,
            optim=None,
            optim_kwargs=None,
            lambda_KL=10,
            fixed_noise_size=32,
            device=None,
            ngpu=0,
            folder="./veganModels/VanillaVAE",
            secure=True):

        self.decoder = Decoder(decoder, input_size=z_dim, device=device, ngpu=ngpu, secure=secure)
        self.encoder = Encoder(encoder, input_size=x_dim, device=device, ngpu=ngpu, secure=secure)
        self.autoencoder = Autoencoder(self.encoder, self.decoder)
        self.neural_nets = {
            "Autoencoder": self.autoencoder
        }


        super().__init__(
            x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs, feature_layer=None,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu, secure=secure
        )
        self.mu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.encoder.output_size), np.prod(z_dim)),
            LayerReshape(shape=z_dim)
        ).to(self.device)
        self.log_variance = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.encoder.output_size), np.prod(z_dim)),
            LayerReshape(shape=z_dim)
        ).to(self.device)

        self.lambda_KL = lambda_KL
        self.hyperparameters["lambda_KL"] = lambda_KL

        if self.secure:
            # if self.encoder.output_size == self.z_dim:
            #     raise ValueError(
            #         "Encoder output size is equal to z_dim, but for VAE algorithms the encoder last layers for mu and sigma " +
            #         "are constructed by the algorithm itself.\nSpecify up to the second last layer for this particular encoder."
            #     )
            assert (self.decoder.output_size == self.x_dim), (
                "Decoder output shape must be equal to x_dim. {} vs. {}.".format(self.decoder.output_size, self.x_dim)
            )

    def _define_loss(self):
        loss_functions = {"Autoencoder": MSELoss()}
        return loss_functions


    #########################################################################
    # Actions during training
    #########################################################################
    def encode(self, x):
        return self.encoder(x)

    def calculate_losses(self, X_batch, Z_batch, who=None):
        losses = self._calculate_autoencoder_loss(X_batch=X_batch, Z_batch=Z_batch)
        return losses

    def _calculate_autoencoder_loss(self, X_batch, Z_batch, fake_images=None):
        encoded_output = self.encode(X_batch)
        mu = self.mu(encoded_output)
        log_variance = self.log_variance(encoded_output)

        if fake_images is None:
            Z_batch_encoded = mu + torch.exp(log_variance)*Z_batch
            fake_images = self.generate(Z_batch_encoded)

        kl_loss = 0.5*(log_variance.exp() + mu**2 - log_variance - 1).sum()
        reconstruction_loss = self.loss_functions["Autoencoder"](
            fake_images, X_batch
        )

        total_loss = reconstruction_loss + self.lambda_KL*kl_loss
        return {
            "Autoencoder": total_loss,
            "Kullback-Leibler": self.lambda_KL*kl_loss,
            "Reconstruction": reconstruction_loss,
        }
