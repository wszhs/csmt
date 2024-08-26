"""
ConditionalWassersteinGANGP
---------------------------
Implements the conditional variant of the Wasserstein GAN Gradient Penalized[1].

Uses the Wasserstein loss to determine the realness of real and fake images.
The Wasserstein loss has several theoretical advantages over the Jensen-Shanon divergence
minimised by the original GAN. In this architecture the critic (discriminator) is often
trained multiple times for every generator step.
Lipschitz continuity is "enforced" by gradient penalization.

Losses:
    - Generator: Wasserstein
    - Critic: Wasserstein + Gradient penalization
Default optimizer:
    - torch.optim.RMSprop
Custom parameter:
    - lambda_grad: Weight for the reconstruction loss of the gradients. Pushes the norm of the gradients to 1.

References
----------
.. [1] https://arxiv.org/abs/1704.00028
"""

import torch

import numpy as np

from csmt.zoopt.vegans.models.unconditional.WassersteinGANGP import WassersteinGANGP
from csmt.zoopt.vegans.models.conditional.AbstractConditionalGAN1v1 import AbstractConditionalGAN1v1


class ConditionalWassersteinGANGP(AbstractConditionalGAN1v1, WassersteinGANGP):
    """
    Parameters
    ----------
    generator: nn.Module
        Generator architecture. Produces output in the real space.
    adversary: nn.Module
        Adversary architecture. Produces predictions for real and fake samples to differentiate them.
    x_dim : list, tuple
        Number of the output dimensions of the generator and input dimension of the discriminator / critic.
        In the case of images this will be [nr_channels, nr_height_pixels, nr_width_pixels].
    z_dim : int, list, tuple
        Number of the latent dimensions for the generator input. Might have dimensions of an image.
    y_dim : int, list, tuple
        Number of dimensions for the target label. Might have dimensions of image for image to image translation, i.e.
        [nr_channels, nr_height_pixels, nr_width_pixels] or an integer representing a number of classes.
    optim : dict or torch.optim
        Optimizer used for each network. Could be either an optimizer from torch.optim or a dictionary with network
        name keys and torch.optim as value, i.e. {"Generator": torch.optim.Adam}.
    optim_kwargs : dict
        Optimizer keyword arguments used for each network. Must be a dictionary with network
        name keys and dictionary with keyword arguments as value, i.e. {"Generator": {"lr": 0.0001}}.
    lambda_grad: float
        Weight for the reconstruction loss of the gradients. Pushes the norm of the gradients to 1.
    feature_layer : torch.nn.*
        Output layer used to compute the feature loss. Should be from either the discriminator or critic.
        If `feature_layer` is not None, the original generator loss is replaced by a feature loss, introduced
        [here](https://arxiv.org/abs/1606.03498v1).
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
            generator,
            adversary,
            x_dim,
            z_dim,
            y_dim,
            optim=None,
            optim_kwargs=None,
            feature_layer=None,
            fixed_noise_size=32,
            lmbda_grad=10,
            device=None,
            ngpu=None,
            folder="./veganModels/cWassersteinGANGP",
            secure=True):

        super().__init__(
            generator=generator, adversary=adversary,
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Critic",
            optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size,
            device=device,
            folder=folder,
            ngpu=ngpu, secure=secure
        )
        self.lmbda_grad = lmbda_grad
        self.hyperparameters["lmbda_grad"] = lmbda_grad

    #########################################################################
    # Actions during training
    #########################################################################
    def _calculate_adversary_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch).detach()
        fake_concat = self.concatenate(fake_images, y_batch)
        real_concat = self.concatenate(X_batch, y_batch)
        return WassersteinGANGP._calculate_adversary_loss(self, X_batch=real_concat, Z_batch=None, fake_images=fake_concat)