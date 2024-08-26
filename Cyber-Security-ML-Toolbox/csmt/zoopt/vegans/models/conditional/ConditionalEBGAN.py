"""
ConditionalEBGAN
----------------
Implements conditional variant of the Energy based GAN[1].

Uses an auto-encoder as the adversary structure.

Losses:
    - Generator: L2 (Mean Squared Error)
    - Autoencoder: L2 (Mean Squared Error)
Default optimizer:
    - torch.optim.Adam
Custom parameter:
    - m: Cut off for the hinge loss. Look at reference for more information.

References
----------
.. [1] https://arxiv.org/pdf/1609.03126.pdf
"""

import torch

from csmt.zoopt.vegans.models.unconditional.EBGAN import EBGAN
from csmt.zoopt.vegans.models.conditional.AbstractConditionalGAN1v1 import AbstractConditionalGAN1v1

class ConditionalEBGAN(AbstractConditionalGAN1v1, EBGAN):
    """
    Parameters
    ----------
    generator: nn.Module
        Generator architecture. Produces output in the real space.
    adversary: nn.Module
        Adversary architecture. Produces predictions for real and fake samples to differentiate them.
    encoder: nn.Module
        Encoder architecture. Produces predictions in the latent space.
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
    m: float
        Cut off for the hinge loss. Look at reference for more information.
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
            m=None,
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=None,
            folder="./veganModels/cEBGAN",
            secure=True):

        super().__init__(
            generator=generator, adversary=adversary,
            z_dim=z_dim, x_dim=x_dim, y_dim=y_dim, adv_type="Autoencoder",
            optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu, secure=secure
        )
        self.m = m
        self.hyperparameters["m"] = m

        if self.secure:
            assert self.adversary.output_size == x_dim, (
                "AutoEncoder structure used for adversary. Output dimensions must equal x_dim. " +
                "Output: {}. x_dim: {}.".format(self.adversary.output_size, x_dim)
            )

    def _calculate_generator_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch)
        if self.feature_layer is None:
            fake_predictions = self.predict(x=fake_images, y=y_batch)
            gen_loss = self.loss_functions["Generator"](
                fake_images, fake_predictions
            )
        else:
            fake_concat = self.concatenate(fake_images, y_batch)
            real_concat = self.concatenate(X_batch, y_batch)
            gen_loss = self._calculate_feature_loss(X_real=real_concat, X_fake=fake_concat)
        return {"Generator": gen_loss}

    def _calculate_adversary_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch).detach()
        fake_predictions = self.predict(x=fake_images, y=y_batch)
        real_predictions = self.predict(x=X_batch, y=y_batch)

        adv_loss_fake = self.loss_functions["Adversary"](
            fake_predictions, fake_images
        )
        if adv_loss_fake < self.m:
            adv_loss_fake = self.m - adv_loss_fake
        else:
            adv_loss_fake = torch.Tensor([0]).to(self.device)
        adv_loss_real = self.loss_functions["Adversary"](
            real_predictions, X_batch
        )
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real).float()
        return {
            "Adversary": adv_loss,
            "Adversary_fake": adv_loss_fake,
            "Adversary_real": adv_loss_real,
            "RealFakeRatio": adv_loss_real / adv_loss_fake
        }
