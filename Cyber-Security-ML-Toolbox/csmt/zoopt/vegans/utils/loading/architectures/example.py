import numpy as np
import torch.nn as nn

from csmt.zoopt.vegans.utils import get_input_dim
from csmt.zoopt.vegans.utils.layers import LayerReshape

class MyGenerator(nn.Module):
    def __init__(self, gen_in_dim, x_dim):
        super().__init__()
        self.hidden_part = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(gen_in_dim), 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, int(np.prod(x_dim))),
            LayerReshape(x_dim)
        )
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden_part(x)
        return self.output(x)

def load_example_generator(x_dim, z_dim, y_dim=None):
    """ Load some example architecture for the generator.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : integer, list
        Indicating the number of dimensions for the latent space.
    y_dim : integer, list, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for generator,.
    """
    if y_dim is not None:
        gen_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
    else:
        gen_in_dim = z_dim

    return MyGenerator(gen_in_dim=gen_in_dim, x_dim=x_dim)


class MyAdversary(nn.Module):
    def __init__(self, adv_in_dim, first_layer, last_layer):
        super().__init__()
        self.hidden_part = nn.Sequential(
            first_layer,
            nn.Flatten(),
            nn.Linear(np.prod(adv_in_dim), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
        )
        self.feature_part = nn.Linear(256, 1)
        self.output = last_layer()

    def forward(self, x):
        x = self.hidden_part(x)
        x = self.feature_part(x)
        return self.output(x)

def load_example_adversary(x_dim, y_dim=None, adv_type="Critic"):
    """ Load some example architecture for the adversary.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    y_dim : integer, list, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for adversary.
    """
    possible_types = ["Discriminator", "Critic"]
    if adv_type == "Critic":
        last_layer = nn.Identity
    elif adv_type == "Discriminator":
        last_layer = nn.Sigmoid
    else:
        raise ValueError("'adv_type' must be one of: {}.".format(possible_types))

    x_dim = [x_dim] if isinstance(x_dim, int) else x_dim
    if y_dim is not None:
        adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
    else:
        adv_in_dim = x_dim

    if len(adv_in_dim) == 3 and np.prod(adv_in_dim)>1024:
        first_layer = nn.Conv2d(in_channels=adv_in_dim[0], out_channels=3, kernel_size=5, stride=2)
        out_pixels_x = int((adv_in_dim[1] - (5 - 1) - 1) / 2 + 1)
        out_pixels_y = int((adv_in_dim[2] - (5 - 1) - 1) / 2 + 1)
        adv_in_dim = (3, out_pixels_x, out_pixels_y)
    else:
        first_layer = nn.Identity()

    return MyAdversary(adv_in_dim=adv_in_dim, first_layer=first_layer, last_layer=last_layer)


class MyEncoder(nn.Module):
    def __init__(self, enc_in_dim, z_dim, first_layer):
        super().__init__()
        self.hidden_part = nn.Sequential(
            first_layer,
            nn.Flatten(),
            nn.Linear(np.prod(enc_in_dim), 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, np.prod(z_dim)),
            LayerReshape(z_dim)
        )
        self.output = nn.Identity()

    def forward(self, x):
        x = self.hidden_part(x)
        return self.output(x)

def load_example_encoder(x_dim, z_dim, y_dim=None):
    """ Load some example architecture for the encoder.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : integer, list
        Indicating the number of dimensions for the latent space.
    y_dim : None, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for encoder.
    """
    z_dim = [z_dim] if isinstance(z_dim, int) else z_dim

    if y_dim is not None:
        enc_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
    else:
        enc_in_dim = x_dim

    if len(enc_in_dim) == 3 and np.prod(enc_in_dim)>1024:
        first_layer = nn.Conv2d(in_channels=enc_in_dim[0], out_channels=3, kernel_size=5, stride=2)
        out_pixels_x = int((enc_in_dim[1] - (5 - 1) - 1) / 2 + 1)
        out_pixels_y = int((enc_in_dim[2] - (5 - 1) - 1) / 2 + 1)
        enc_in_dim = (3, out_pixels_x, out_pixels_y)
    else:
        first_layer = nn.Identity()

    return MyEncoder(enc_in_dim=enc_in_dim, z_dim=z_dim, first_layer=first_layer)



class MyDecoder(nn.Module):
    def __init__(self, x_dim, dec_in_dim):
        super().__init__()
        self.hidden_part = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(dec_in_dim), 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, np.prod(x_dim)),
            LayerReshape(x_dim)
        )
        self.output = nn.Identity()

    def forward(self, x):
        x = self.hidden_part(x)
        return self.output(x)

def load_example_decoder(x_dim, z_dim, y_dim=None):
    """ Load some example architecture for the decoder.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    z_dim : integer, list
        Indicating the number of dimensions for the latent space.
    y_dim : None, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for decoder.
    """
    x_dim = [x_dim] if isinstance(x_dim, int) else x_dim

    if y_dim is not None:
        dec_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
    else:
        dec_in_dim = z_dim

    return MyDecoder(x_dim=x_dim, dec_in_dim=dec_in_dim)


class MyAutoEncoder(nn.Module):
    def __init__(self, adv_in_dim, x_dim, first_layer):
        super().__init__()
        self.hidden_part = nn.Sequential(
            first_layer,
            nn.Flatten(),
            nn.Linear(np.prod(adv_in_dim), 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, np.prod(x_dim)),
            LayerReshape(x_dim)
        )
        self.output = nn.Identity()

    def forward(self, x):
        x = self.hidden_part(x)
        return self.output(x)

def load_example_autoencoder(x_dim, y_dim=None):
    """ Load some example architecture for the auto-encoder.

    Parameters
    ----------
    x_dim : integer, list
        Indicating the number of dimensions for the real data.
    y_dim : integer, list, optional
        Indicating the number of dimensions for the labels.

    Returns
    -------
    torch.nn.Module
        Architectures for autoencoder.
    """
    if y_dim is not None:
        adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
    else:
        adv_in_dim = x_dim

    if len(adv_in_dim) == 3 and np.prod(adv_in_dim)>1024:
        first_layer = nn.Conv2d(in_channels=adv_in_dim[0], out_channels=3, kernel_size=5, stride=2)
        out_pixels_x = int((adv_in_dim[1] - (5 - 1) - 1) / 2 + 1)
        out_pixels_y = int((adv_in_dim[2] - (5 - 1) - 1) / 2 + 1)
        adv_in_dim = (3, out_pixels_x, out_pixels_y)
    else:
        first_layer = nn.Identity()

    return MyAutoEncoder(adv_in_dim=adv_in_dim, x_dim=x_dim, first_layer=first_layer)