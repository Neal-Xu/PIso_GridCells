import ml_collections
import numpy as np


# training config
def TrainConfig():
    config = ml_collections.ConfigDict()
    config.batch_size = 64  # batch size default 64
    config.epochs = 50000  # number of epochs  analyze cell - 50000, noise - 100000
    config.lr = 1e-3  # learning rate
    config.n_checkpoints = 500  # number of checkpoints to save

    return config


# dataset config
def DatasetConfig():
    config = ml_collections.ConfigDict()
    config.box_size = 2*np.pi  # box_size x box_size environment, default 2*np.pi

    # grid
    config.bins = 64  # number of bins in the grid,default 64

    return config


# model config
def ModelConfig():
    config = ml_collections.ConfigDict()

    config.isnorm = True  # whether to normalize the activity

    # hidden layers
    config.n_units_fc = [4, 64, 256, 32] # number of units in each layer

    # isometry loss
    # config.rho = 1.0  # dg / dr
    config.sigma = 1.2  # range of the local isometry, default 1.2

    # size of torus
    # config.s0 = 0.84  # size of the torus

    # hyperparameters
    config.selected_losses = ['isometry_loss', 'size_loss']
    config.loss_weights = {'isometry_loss': 1, 'size_loss': 2}  # weights of the losses

    return config
