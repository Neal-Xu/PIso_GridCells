import torch
import numpy as np


class DatasetMakerRandom(object):
    def __init__(self, config):
        self.box_size = config.box_size
        self.config = config

    def generate_data(self, samples=None):
        if samples is None:
            x = np.linspace(-1, 1, self.config.bins) * self.box_size * 0.8
            y = np.linspace(-1, 1, self.config.bins) * self.box_size * 0.8
            xx, yy = np.meshgrid(x, y)
            r = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        else:
            r = np.random.rand(samples, 2) * self.box_size * 2 - self.box_size
        v = np.zeros(1)  # no velocity
        return torch.tensor(r.astype('float32')), torch.tensor(v.astype('float32'))




if __name__ == "__main__":
    import ml_collections
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    config = ml_collections.ConfigDict()
    config.box_size = 2 * np.pi  # box_size x box_size environment
    config.timesteps = 10  # number of samples per trial

    dm = DatasetMakerRandom(config)
    rt, vt = dm.generate_data(samples=10)
    for i in range(10):
        plt.plot(rt[i, ..., 0], rt[i, ..., 1])
    plt.axis(config.box_size * np.array([-1, 1, -1, 1]))
    plt.axis('equal')
    # plt.show()

    # save
    currentFolder = os.path.dirname(os.path.abspath(__file__))
    programFolder = os.path.dirname(currentFolder)
    figureFolder = os.path.join(programFolder, 'data', 'figures')
    if not os.path.exists(figureFolder):
        os.makedirs(figureFolder)
    plt.savefig(os.path.join(figureFolder, 'sample_trial.png'))
