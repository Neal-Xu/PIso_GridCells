# Adapted from Pettersen et al. (2024) [https://github.com/bioAI-Oslo/GridsWithoutPI].
# Modified to constrains population activity to toroidal manifolds and include torus-size regularization.
# We also injected noise to the network.

import torch
import numpy as np
import pickle
import os
import math
import torch.nn.functional as F


class baseGCNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.current_s = None
        self.config = config
        self.build_net()
        self.init_state()

    def build_net(self):
        pass

    def norm_relu(self, x):
        rx = self.nonlinear(x)
        if self.config.isnorm:
            norm = (torch.linalg.norm(rx, dim=-1)[..., None])
            return rx / torch.maximum(norm, 1e-13 * torch.ones(norm.shape, device=x.device))  #  norm*2
        else:
            return rx

    def distance_loss(self, g, r):
        g = torch.reshape(g, (-1, g.shape[-1]))
        r = torch.reshape(r, (-1, r.shape[-1]))
        dg = torch.nn.functional.pdist(g)  # state distance
        dr = torch.nn.functional.pdist(r)  # spatial distance

        # conformal isometry
        diff = (dg - self.config.rho * dr) ** 2  # dr*
        # loss envelope function within a local neighborhood
        envelope = torch.exp(-dr ** 2 / (2 * self.config.sigma ** 2))

        # sm = self.cossml_loss(g, r)
        # ore = self.ore_loss(g, r)

        return torch.mean(diff * envelope)  #  + 0.001*sm + 5*ore


    def size_loss(self, g, r):
        g = torch.reshape(g, (-1, g.shape[-1]))
        s = 1 - torch.sum(torch.mean(g, dim=0) ** 2)
        s_0 = self.config.s0
        self.current_s = s
        return (s-s_0) ** 2


    def cossml_loss(self, g, r):
        g = torch.reshape(g, (-1, g.shape[-1]))
        a = torch.mean(g, dim=0)
        d = a.size(0)  # length of the vector
        diagonal_vector = torch.ones_like(a)  # the vector (1, 1, ..., 1)

        # Normalize a
        a_normalized = a / torch.norm(a)

        # Normalize the diagonal vector
        diagonal_vector_normalized = diagonal_vector / torch.sqrt(torch.tensor(d, dtype=torch.float))

        # Compute the cosine similarity (without normalization of a)
        cosine_similarity = torch.sum(a_normalized * diagonal_vector_normalized)

        # The loss is 1 - cosine similarity
        loss = 1 - cosine_similarity
        return loss

    def ore_loss(self, g, r):
        # g: (..., d) -> (N, d)
        g = torch.reshape(g, (-1, g.shape[-1]))
        a = torch.mean(g, dim=0)
        b = g - a

        d = a.numel()
        v = torch.ones_like(a)
        v = v / torch.sqrt(torch.tensor(d, dtype=a.dtype, device=a.device))


        proj = b @ v
        b_norm = torch.norm(b, dim=-1) + 1e-8
        cos = proj / b_norm

        loss = torch.mean(cos ** 2)
        return loss

    def init_state(self):
        self.loss_history = {}
        for loss_name in self.config.selected_losses:
            self.loss_history[loss_name] = []
        self.loss_history["loss"] = []
        self.loss_history["size"] = []

    def update_state(self, loss):
        for loss_name in self.loss_history.keys():
            self.loss_history[loss_name].append(loss[loss_name].item())

    def forward(self, rs, vs):
        pass

    def train_step(self, labels, vels, optimizer):
        # zero grad
        optimizer.zero_grad()

        # forward
        gs, _ = self(labels, vels)

        # loss
        total_loss = 0
        loss = {}
        for loss_name in self.config.selected_losses:
            if loss_name == "isometry_loss":
                loss_value = self.distance_loss(gs, labels)
            elif loss_name == "size_loss":
                loss_value = self.size_loss(gs, labels)
                loss['size'] = self.current_s
            else:
                raise ValueError

            total_loss += loss_value * self.config.loss_weights[loss_name]
            loss[loss_name] = loss_value
        loss["loss"] = total_loss
        self.update_state(loss)

        # backward
        total_loss.backward()
        optimizer.step()

        return self.loss_history


class FFGC(baseGCNet):
    def __init__(self, config):
        super().__init__(config=config)

    def build_net(self):
        self.build_encoder()

    def build_encoder(self):
        self.nonlinear = torch.nn.ReLU()
        # self.noise = GaussianNoise(mean=0, std=0.1).to(self.config.device) # std=0.1
        std = float(os.environ.get("STD", str(getattr(self.config, "std", 0))))
        self.noise = GaussianNoise(mean=0, std=std).to(self.config.device)

        n_units = self.config.n_units_fc
        layers = []
        layers.append(ToriActivation(self.config))
        for i in range(len(n_units) - 1):
            layers.append(torch.nn.Linear(n_units[i], n_units[i + 1]))
            if i < len(n_units) - 2:
                layers.append(self.nonlinear)
        self.rg = torch.nn.Sequential(*layers)

        # to device
        self.rg.to(self.config.device)

    def forward(self, rs, vs=None):
        g = self.rg(rs)
        g = self.noise(g)
        g = self.norm_relu(g)
        return g, None


class ToriActivation(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cycle_encoder = torch.nn.Linear(2, int(self.config.n_units_fc[0]/2), bias=False)
        self.ks = torch.nn.Parameter(torch.ones(int(self.config.n_units_fc[0] / 2), dtype=torch.float32))

    def forward(self, rs):
        re = self.cycle_encoder(rs)
        components = []
        for i in range(re.shape[1]):
            components.append(torch.cos(re[:, i]))
            components.append(torch.sin(re[:, i]))
        resize_components = []
        for i, k in enumerate(self.ks):
            resize_components.append(components[2 * i] * k)
            resize_components.append(components[2 * i + 1] * k)
        return torch.stack(resize_components, dim=1)


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise


def get_flattened_data(tensor, device):
    """Helper function to detach, move to CPU if needed, and flatten the tensor."""
    if device == 'cpu':
        return tensor.detach().numpy().flatten()
    else:
        return tensor.cpu().detach().numpy().flatten()


def process_model_data(model, device):
    """Process model data based on the number of modules and append to results."""
    ks, rs = [], []
    ks.append(get_flattened_data(model.rg[0].cycle_encoder.weight, device))
    rs.append(get_flattened_data(model.rg[0].ks.data, device))
    return ks, rs
