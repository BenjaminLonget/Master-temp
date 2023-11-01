import torch
import torch.nn as nn
from dataPlotter import DataPlotter
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init()
        self.loss_plotter = DataPlotter("AutoEncoder Loss", "Updates", "Mean Squared Error")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
            )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(input_dim),
        )

    def forward(self, x):
        # Encode the input
        encoded = self.encoder(x)
        # Decode the encoded representation
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(autoencoder, state_list, n_states):
    
    autoencoder_loss_list = []
    mini_batch_list = [] # to store n_states
    for i in range(0, len(state_list), n_states):
        batch = state_list[i:i+n_states]  # Slice a batch of states
        mini_batch_list.append(np.stack(batch, axis=0))  # Stack the batch into a numpy array

