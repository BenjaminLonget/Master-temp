from cv2 import exp
import torch
import torch.nn as nn
import torch.optim as optim
from dataPlotter import DataPlotter
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        input_dim = input_dim[0] * input_dim[1] # Flattens the input of n_states * state_size
        print(f"Input dim: {input_dim}")
        self.encoder = nn.Sequential(
            # nn.Linear(input_dim, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            #nn.LSTM
            nn.Linear(input_dim, 512),
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
            nn.Linear(512, input_dim)
            # nn.Linear(512, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        # Encode the input
        encoded = self.encoder(x)
        # Decode the encoded representation
        decoded = self.decoder(encoded)
        return decoded

def preprocess_states(states_list, n_states, batch_size):
    # Initialize the result list
    print(f"preprocessing {len(states_list)} states...")
    flattened_states = []
    batches = []

    for i in range(0, len(states_list), n_states):
        
        # Collect n_states states into a single array
        flattened_state = np.concatenate(states_list[i:i + n_states], axis=0)
        flattened_states.append(flattened_state)

    # flattened_states are of length: total states / n_states
    # each state within flattened is of length: n_states
    #print(flattened_states[0])
    for i in range(0, len(flattened_states), batch_size):
        batches.append(flattened_states[i:i + batch_size])
    
    print(f"{len(states_list)} where preprocessed into {len(batches)} batches.")
    #print(f"Batches last: {batches[-1]}")
    
    return batches

def train_autoencoder(autoencoder, state_list, n_states, batch_size, ae_number, autoencoder_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    autoencoder.to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.003)
    mse_loss = nn.MSELoss().cuda()
    print(f"Device: {device}")
    loss_plot = DataPlotter(autoencoder_dir + f"Autoencoder {ae_number} loss", "Mini-batch", "Loss")
    #novel_reward_plot = DataPlotter("Novelty reward (no weight)", "Mini-batch", "Novelty score")
    batch_list = preprocess_states(state_list, n_states, batch_size)
    #batch_list should now contain batch_size flattened states, each containing n_states

    #batch_list_tensor = torch.tensor(batch_list, dtype=torch.float, device=device)
    for mini_batch in batch_list:
        #print(f"minibatch length: {len(mini_batch)}, size: {np.size(mini_batch, 0)},{np.size(mini_batch, 1)}")
        #print(mini_batch)
        #print(np.shape(np.stack(mini_batch, axis=0)))
        mini_batch_tensor = torch.tensor(np.array(mini_batch), dtype=torch.float, device=device)
        #mini_batch_tensor = mini_batch_tensor.view(-1, np.shape(mini_batch)[0] * np.shape(mini_batch)[1])
        recreated_states = autoencoder(mini_batch_tensor)
        loss = mse_loss(recreated_states, mini_batch_tensor)   # ER DER NOGET HER????
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_plot.add_data(loss.item())
        #novel_reward_plot.add_data(-np.exp(-loss.item()))

    loss_plot.display_data()
    #novel_reward_plot.display_data()

    return autoencoder