import torch
import torch.nn as nn
import numpy as np

# Code inspired by: https://github.com/hellojinwoo/TorchCoder/blob/master/autoencoders/rae.py

class LSTMEncoder(nn.Module):
    def __init__(self, sequence_length, state_size, embedding_size):
        super().__init__()
        self.sequence_length = sequence_length
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.hidden_size = 2 * embedding_size
        self.LSTM = nn.LSTM(
            input_size=self.state_size,
            hidden_size=self.embedding_size,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x, (h, c) = self.LSTM(x)
        last_h = h[-1, :, :]
        last_x = x[:, -1, :]
        return last_x


class LSTMDecoder(nn.Module):
    def __init__(self, sequence_length, embedding_size, output_size):
        super().__init__()
        self.sequence_length = sequence_length
        self.ouput_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = 2 * embedding_size
        self.LSTM = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        x, (h, c) = self.LSTM(x)    
        x = x.reshape((-1, self.sequence_length, self.hidden_size))
        out_sequence = self.fc(x)
        next_state = h[:, -1, :]
        return out_sequence, next_state


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, state_size, embedding_size, lr, epochs, max_grad_norm):
        super().__init__()

        self.seq_len = seq_len
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.lr = lr
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm

        # self.encoder = LSTMEncoder(1, state_size, embedding_size)
        # self.decoder = LSTMDecoder(1, embedding_size, state_size)
        self.encoder = LSTMEncoder(self.seq_len, self.state_size, self.embedding_size)
        self.decoder = LSTMDecoder(self.seq_len, self.embedding_size, self.state_size)

        self.loss_list = []

    def forward(self, x):
        
        encoded_sequence = self.encoder(x)
        decoded_sequence, next_state_estimate = self.decoder(encoded_sequence)
        # print(f"Encoded sequence shape: {encoded_sequence.shape}")
        # print(f"Decoded sequence shape: {decoded_sequence.shape}")
        return decoded_sequence, next_state_estimate
    
    def preprocess_state_sequence(self, state_sequence):
        data_in_tensor = torch.tensor(np.array(state_sequence), dtype = torch.float)
        unsqueezed_data = data_in_tensor.unsqueeze(1)
        
        self.seq_len = unsqueezed_data.shape[0]
        print(f"seq_len: {self.seq_len}")
        self.state_size = unsqueezed_data.shape[2]
        print(f"state_size: {self.state_size}")
        return unsqueezed_data

    def train_LSTMAutoencoder(self, x):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #sequence_chunks, next_states = self.preprocess_state_sequence(x)
        prepared_data = self.preprocess_state_sequence(x).squeeze(0)
        #prepared_data.squeeze(2)
        self.train() #??
        # print(f"prepared_data: {prepared_data.shape}")
        # print(f"x: {x.shape}")

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            decoded_sequence, next_state_estimate = self(prepared_data) # Should i train for the next state?? Make preprocess that returns [sequence, next_state] from collective sequence
            loss = criterion(decoded_sequence, prepared_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            optimizer.step()
            #print(f"Epoch: {epoch}, Loss: {loss.item()}")
            if loss.item() < 0.001:
                break
        return loss.item()

    def get_novelty(self, state_sequence):
        criterion = nn.MSELoss()
        
        prepared_data = self.preprocess_state_sequence(state_sequence)
        # print(f"prepared_data: {prepared_data.shape}")
        # print(f"state_sequence: {state_sequence.shape}")
        self.eval()
        with torch.no_grad():
            decoded_sequence, next_state_estimate = self(prepared_data)
            
            # Making it estimate the next state rather than recreating the current.
            # This does not work as intended, as the current state is not the same as the next state in the sequence.
            # Would need to implement with cell-wise LSTM modules to work as intended.
            decoded_sequence = decoded_sequence.numpy() 
            prepared_data = prepared_data.numpy()   
            # decoded_sequence = decoded_sequence[1:].numpy()
            # prepared_data = prepared_data[:-1].numpy()

            # print(f"decoded_sequence: {decoded_sequence.shape}")
            # print(f"prepared_data: {prepared_data.shape}")
            loss_list = []
            for i in range(len(decoded_sequence)):
                #loss = criterion(decoded_sequence[i], prepared_data[i])
                #print(f"decoded_sequence[i]: {decoded_sequence[i]}")
                loss = np.mean(np.abs(decoded_sequence[i] - prepared_data[i]), axis=1)
                # loss_list.append(loss.item())
                loss_list.append(loss[0])
            #loss = criterion(decoded_sequence, prepared_data)
            # print(f"loss_list: {loss_list}")
            return np.array(loss_list), next_state_estimate


