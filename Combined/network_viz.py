from torchviz import make_dot
import torch
from combined_Autoencoder import Autoencoder

n_states = 32
obs_space = 24
net = Autoencoder((n_states, obs_space))

x = torch.randn(1, n_states * obs_space)
y = net(x)
print(net)
make_dot(y, params=dict(list(net.named_parameters()))).render("linear_autoencoder", format="png")

# import matplotlib.pyplot as plt

# # Define the number of neurons in each layer
# layer_sizes = [n_states * obs_space, 512, 256, 128, 64, 32]

# # Plot the network structure
# plt.figure(figsize=(10, 5))
# plt.title("Autoencoder Network Structure")
# plt.xlabel("Layer")
# plt.ylabel("Number of Neurons")
# plt.bar(range(len(layer_sizes)), layer_sizes, color='skyblue')
# plt.xticks(range(len(layer_sizes)), [f"Layer {i+1}" for i in range(len(layer_sizes))])
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()