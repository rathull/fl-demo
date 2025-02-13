"""demo: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from demo.task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        # ...
        self.net.to(self.device)

    def fit(self, parameters, config):
        # Set weights of local network to incoming parameters sent from server
        train_loss = # train...
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        # Set weights of local network to incoming parameters sent from server
        loss, accuracy = # test...
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    # Create a local network
    # Get the partition_id from this node config
    # Get the number of partitions from this node config
    # Load dataset assigned to this partition
    # Get the number of epochs to train on locally from this node config

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
