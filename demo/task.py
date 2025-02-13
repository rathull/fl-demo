"""demo: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    def __init__(self):
        # Define your own architecture!
        pass

    def forward(self, x):
        # Define your own architecture!
        pass


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    # move model to GPU if available
    # Define loss function
    # Define optimizer
    net.train()
    running_loss = 0.0
    for epoch in range(epochs):
        for batch in trainloader:
            # Get images from batch
            # Get labels from batch
            # Zero gradient
            # Get loss
            # Backprop
            # Update model weights
            # Add to total loss
            pass

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    # Define loss function
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            # Get images from batch
            # Get labels from batch
            # Get outputs
            # Get loss
            # Add to running loss
            # Check how many outputs were correct
            # Add to count of correct outputs
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def save_model(net, file_path="global_model.pth"):
    """Save the model's state dictionary to the specified file path."""
    torch.save(net.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def load_model(net, file_path="global_model.pth"):
    """Load the model's state dictionary from the specified file path."""
    state_dict = torch.load(file_path, map_location=torch.device("cpu"))
    net.load_state_dict(state_dict)
    print(f"Model loaded from {file_path}")
