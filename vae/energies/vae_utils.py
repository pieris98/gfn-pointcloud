import torch
import torchvision
from torchvision import datasets
import math
import torch_geometric
import numpy as np

_EPS = 1.e-5

logtwopi = math.log(2 * math.pi)

_VAE_DATA_PATH = 'energies/data/'
_MNIST_EVALUATION_SUBSET = 'energies/data/mnist_evaluation_subset.npy'

_VAEP_DATA_PATH_POINTCLOUD = 'energies/data/ModelNet10'
def load_modelnet(num_points=2048, batch_size=32, num_classes=10):
    """
    Load the ModelNet dataset using PyTorch Geometric.
    
    Args:
        num_points (int): Number of points to sample from each mesh
        batch_size (int): Batch size for the data loader
        num_classes (int): Number of classes (10 or 40)
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    # Define the transformations
    transforms = Compose([
        SamplePoints(num_points, include_normals=True),
        NormalizeScale()
    ])

    # Load training dataset
    train_dataset = ModelNet(
        root='data/ModelNet{}'.format(num_classes),
        name=str(num_classes),
        train=True,
        transform=transforms
    )

    # Load test dataset
    test_dataset = ModelNet(
        root='data/ModelNet{}'.format(num_classes),
        name=str(num_classes),
        train=False,
        transform=transforms
    )

    class PointCloudDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            # Extract xyz coordinates and reshape to (3, num_points)
            points = data.pos.T  # Transpose from (num_points, 3) to (3, num_points)
            return points, data.y

    # Wrap the datasets with our custom dataset class
    train_dataset = PointCloudDataset(train_dataset)
    test_dataset = PointCloudDataset(test_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader

def gaussian_likelihood(x_hat, logscale, x):
    scale = torch.exp(logscale)
    mean = x_hat
    distribution = torch.distributions.Normal(mean, scale)

    log_likelihood = distribution.log_prob(x).sum(1)
    return log_likelihood


def log_normal_diag(x, mu, log_var):
    log_p = -0.5 * (logtwopi + log_var + torch.exp(-log_var) * (x - mu) ** 2.).sum(1)
    return log_p


def log_standard_normal(x):
    log_p = -0.5 * (logtwopi + x ** 2.).sum(1)
    return log_p


def log_bernoulli(x, p):
    pp = torch.clamp(p, _EPS, 1. - _EPS)
    log_p = (x * torch.log(pp) + (1. - x) * torch.log(1. - pp)).sum(1)
    return log_p


def estimate_distribution(model):
    distribution = torch.distributions.MultivariateNormal(torch.zeros(128), torch.eye(128))
    # z_samples = distribution.sample((100000,)).to(device)
    z_samples = distribution.sample((100000,))
    x_prediction_samples = model.decode(z_samples)
    vae_posterior_mu = torch.mean(x_prediction_samples, dim=0)
    vae_posterior_std = torch.std(x_prediction_samples, dim=0)
    vae_posterior_logvar = torch.log(vae_posterior_std ** 2)
    return vae_posterior_mu, vae_posterior_std, vae_posterior_logvar


def get_dataloaders(batch_size, device):

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == torch.device("cuda") else {}

    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(_VAE_DATA_PATH, train=True, download=True,
                       transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(_VAE_DATA_PATH, train=False, transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size, shuffle=False, **kwargs)

    evaluation_subset = torch.from_numpy(np.load(_MNIST_EVALUATION_SUBSET)).to(device)

    return train_dataloader, test_dataloader, evaluation_subset

import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose, SamplePoints, NormalizeScale
from torch.utils.data import DataLoader


def get_pointcloud_dataloaders(batch_size=32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),root_dir=_VAEP_DATA_PATH_POINTCLOUD, num_points=2048, num_classes=10):
    """
    Create DataLoaders for ModelNet10 point cloud data.

    Args:
        root_dir (str): Root directory of the ModelNet10 dataset.
        batch_size (int): Batch size for the DataLoader.
        device (torch.device): Device to assign the tensors.
        num_points (int): Number of points to sample from each point cloud.
        num_classes (int): Number of classes in the dataset (default 10).

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test set.
        evaluation_subset (torch.Tensor): Test data in tensor format for evaluation.
    """
    # Define the transformations
    transforms = Compose([
        SamplePoints(num_points, include_normals=True),
        NormalizeScale()
    ])

    # Load training dataset
    train_dataset = ModelNet(
        root=root_dir,
        name=str(num_classes),
        train=True,
        transform=transforms
    )

    # Load test dataset
    test_dataset = ModelNet(
        root=root_dir,
        name=str(num_classes),
        train=False,
        transform=transforms
    )

    class PointCloudDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            # Extract xyz coordinates and reshape to (3, num_points)
            points = data.pos.T  # Transpose from (num_points, 3) to (3, num_points)
            label = data.y
            return points, label

    # Wrap the datasets with the custom dataset class
    train_dataset = PointCloudDataset(train_dataset)
    test_dataset = PointCloudDataset(test_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create evaluation_subset from test_loader
    evaluation_subset = []
    for data, _ in test_loader:
        evaluation_subset.append(data)
    evaluation_subset = torch.cat(evaluation_subset, dim=0).to(device)

    return train_loader, test_loader, evaluation_subset