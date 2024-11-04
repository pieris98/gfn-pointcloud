import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

import os
import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose, SamplePoints, NormalizeScale
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
try:
    from .vae_utils import load_modelnet, load_modelnet_chair
except:
    from vae_utils import load_modelnet, load_modelnet_chair

class VAEPointCloudOLD(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAEPointCloudOLD, self).__init__()
        self.encoder = PointCloudEncoder(latent_dim)
        self.decoder = PointCloudDecoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
    
    def encode(self,x):
        x = x.view(-1,3,2048)
        mu, logvar = self.encoder(x)
        return mu,logvar
    
    def decode(self, z):
        z = self.decoder(z)              # (batch_size, 3, 2048)
        # z = z.view(-1, 3 * 2048)        # (batch_size, 3 * 2048=6144)
        return z
    
    def generate(self, num_samples=1):
        """
        Generate new point clouds from random latent vectors
        """
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    
    
    def loss_function(self,recon_x, x, mu, logvar):
        """
        Compute the VAE loss function with Chamfer Distance.
        
        Args:
            recon_x: Reconstructed point cloud (batch_size, 3, 2048)
            x: Original point cloud (batch_size, 3, 2048)
            mu: Mean from the encoder's latent space
            logvar: Log variance from the encoder's latent space
        
        Returns:
            Total loss as a scalar tensor
        """
        # Transpose to (batch_size, num_points, coordinates)
        recon_x = recon_x.permute(0, 2, 1)  # (batch_size, 2048, 3)
        x = x.permute(0, 2, 1)              # (batch_size, 2048, 3)
        
        # Compute Chamfer Distance
        chamfer = self.chamfer_distance(recon_x, x)
        
        # Compute KL Divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        return chamfer + KLD
    
    def chamfer_distance(self,pc1, pc2):
        """
        Compute Chamfer Distance between two point clouds.
        
        Args:
            pc1: Tensor of shape (batch_size, num_points, 3)
            pc2: Tensor of shape (batch_size, num_points, 3)
        
        Returns:
            chamfer_dist: Scalar tensor representing the Chamfer Distance
        """
        # Expand dimensions to compute pairwise distances
        pc1_expand = pc1.unsqueeze(2)  # (batch_size, num_points, 1, 3)
        pc2_expand = pc2.unsqueeze(1)  # (batch_size, 1, num_points, 3)
        
        # Compute pairwise distances
        dist = torch.norm(pc1_expand - pc2_expand, dim=3)  # (batch_size, num_points, num_points)
        
        # For each point in pc1, find the nearest point in pc2
        min_dist_pc1, _ = dist.min(dim=2)  # (batch_size, num_points)
        
        # For each point in pc2, find the nearest point in pc1
        min_dist_pc2, _ = dist.min(dim=1)  # (batch_size, num_points)
        
        # Average the minimal distances
        chamfer_dist = (min_dist_pc1.mean(dim=1) + min_dist_pc2.mean(dim=1)).mean()
        
        return chamfer_dist

class VAEPointCloud(nn.Module):
    def __init__(self, num_points=2048, latent_dim=128, hidden_dim=512):
        super(VAEPointCloud, self).__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # Point-wise features
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Global features
            nn.Conv1d(256, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1024),
            nn.ReLU()
        )
        
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 3, 1)
        )
        
    def encode(self, x):
        # x shape: (batch_size, 3, num_points)
        batch_size = x.size(0)
        
        # Extract features
        features = self.encoder(x)
        
        # Global max pooling
        features = torch.max(features, dim=2)[0]
        
        # Get latent parameters
        mu = self.fc_mu(features)
        log_var = self.fc_var(features)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        batch_size = z.size(0)
        
        # Generate initial point features
        features = self.decoder_fc(z)
        features = features.view(batch_size, 1024, 1)
        features = features.repeat(1, 1, self.num_points)
        
        # Generate point cloud
        points = self.decoder_conv(features)
        return points
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var
    
    def loss_function(self, reconstruction, x, mu, log_var, beta=1.0):
        """
        Compute the VAE loss with Chamfer Distance and KL divergence
        """
        batch_size = x.size(0)
        
        # Chamfer Distance
        x_expanded = x.unsqueeze(2)  # (B, 3, 1, N)
        reconstruction_expanded = reconstruction.unsqueeze(3)  # (B, 3, N, 1)
        
        # Compute pairwise distances
        dist = torch.sum((x_expanded - reconstruction_expanded) ** 2, dim=1)  # (B, N, N)
        
        # Compute bidirectional Chamfer distance
        chamfer_dist = torch.mean(torch.min(dist, dim=1)[0]) + torch.mean(torch.min(dist, dim=2)[0])
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_div = kl_div / batch_size
        
        # Total loss
        loss = chamfer_dist + beta * kl_div
        
        return loss
    
    def generate(self, num_samples=1):
        """
        Generate new point clouds from random latent vectors
        """
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    
class PointCloudEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(PointCloudEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)
        self.fc1 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))   # (batch_size, 64, 2048)
        x = self.relu(self.conv2(x))   # (batch_size, 128, 2048)
        x = self.relu(self.conv3(x))   # (batch_size, 256, 2048)
        x = self.relu(self.conv4(x))   # (batch_size, 512, 2048)
        x = torch.max(x, 2)[0]          # (batch_size, 512)
        x = self.relu(self.fc1(x))     # (batch_size, 256)
        mu = self.fc_mu(x)             # (batch_size, latent_dim)
        logvar = self.fc_logvar(x)     # (batch_size, latent_dim)
        return mu, logvar

class PointCloudDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(PointCloudDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 3 * 2048)
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.relu(self.fc1(z))     # (batch_size, 256)
        x = self.relu(self.fc2(x))     # (batch_size, 512)
        x = self.relu(self.fc3(x))     # (batch_size, 1024)
        x = self.fc4(x)                # (batch_size, 3*2048)
        x = x.view(-1, 3, 2048)        # (batch_size, 3, 2048)
        return x

class PointCloudVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(PointCloudVAE, self).__init__()
        self.encoder = PointCloudEncoder(latent_dim)
        self.decoder = PointCloudDecoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def generate(self, num_samples=1):
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    
    def loss_function(self,recon_x, x, mu, logvar):
        chamfer_dist = self.decoder.chamfer_distance(recon_x, x)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return chamfer_dist + kl_div
    







class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # point-wise mlp
        # Implement point-wise mlp model based on PointNet Architecture.
        '''I implement conv1d because shared MLP shares weights between neurons. We could use Linear layer with permuted inputs.
        Link explaining equivalence: https://stackoverflow.com/questions/55576314/conv1d-with-kernel-size-1-vs-linear-layer'''
        self.pointmlp1_1 = nn.Sequential(nn.Conv1d(3,64,1),nn.BatchNorm1d(64))
        self.pointmlp1_2 = nn.Sequential(nn.Conv1d(64,64,1),nn.BatchNorm1d(64))

        self.pointmlp2_1 = nn.Sequential(nn.Conv1d(64,64,1),nn.BatchNorm1d(64))
        self.pointmlp2_2 = nn.Sequential(nn.Conv1d(64,128,1),nn.BatchNorm1d(128))
        self.pointmlp2_3 = nn.Sequential(nn.Conv1d(128,1024,1),nn.BatchNorm1d(1024))
        
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        
        (Intermediate):
        - Global feature: [B,1024])
        
        Output:
            - mu: [B,latent_dim]
            - logvar: [B,latent_dim]
        """
        # TODO : Implement forward function.
        # print('pointcloud shape',pointcloud.shape)
        
        '''pass through 1st MLP'''
        pointcloud = F.relu(self.pointmlp1_1(pointcloud))
        pointcloud = F.relu(self.pointmlp1_2(pointcloud))
        # print('after mlp1 shape',pointcloud.shape)
        
        '''pass through 2nd MLP'''
        pointcloud = F.relu(self.pointmlp2_1(pointcloud))
        pointcloud = F.relu(self.pointmlp2_2(pointcloud))
        pointcloud = F.relu(self.pointmlp2_3(pointcloud))

        ''' max pooling across features (columns) to get global feature vector'''
        glob_features = torch.max(pointcloud, 2)[0]
        # print('global feature shape:',pointcloud.shape)
        mu, logvar = self.fc_mu(glob_features), self.fc_logvar(glob_features)
        return mu, logvar

class PointNetDecoder(nn.Module):
    def __init__(self, num_points=2048,latent_dim=256):
        super(PointNetDecoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(latent_dim, int(num_points/4)),nn.BatchNorm1d(int(num_points/4)))
        self.fc2 = nn.Sequential(nn.Linear(int(num_points/4), int(num_points/2)),nn.BatchNorm1d(int(num_points/2)))
        self.fc3 = nn.Sequential(nn.Linear(int(num_points/2), num_points),nn.Dropout(),nn.BatchNorm1d(num_points))
        self.fc4 = nn.Linear(num_points, num_points*3)
        self.num_points = num_points
    
    def forward(self, z):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # mplement decoder forward function.

        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = self.fc4(z)
        z = z.reshape(-1, 3,self.num_points)
        return z

class PointNetVAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(PointNetVAE, self).__init__()
        self.pointnetfeat = PointNetFeat(latent_dim)
        self.decoder = PointNetDecoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        mu, logvar = self.pointnetfeat(x)
        z = self.reparameterize(mu, logvar)
        # print(f"z shape: {z.shape}")  # Expected: [B, 128]
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def loss_function(self, reconstruction, x, mu, log_var, beta=1.0):
        """
        Compute the VAE loss with Chamfer Distance and KL divergence
        """
        batch_size = x.size(0)
        
        # Chamfer Distance
        x_expanded = x.unsqueeze(2)  # (B, 3, 1, N)
        reconstruction_expanded = reconstruction.unsqueeze(3)  # (B, 3, N, 1)
        
        # Compute pairwise distances
        dist = torch.sum((x_expanded - reconstruction_expanded) ** 2, dim=1)  # (B, N, N)
        
        # Compute bidirectional Chamfer distance
        chamfer_dist = torch.mean(torch.min(dist, dim=1)[0]) + torch.mean(torch.min(dist, dim=2)[0])
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_div = kl_div / batch_size
        
        # Total loss
        loss = chamfer_dist + beta * kl_div
        
        return loss
    
    def generate(self, num_samples=1):
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    
    def decode(self, z):
        return self.decoder(z)
    
    def encode(self, x):
        return self.pointnetfeat(x)

'''Debugging
pointnetfeat=PointNetFeat(input_transform=True, feature_transform=True)
points=Variable(torch.rand(size=(10,2048,3)))
print('input shape',points.shape)
out_feat, _ = pointnetfeat(points)
print('out feat shape',out_feat.shape)
'''



"""NEW IDEA FROM GUIBAS MITLIAGKAS ET AL."""
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PointNetGMMEncoder(nn.Module):
    """PointNet encoder that maps point clouds to a latent space"""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # point-wise mlp layers
        self.pointmlp1_1 = nn.Sequential(nn.Conv1d(3,64,1), nn.BatchNorm1d(64))
        self.pointmlp1_2 = nn.Sequential(nn.Conv1d(64,64,1), nn.BatchNorm1d(64))

        self.pointmlp2_1 = nn.Sequential(nn.Conv1d(64,64,1), nn.BatchNorm1d(64))
        self.pointmlp2_2 = nn.Sequential(nn.Conv1d(64,128,1), nn.BatchNorm1d(128))
        self.pointmlp2_3 = nn.Sequential(nn.Conv1d(128,1024,1), nn.BatchNorm1d(1024))
        
        self.fc = nn.Linear(1024, latent_dim)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - latent: [B,latent_dim]
        """
        # First MLP block
        x = F.relu(self.pointmlp1_1(pointcloud))
        x = F.relu(self.pointmlp1_2(x))
        
        # Second MLP block
        x = F.relu(self.pointmlp2_1(x))
        x = F.relu(self.pointmlp2_2(x))
        x = F.relu(self.pointmlp2_3(x))

        # Global max pooling
        x = torch.max(x, 2)[0]
        
        # Project to latent space
        latent = self.fc(x)
        return latent

class PointNetGMMDecoder(nn.Module):
    def __init__(self, num_points=2048, latent_dim=128):
        super().__init__()
        self.num_points = num_points
        
        self.fc1 = nn.Sequential(nn.Linear(latent_dim, int(num_points/4)), nn.BatchNorm1d(int(num_points/4)))
        self.fc2 = nn.Sequential(nn.Linear(int(num_points/4), int(num_points/2)), nn.BatchNorm1d(int(num_points/2)))
        self.fc3 = nn.Sequential(nn.Linear(int(num_points/2), num_points), nn.Dropout(), nn.BatchNorm1d(num_points))
        self.fc4 = nn.Linear(num_points, num_points*3)

        # self.fc1 = nn.Sequential(nn.Linear(latent_dim, int(num_points/8)), nn.BatchNorm1d(int(num_points/8)))
        # self.fc2 = nn.Sequential(nn.Linear(int(num_points/8), int(num_points/4)), nn.BatchNorm1d(int(num_points/4)))
        # self.fc3 = nn.Sequential(nn.Linear(int(num_points/4), num_points/2), nn.Dropout(), nn.BatchNorm1d(num_points/2))
        # self.fc4 = nn.Linear(num_points/2, num_points*3)    

    def forward(self, z):
        """
        Input:
            - z: latent vectors [B,latent_dim]
        Output:
            - pointcloud: [B,3,N]
        """
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.reshape(-1, 3, self.num_points)
        return x

class PointNetAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = PointNetGMMEncoder(latent_dim=latent_dim)
        self.decoder = PointNetGMMDecoder(latent_dim=latent_dim)
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


import torch
import torch.nn as nn
import torch.nn.functional as F

# Wrap in a module for easier integration with neural networks
class ApproxEMDLoss(nn.Module):
    def __init__(self, num_iterations=50):
        super().__init__()
        self.num_iterations = num_iterations

    def approx_emd_loss(self, x1, x2, num_iterations=50):
        """
        Approximated Earth Mover Distance based on the approach in 
        "Learning Representations and Generative Models for 3D Point Clouds"
        
        Args:
            x1: Point cloud 1, shape [B, 3, N] or [B, N, 3]
            x2: Point cloud 2, shape [B, 3, N] or [B, N, 3]
            num_iterations: Number of Sinkhorn iterations
            
        Returns:
            emd: Approximated EMD, shape [B]
        """
        # Make sure both point clouds have shape [B, N, 3]
        if x1.size(1) == 3:
            x1 = x1.transpose(1, 2)
        if x2.size(1) == 3:
            x2 = x2.transpose(1, 2)
        
        # Get batch size and number of points
        batch_size = x1.size(0)
        num_points = x1.size(1)
        
        # Compute pairwise squared distances between points
        # [B, N, N]
        cost_matrix = torch.sum(
            (x1.unsqueeze(2) - x2.unsqueeze(1)) ** 2, 
            dim=-1
        )
        
        # Initialize the transport matrix uniformly
        # [B, N, N]
        transport_matrix = torch.ones(
            batch_size, 
            num_points, 
            num_points, 
            device=x1.device
        ) / (num_points * num_points)
        
        # Initialize marginals for Sinkhorn iterations
        # [B, N]
        mu = torch.ones(
            batch_size, 
            num_points, 
            device=x1.device
        ) / num_points
        nu = torch.ones(
            batch_size, 
            num_points, 
            device=x1.device
        ) / num_points
        
        # Sinkhorn iterations
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        
        # Following notation from Algorithm 2 in paper "":
        # M is cost_matrix
        # r is mu (source marginal)
        # c is nu (target marginal)
        # K is transport_matrix
        # u, v are scaling factors
        
        for _ in range(num_iterations):
            # Scale transport matrix by costs
            K = torch.exp(-cost_matrix / 0.01)  # epsilon=0.01 as suggested in paper
            
            # Update u
            u = torch.log(mu + 1e-8) - torch.log(
                torch.sum(K * torch.exp(v.unsqueeze(1)), dim=2) + 1e-8
            )
            
            # Update v
            v = torch.log(nu + 1e-8) - torch.log(
                torch.sum(K * torch.exp(u.unsqueeze(2)), dim=1) + 1e-8
            )
            
            # Update transport matrix
            transport_matrix = K * torch.exp(u.unsqueeze(2) + v.unsqueeze(1))
        
        # Compute final EMD as sum of element-wise product of cost and transport matrices
        emd = torch.sum(transport_matrix * cost_matrix, dim=(1, 2))
        # print('emd shape',emd.shape)
        return emd

    def forward(self, x1, x2):
        return self.approx_emd_loss(x1, x2, self.num_iterations)


class ChamferDistanceLoss(nn.Module):
    """Chamfer Distance loss module"""
    def __init__(self):
        super().__init__()
    
    def chamfer_distance_loss(self,x1, x2):
        """
        Compute Chamfer Distance between two point clouds as defined in the paper
        "Learning Representations and Generative Models for 3D Point Clouds"
        
        Args:
            x1: Point cloud 1, shape [B, 3, N] or [B, N, 3]
            x2: Point cloud 2, shape [B, 3, N] or [B, N, 3]
            
        Returns:
            chamfer_dist: Chamfer distance, shape [B]
        """
        # Make sure both point clouds have shape [B, N, 3]
        if x1.size(1) == 3:
            x1 = x1.transpose(1, 2)
        if x2.size(1) == 3:
            x2 = x2.transpose(1, 2)
        
        # Get batch size and number of points
        batch_size = x1.size(0)
        num_points1 = x1.size(1)
        num_points2 = x2.size(1)
        
        # Compute pairwise distances between points
        # [B, N1, N2]
        dist_matrix = torch.sum(
            (x1.unsqueeze(2) - x2.unsqueeze(1)) ** 2,
            dim=-1
        )
        
        # Compute minimal distances in both directions
        # From x1 to x2: for each point in x1, find its closest point in x2
        min_dist_1_to_2 = torch.min(dist_matrix, dim=2)[0]  # [B, N1]
        
        # From x2 to x1: for each point in x2, find its closest point in x1
        min_dist_2_to_1 = torch.min(dist_matrix, dim=1)[0]  # [B, N2]
        
        # Sum the averages of minimal distances in both directions
        chamfer_dist = torch.mean(min_dist_1_to_2, dim=1) + torch.mean(min_dist_2_to_1, dim=1)
        # print('chamfer dist batch loss shape:',chamfer_dist.shape)
        return chamfer_dist


    # Optional: More efficient implementation using matrix operations
    def chamfer_distance_loss_efficient(self,x1, x2):
        """
        More efficient implementation of Chamfer Distance using matrix operations
        Mathematically equivalent but potentially faster on GPU
        """
        # Make sure both point clouds have shape [B, N, 3]
        if x1.size(1) == 3:
            x1 = x1.transpose(1, 2)
        if x2.size(1) == 3:
            x2 = x2.transpose(1, 2)
        
        # Compute squared norms of each point cloud
        x1_norm = torch.sum(x1 ** 2, dim=2, keepdim=True)    # [B, N1, 1]
        x2_norm = torch.sum(x2 ** 2, dim=2).unsqueeze(1)     # [B, 1, N2]
        
        # Compute cross term efficiently using batch matrix multiplication
        cross_term = torch.bmm(x1, x2.transpose(1, 2))       # [B, N1, N2]
        
        # Compute full distance matrix
        dist_matrix = x1_norm + x2_norm - 2 * cross_term     # [B, N1, N2]
        
        # Compute minimal distances
        min_dist_1_to_2 = torch.min(dist_matrix, dim=2)[0]   # [B, N1]
        min_dist_2_to_1 = torch.min(dist_matrix, dim=1)[0]   # [B, N2]
        
        # Compute final Chamfer distance
        chamfer_dist = torch.mean(min_dist_1_to_2, dim=1) + torch.mean(min_dist_2_to_1, dim=1)
        
        return chamfer_dist
    
    def forward(self, x1, x2):
        return self.chamfer_distance_loss_efficient(x1, x2)


# Modified training function to use the ModelNet loaders
def train_vae_with_modelnet(epochs=100, batch_size=24, log_interval=10, latent_dim=128, hidden_dim=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the VAE model
    model = VAEPointCloud(latent_dim=latent_dim,hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Load ModelNet data
    train_loader, test_loader = load_modelnet(batch_size=batch_size)
    
    print(f"Training on device: {device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Create directory for saving results if it doesn't exist
    os.makedirs('vae_pointcloud_training_data', exist_ok=True)
    
    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            # Unpack the tuple returned by loss_function
            loss = model.loss_function(recon_batch, data, mu, logvar)
            
            # Backpropagate the loss
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item() / len(data)))
        
        avg_loss = train_loss / len(train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch + 1, avg_loss))
        return avg_loss

    def test(epoch):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon, mu, logvar = model(data)
                # Unpack the tuple returned by loss_function
                loss = model.loss_function(recon, data, mu, logvar)
                test_loss += loss
                if i == 0:
                    # Save the first batch of reconstructions
                    comparison = torch.cat([data[:8], recon[:8]])
                    torch.save(comparison.cpu(), 
                             f'vae_pointcloud_training_data/modelnet_reconstruction_{epoch + 1}.pt')
        
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss

    # Training loop
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        train_loss = train(epoch)
        test_loss = test(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Save sample generations
        if epoch % log_interval == 0:
            with torch.no_grad():
                sample = torch.randn(64, 128).to(device)
                sample = model.decode(sample).cpu()
                torch.save(sample, f'vae_pointcloud_training_data/modelnet_sample_{epoch + 1}.pt')
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses,
            }, f'vae_pointcloud_training_data/modelnet_vae_checkpoint_{epoch + 1}.pt')

    # Save final model
    torch.save(model.state_dict(), 'vae_pointcloud_training_data/modelnet_vae_final.pt')
    
    return model, train_losses, test_losses


def train_old_vae_with_modelnet(epochs=100, batch_size=24, log_interval=10, latent_dim=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the VAE model
    model = VAEPointCloudOLD(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Load ModelNet data
    train_loader, test_loader = load_modelnet(batch_size=batch_size)
    
    print(f"Training on device: {device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Create directory for saving results if it doesn't exist
    os.makedirs('vae_old_training_data', exist_ok=True)
    
    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            # Unpack the tuple returned by loss_function
            loss = model.loss_function(recon_batch, data, mu, logvar)
            
            # Backpropagate the loss
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item() / len(data)))
        
        avg_loss = train_loss / len(train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch + 1, avg_loss))
        return avg_loss

    def test(epoch):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon, mu, logvar = model(data)
                # Unpack the tuple returned by loss_function
                loss= model.loss_function(recon, data, mu, logvar)
                test_loss += loss
                if i == 0:
                    # Save the first batch of reconstructions
                    comparison = torch.cat([data[:8], recon[:8]])
                    torch.save(comparison.cpu(), 
                             f'vae_old_training_data/modelnet_reconstruction_{epoch + 1}.pt')
        
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss

    # Training loop
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        train_loss = train(epoch)
        test_loss = test(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Save sample generations
        if epoch % log_interval == 0:
            with torch.no_grad():
                sample = torch.randn(64, 128).to(device)
                sample = model.decode(sample).cpu()
                torch.save(sample, f'vae_old_training_data/modelnet_sample_{epoch + 1}.pt')
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses,
            }, f'vae_old_training_data/modelnet_vae_checkpoint_{epoch + 1}.pt')

    # Save final model
    torch.save(model.state_dict(), 'vae_old_training_data/modelnet_vae_final.pt')
    
    return model, train_losses, test_losses


def train_pointnet_with_modelnet(epochs=100, batch_size=24, log_interval=10, latent_dim=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNetVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader, test_loader, _ = load_modelnet_chair(batch_size=batch_size)
    
    print(f"Training on device: {device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")


    # Create directory for saving results if it doesn't exist
    os.makedirs('vae_pointnet_training_data', exist_ok=True)
    
    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            # Unpack the tuple returned by loss_function
            loss = model.loss_function(recon_batch, data, mu, logvar)
            
            # Backpropagate the loss
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item() / len(data)))
        
        avg_loss = train_loss / len(train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch + 1, avg_loss))
        return avg_loss

    def test(epoch):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon, mu, logvar = model(data)
                # Unpack the tuple returned by loss_function
                loss = model.loss_function(recon, data, mu, logvar)
                test_loss += loss
                if i == 0:
                    # Save the first batch of reconstructions
                    comparison = torch.cat([data[:8], recon[:8]])
                    torch.save(comparison.cpu(), 
                             f'vae_pointnet_training_data/modelnet_reconstruction_{epoch + 1}.pt')
        
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss

    # Training loop
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        train_loss = train(epoch)
        test_loss = test(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Save sample generations
        if epoch % log_interval == 0:
            with torch.no_grad():
                sample = torch.randn(64, latent_dim).to(device)
                sample = model.decode(sample).cpu()
                torch.save(sample, f'vae_pointnet_training_data/modelnet_sample_{epoch + 1}.pt')
        
        # Save model checkpoint
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses,
            }, f'vae_pointnet_training_data/modelnet_vae_checkpoint_{epoch + 1}.pt')

    # Save final model
    torch.save(model.state_dict(), 'vae_pointnet_training_data/modelnet_vae_final.pt')
    
    return model, train_losses, test_losses


def train_pointnet_ae_with_modelnet(epochs=100, batch_size=24, log_interval=10, latent_dim=128,loss_mode='cd'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize autoencoder
    model = PointNetAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_loader, test_loader, _ = load_modelnet_chair(batch_size=batch_size)
    print(f"Training on device: {device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Create directory for saving results
    os.makedirs('ae_pointnet_training_data', exist_ok=True)
    
    def train(epoch,loss_mode='cd'):
        model.train()
        train_loss = 0
        if loss_mode=='emd':
            criterion = ApproxEMDLoss(num_iterations=50)
        elif loss_mode=='cd':
            criterion = ChamferDistanceLoss()
        else:
            raise ValueError
        
        for batch_idx, (data, *_) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass through autoencoder
            z = model.encode(data)
            recon = model.decode(z)
            
            # Compute reconstruction loss using (EMD or CD)
            batch_losses = criterion(recon, data)
            loss = batch_losses.mean() #take mean/sum of batch losses to have a scalar for backprop

            # Backpropagate
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                
        avg_loss = train_loss / len(train_loader)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch + 1, avg_loss))
        return avg_loss
    
    def test(epoch,loss_mode='cd'):
        model.eval()
        test_loss = 0

        if loss_mode=='emd':
            criterion = ApproxEMDLoss(num_iterations=50)
        elif loss_mode=='cd':
            criterion = ChamferDistanceLoss()
        else:
            raise ValueError
        
        with torch.no_grad():
            for i, (data, *_) in enumerate(test_loader):
                data = data.to(device)
                
                # Forward pass
                z = model.encode(data)
                recon = model.decode(z)
                
                # Compute reconstruction loss using (EMD or CD)
                batch_losses = criterion(recon, data)
                loss = batch_losses.mean() #take mean/sum of batch losses to have a scalar for backprop

                test_loss += loss.item()
                
                if i == 0:
                    # Save first batch reconstructions
                    comparison = torch.cat([data[:8], recon[:8]])
                    torch.save(comparison.cpu(),
                             f'ae_pointnet_training_data/modelnet_reconstruction_{epoch + 1}.pt')
                    
        test_loss /= len(test_loader)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss
    
    # Training loop
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        train_loss = train(epoch,loss_mode=loss_mode)
        test_loss = test(epoch,loss_mode=loss_mode)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Save checkpoints
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses,
            }, f'ae_pointnet_training_data/modelnet_ae_checkpoint_{epoch + 1}.pt')
    
    # Save final model
    torch.save(model.state_dict(), 'ae_pointnet_training_data/modelnet_ae_final.pt')
    
    # After AE training, fit GMM on the latent space
    model.eval()
    encoded_data = []
    with torch.no_grad():
        for data, *_ in train_loader:
            data = data.to(device)
            z = model.encode(data)
            encoded_data.append(z.cpu().numpy())
    encoded_data = np.concatenate(encoded_data, axis=0)
    encoded_data_save_path = os.path.join('ae_pointnet_training_data', 'encoded_data.npy')
    np.save(encoded_data_save_path, encoded_data)
    print(f"Saved encoded data to {encoded_data_save_path}")
    
    return model, train_losses, test_losses



if __name__ == '__main__':
    # model, train_losses, test_losses = train_old_vae_with_modelnet(epochs=100, batch_size=16, log_interval=10,latent_dim=128)
    # model, train_losses, test_losses = train_vae_with_modelnet(epochs=100, batch_size=16, log_interval=10,latent_dim=128,hidden_dim=512)
    # model, train_losses, test_losses = train_pointnet_with_modelnet(epochs=100, batch_size=16, log_interval=10,latent_dim=256)
    model, train_losses, test_losses = train_pointnet_ae_with_modelnet(epochs=100, batch_size=8, log_interval=10,latent_dim=128,loss_mode='emd')
