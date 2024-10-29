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

from .vae_utils import load_modelnet

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
        
        return loss, chamfer_dist, kl_div

    def generate(self, num_samples=1):
        """
        Generate new point clouds from random latent vectors
        """
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    
class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(PointNetEncoder, self).__init__()
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


class PointNetDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(PointNetDecoder, self).__init__()
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


class VAEPointCloudOLD(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAEPointCloudOLD, self).__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = PointNetDecoder(latent_dim)

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
        z = z.view(-1, 3 * 2048)        # (batch_size, 3 * 2048=6144)
        return torch.sigmoid(z)
    

    
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

# Modified training function to use the ModelNet loaders
def train_vae_with_modelnet(epochs=100, batch_size=32, log_interval=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the VAE model
    model = VAEPointCloud(latent_dim=128,hidden_dim=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Load ModelNet data
    train_loader, test_loader = load_modelnet(batch_size=batch_size)
    
    print(f"Training on device: {device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Create directory for saving results if it doesn't exist
    os.makedirs('vae_training_data', exist_ok=True)
    
    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            # Unpack the tuple returned by loss_function
            loss, chamfer_dist, kl_div = model.loss_function(recon_batch, data, mu, logvar)
            
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
                loss, chamfer_dist, kl_div = model.loss_function(recon, data, mu, logvar)
                test_loss += loss
                if i == 0:
                    # Save the first batch of reconstructions
                    comparison = torch.cat([data[:8], recon[:8]])
                    torch.save(comparison.cpu(), 
                             f'vae_training_data/modelnet_reconstruction_{epoch + 1}.pt')
        
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
                torch.save(sample, f'vae_training_data/modelnet_sample_{epoch + 1}.pt')
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses,
            }, f'vae_training_data/modelnet_vae_checkpoint_{epoch + 1}.pt')

    # Save final model
    torch.save(model.state_dict(), 'vae_training_data/modelnet_vae_final.pt')
    
    return model, train_losses, test_losses

if __name__ == '__main__':
    model, train_losses, test_losses = train_vae_with_modelnet(epochs=100, batch_size=16, log_interval=10)

