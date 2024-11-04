import torch
from plot_utils import get_vae_pointclouds
import numpy as np
from sklearn.mixture import GaussianMixture
try:
    from .energies.vae_utils import load_modelnet, load_modelnet_chair
except:
    from energies.vae_utils import load_modelnet, load_modelnet_chair

from energies.vae_pointcloud import PointNetAE, ApproxEMDLoss, ChamferDistanceLoss

import matplotlib.pyplot as plt

class PointCloudGMM:
    """Separate GMM class without autoencoder dependency"""
    def __init__(self, latent_dim=128, n_components=32, device='cuda'):
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.device = device
        
        self.gmm = None
        # Cache for GMM parameters as tensors
        self.means = None
        self.covariances = None
        self.weights = None
        self.precisions = None
        self.normalizing_constants = None
        
    def fit(self, encoded_data_path):
        """Fit GMM on saved encoded data"""
        # Load encoded data from numpy file
        print(f"Loading encoded data from {encoded_data_path}")
        encoded_data = np.load(encoded_data_path)
        
        print(f"Fitting GMM with {self.n_components} components...")
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=42
        )
        self.gmm.fit(encoded_data)
        self._cache_gmm_params()
        print("GMM fitting complete")
        
    def _cache_gmm_params(self):
        """Cache GMM parameters as torch tensors"""
        self.means = torch.tensor(self.gmm.means_, dtype=torch.float32).to(self.device)
        self.covariances = torch.tensor(self.gmm.covariances_, dtype=torch.float32).to(self.device)
        self.weights = torch.tensor(self.gmm.weights_, dtype=torch.float32).to(self.device)
        
        # Pre-compute precision matrices and normalizing constants
        self.precisions = torch.tensor(
            np.array([np.linalg.inv(cov) for cov in self.gmm.covariances_]),
            dtype=torch.float32
        ).to(self.device)
        
        self.normalizing_constants = torch.tensor(
            [-0.5 * (self.latent_dim * np.log(2 * np.pi) + np.log(np.linalg.det(cov))) 
             for cov in self.gmm.covariances_],
            dtype=torch.float32
        ).to(self.device)
    
    # def _cache_gmm_params(self):
    #     """Cache GMM parameters as torch tensors with stable computations"""
    #     self.means = torch.tensor(self.gmm.means_, dtype=torch.float32).to(self.device)
    #     self.covariances = torch.tensor(self.gmm.covariances_, dtype=torch.float32).to(self.device)
    #     self.weights = torch.tensor(self.gmm.weights_, dtype=torch.float32).to(self.device)
        
    #     # Pre-compute precision matrices
    #     # Add small diagonal term for numerical stability
    #     eps = 1e-6
    #     stabilized_covs = self.covariances + eps * torch.eye(
    #         self.latent_dim, 
    #         device=self.device
    #     ).unsqueeze(0)
        
    #     # Ensure the covariance matrices are positive-definite
    #     for i in range(stabilized_covs.shape[0]):
    #         while True:
    #             try:
    #                 # Attempt Cholesky decomposition
    #                 L = torch.linalg.cholesky(stabilized_covs[i])
    #                 break  # Break if successful
    #             except RuntimeError:
    #                 # If it fails, add a larger value to the diagonal
    #                 stabilized_covs[i] += eps * torch.eye(self.latent_dim, device=self.device)
        
    #     # Compute precisions using torch.inverse
    #     self.precisions = torch.stack([
    #         torch.inverse(cov) for cov in stabilized_covs
    #     ])
    #
    #    # Compute normalizing constants using torch operations
    #    # log(det(A)) = 2 * sum(log(diag(L))) where L is the Cholesky factor of A
    #    # This is more numerically stable than computing determinant directly
    #    log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=1, dim2=2)), dim=1)
    #    
    #    self.normalizing_constants = -0.5 * (
    #        self.latent_dim * torch.log(2 * torch.tensor(np.pi)) + log_det
    #    )
        
    def save(self, save_path):
        """Save GMM parameters"""
        gmm_params = {
            'means': self.gmm.means_,
            'covariances': self.gmm.covariances_,
            'weights': self.gmm.weights_
        }
        np.save(save_path, gmm_params)
        print(f"Saved GMM parameters to {save_path}")
    
    def load(self, load_path):
        """Load GMM parameters"""
        print(f"Loading GMM parameters from {load_path}")
        gmm_params = np.load(load_path, allow_pickle=True).item()
        
        self.gmm = GaussianMixture(
            n_components=len(gmm_params['weights']),
            covariance_type='full'
        )
        self.gmm.means_ = gmm_params['means']
        self.gmm.covariances_ = gmm_params['covariances']
        self.gmm.weights_ = gmm_params['weights']
        
        # Initialize other GMM parameters
        self.gmm.precisions_cholesky_ = np.array([
            np.linalg.cholesky(np.linalg.inv(cov)) 
            for cov in self.gmm.covariances_
        ])
        
        self._cache_gmm_params()
    
    def energy(self, z, x=None):
        """Compute energy as negative log probability under the GMM"""
        return -self.log_prob(z)
    
    def log_prob(self, z):
        """Compute log probability under the GMM"""
        z = z.unsqueeze(1)  # [B, 1, D]
        means = self.means.unsqueeze(0)  # [1, K, D]
        
        # Compute quadratic term (z-μ)ᵀΣ⁻¹(z-μ) for each component
        diff = z - means  # [B, K, D]
        quad_form = torch.sum(
            diff.unsqueeze(-2) @ self.precisions.unsqueeze(0) @ diff.unsqueeze(-1),
            dim=(-2, -1)
        ).squeeze(-1)  # [B, K]
        
        # Compute log probabilities for each component
        log_probs = self.normalizing_constants - 0.5 * quad_form  # [B, K]
        log_probs = log_probs + torch.log(self.weights)
        return torch.logsumexp(log_probs, dim=-1)  # [B]
    
    def sample(self, n_samples):
        """Sample from the GMM"""
        return torch.tensor(
            self.gmm.sample(n_samples)[0], 
            dtype=torch.float32
        ).to(self.device)

def compute_metrics(generated_points, test_points, resolution=28):
    """
    Compute all metrics from the paper using pre-implemented distance functions
    
    Args:
        generated_points: [N, 3, num_points] tensor of generated point clouds
        test_points: [M, 3, num_points] tensor of test point clouds
        resolution: resolution for JSD voxel grid
    
    Returns:
        dict of metrics
    """
    # Initialize distance functions
    chamfer_dist = ChamferDistanceLoss()
    emd_dist = ApproxEMDLoss()
    
    def compute_mmd(x, y, distance_fn):
        """
        Compute MMD using either Chamfer or EMD distance
        """
        with torch.no_grad():
            N = len(x)
            M = len(y)
            min_distances = []
            
            # For each generated point cloud
            for i in range(N):
                distances = []
                x_i = x[i:i+1]  # [1, 3, num_points]
                
                # Find minimum distance to test set
                for j in range(M):
                    y_j = y[j:j+1]  # [1, 3, num_points]
                    dist = distance_fn(x_i, y_j)
                    distances.append(dist)
                    
                min_distances.append(torch.min(torch.stack(distances)))
            
            return torch.mean(torch.stack(min_distances))
    
    def compute_coverage(x, y, distance_fn):
        """
        Compute coverage score using either Chamfer or EMD distance
        """
        with torch.no_grad():
            N = len(x)
            M = len(y)
            covered = torch.zeros(M, device=x.device)
            
            # For each generated point cloud
            for i in range(N):
                distances = []
                x_i = x[i:i+1]
                
                # Find closest test point cloud
                for j in range(M):
                    y_j = y[j:j+1]
                    dist = distance_fn(x_i, y_j)
                    distances.append(dist)
                
                # Mark the closest test point cloud as covered
                closest_idx = torch.argmin(torch.stack(distances))
                covered[closest_idx] = 1
                
            return torch.mean(covered)
    
    def compute_jsd(x, y, resolution):
        """
        Compute JSD using voxel grid occupancy
        """
        def voxelize(points, res):
            """Convert point clouds to voxel occupancy grid"""
            voxels = torch.zeros(len(points), res, res, res, device=points.device)
            
            # Normalize points to [0, res-1]
            points = points.clone()  # [N, 3, P]
            points = points.permute(0, 2, 1)  # [N, P, 3]
            points = ((points + 1) * (res - 1) / 2).long()
            points = points.clamp(0, res - 1)
            
            # Mark occupied voxels
            for i in range(len(points)):
                voxels[i, points[i, :, 0], points[i, :, 1], points[i, :, 2]] = 1
                
            return voxels
        
        with torch.no_grad():
            voxels_gen = voxelize(x, resolution)
            voxels_ref = voxelize(y, resolution)
            
            # Compute distributions
            P = torch.mean(voxels_gen.flatten(1), dim=0)
            Q = torch.mean(voxels_ref.flatten(1), dim=0)
            M = 0.5 * (P + Q)
            
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            P = P + eps
            Q = Q + eps
            M = M + eps
            
            # Compute JSD
            jsd = 0.5 * (
                torch.sum(P * torch.log2(P / M)) +
                torch.sum(Q * torch.log2(Q / M))
            )
            
            return jsd
    
    # Compute all metrics
    with torch.no_grad():
        metrics = {
            'MMD-CD': compute_mmd(generated_points, test_points, chamfer_dist).item(),
            'MMD-EMD': compute_mmd(generated_points, test_points, emd_dist).item(),
            'Coverage-CD': compute_coverage(generated_points, test_points, chamfer_dist).item(),
            'Coverage-EMD': compute_coverage(generated_points, test_points, emd_dist).item(),
            'JSD': compute_jsd(generated_points, test_points, resolution).item()
        }
    
    return metrics

# Example usage:
def evaluate_samples(gmm, decoder, test_loader, device='cuda', batch_size=16):
    """Evaluate samples from GMM using loaded decoder"""
    decoder.eval()
    
    # Get test set point clouds
    test_points = []
    with torch.no_grad():
        for data, *_ in test_loader:
            test_points.append(data.to(device))
    test_points = torch.cat(test_points, dim=0)
    
    # Generate samples
    generated_points = []
    with torch.no_grad():
        num_batches = (len(test_points) + batch_size - 1) // batch_size
        print('Number of batches of point clouds:', num_batches, ', total no. of pointclouds', len(test_points))
        for i in range(num_batches):
            print('Generating point cloud ',i,'...')
            z = gmm.sample(batch_size)
            points = decoder(z)
            generated_points.append(points)
            
    generated_points = torch.cat(generated_points, dim=0)[:len(test_points)]
    
    print('Computing evaluation metrics')
    # Compute metrics
    #metrics = compute_metrics(generated_points, test_points)
    metrics = np.zeros(5)
    return metrics, generated_points

# Print metrics in a nice format
def print_metrics(metrics):
    print("\nEvaluation Metrics:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric:<15}: {value:.6f}")
    print("-" * 50)


# Example usage:
def main(root_path='ae_pointnet_training_data', latent_dim=256,batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load GMM
    gmm = PointCloudGMM(latent_dim=latent_dim, device=device)
    gmm.fit(f'{root_path}/encoded_data.npy')
    # or load pre-trained GMM
    # gmm.load('path/to/gmm_parameters.npy')
    
    # Load autoencoder
    ae = PointNetAE(latent_dim=latent_dim).to(device)
    checkpoint = torch.load(f'{root_path}/modelnet_ae_final.pt')
    ae.load_state_dict(checkpoint)
    print('Loaded state dict of AE model.')


    train_loader, test_loader, evaluation_subset = load_modelnet_chair() 
    print('Loaded ModelNet Dataset.')

    # Generate and evaluate samples
    print('Calculating evaluation metrics and samples...')
    # metrics, samples = evaluate_samples(gmm, ae.decoder, test_loader, device, batch_size=batch_size)
    # TRY OVERFIT CASE
    metrics, samples = evaluate_samples(gmm, ae.decoder, train_loader, device, batch_size=batch_size)
    print("Evaluation metrics:", metrics)
    
    # Compute energy for some samples
    print('Sampling 10 new GMM samples to calculate energies')
    z = gmm.sample(10)
    energy = gmm.energy(z)
    print("Energy of 10 GMM samples:", energy)

    gmm.save(save_path=f'{root_path}/fitted_gmm_params.npy')
    print('Saved fitted GMM parameters.')

    print('Visualizing generated point clouds')
    # Visualize GMM generated samples
    fig, ax = get_vae_pointclouds(samples[:16])
    plt.show()
    
if __name__=='__main__':
    main(root_path='ae_pointnet_training_data_100_bs16_cd_ineff_ld256',latent_dim=256,batch_size=16)