import torch
from energies.vae_pointcloud import VAEPointCloud, VAEPointCloudOLD, PointNetVAE
from energies.vae import VAE
from plot_utils import get_vae_pointclouds, get_vae_images
import matplotlib.pyplot as plt
import os

def load_model(checkpoint_path, device, type='pointnet'):
    """
    Load the VAEPointCloud model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        device (torch.device): Device to load the model onto.

    Returns:
        model (VAEPointCloud): Loaded VAEPointCloud model.
    """
    if type == 'pointnet':
        model = VAEPointCloud(latent_dim=128, hidden_dim=512).to(device)
    elif type == 'old':
        model = VAEPointCloudOLD(latent_dim=128).to(device)
    elif type == 'mnist':
        model = VAE().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def load_model_old(checkpoint_path, device):
    """
    Load the VAEPointCloud model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        device (torch.device): Device to load the model onto.

    Returns:
        model (VAEPointCloud): Loaded VAEPointCloud model.
    """
    model = VAEPointCloudOLD(latent_dim=128).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def load_model_pointnet(checkpoint_path, device):
    model = PointNetVAE(latent_dim=256).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def generate_samples(model, num_samples=16, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Generate new point cloud samples using the trained VAE model.

    Args:
        model (VAEPointCloud): Trained VAEPointCloud model.
        num_samples (int): Number of samples to generate.
        device (torch.device): Device to perform computations on.

    Returns:
        samples (torch.Tensor): Generated point clouds of shape (num_samples, 3, 2048).
    """
    with torch.no_grad():
        samples = model.generate(num_samples=num_samples)
    return samples

def visualize_pointclouds(samples, save_path=None):
    """
    Visualize the generated point clouds using a 3D scatter plot.

    Args:
        samples (torch.Tensor): Generated point clouds of shape (num_samples, 3, 2048).
        save_path (str, optional): Path to save the visualization image. Defaults to None.
    """
    fig, axs = get_vae_pointclouds(samples)
    plt.show()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Point cloud visualization saved to {save_path}")

def main():
    # Configuration
    checkpoint_path = 'energies/vae_pointnet_training_data/modelnet_vae_final.pt'  # Path to the trained model checkpoint
    num_samples = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'vae_generated_samples'
    os.makedirs(output_dir, exist_ok=True)
    visualization_path = os.path.join(output_dir, 'generated_pointclouds.png')

    # Load the trained model
    print("Loading the trained VAE model...")
    model = load_model(checkpoint_path, device, type='pointnet')
    print("Model loaded successfully.")

    # Generate samples
    print(f"Generating {num_samples} new point cloud samples...")
    samples = generate_samples(model, num_samples=num_samples, device=device)
    print("Samples generated with SHAPE", samples.shape)
    # samples = samples.view(num_samples, 2048, 3)
    # print("Samples with flipped dimensions to (N, 2048, 3)", samples.view(num_samples, 2048, 3))

    # Visualize the generated samples
    print("Visualizing the generated point clouds...")
    visualize_pointclouds(samples, save_path=visualization_path)
    print("Visualization complete.")

def main_old():
    # Configuration
    checkpoint_path = 'energies/vae_old_training_data/modelnet_vae_final.pt'  # Path to the trained model checkpoint
    num_samples = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'vae_old_generated_samples'
    os.makedirs(output_dir, exist_ok=True)
    visualization_path = os.path.join(output_dir, 'generated_pointclouds.png')

    # Load the trained model
    print("Loading the trained OLD VAE model...")
    model = load_model(checkpoint_path, device, type='old')
    print("Model loaded successfully.")

    # Generate samples
    print(f"Generating {num_samples} new point cloud samples...")
    samples = generate_samples(model, num_samples=num_samples, device=device)
    print("Samples generated with SHAPE", samples.shape)
    samples = samples.view(num_samples, 3, 2048)
    print("Samples with flipped dimensions to (N, 2048, 3)")

    # Visualize the generated samples
    print("Visualizing the generated point clouds...")
    visualize_pointclouds(samples, save_path=visualization_path)
    print("Visualization complete.")

def main_pointnet():
    checkpoint_path = 'energies/vae_pointnet_training_data/modelnet_vae_final.pt'
    num_samples = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'vae_pointnet_generated_samples'
    os.makedirs(output_dir, exist_ok=True)
    visualization_path = os.path.join(output_dir, 'generated_pointclouds.png')

    # Load the trained model
    print("Loading the trained OLD VAE model...")
    model = load_model_pointnet(checkpoint_path, device)
    print("Model loaded successfully.")

    # Generate samples
    print(f"Generating {num_samples} new point cloud samples...")
    samples = generate_samples(model, num_samples=num_samples, device=device)
    print("Samples generated with SHAPE", samples.shape)
    samples = samples.view(num_samples, 3, 2048)
    print("Samples with flipped dimensions to (N, 2048, 3)")

    # Visualize the generated samples
    print("Visualizing the generated point clouds...")
    visualize_pointclouds(samples, save_path=visualization_path)
    print("Visualization complete.")

def main_mnist():
    checkpoint_path = 'energies/data/mnist_vae_100_epochs_seed_1_mps.pt'
    num_samples = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'vae_mnist_generated_samples'
    os.makedirs(output_dir, exist_ok=True)
    visualization_path = os.path.join(output_dir, 'generated_images.png')

    # Load the trained model
    print("Loading the trained VAE model...")
    model = load_model(checkpoint_path, device, type='mnist')
    print("Model loaded successfully.")

    # Generate samples
    print(f"Generating {num_samples} new point cloud samples...")
    samples = generate_samples(model, num_samples=num_samples, device=device)
    print("Samples generated with SHAPE", samples.shape)
    get_vae_images(samples)

if __name__ == '__main__':
    #main()
    # main_old()
    # main_pointnet()
    main_mnist()