import torch
import matplotlib.pyplot as plt
from plot_utils import get_vae_pointclouds
# Path to your .pt file
pt_file_path = 'ae_pointnet_training_data_100_bs16_cd_eff_ld128/modelnet_reconstruction_99.pt'  # Replace with your file path

# Load the tensor
# Assuming the saved tensor has shape [16, 2048, 3] (e.g., 8 originals + 8 reconstructions)
data = torch.load(pt_file_path)

# Verify the tensor shape
print(f"Loaded tensor shape: {data.shape}")  # Expected: [16, N, 3]

# data = data.permute(0, 2, 1)
# print(f"Permuted tensor shape: {data.shape}")  # Expected: [16, N, 3]

# Select a specific point cloud to visualize
# For example, visualize the first reconstructed point cloud (index 8 to 15)
# reconstructed = data[0].unsqueeze(0)  # Indexing based on how you saved the data
reconstructed = data  # Indexing based on how you saved the data
# If you saved as data[:8] originals and data[8:] reconstructions

print(f"Reconstructed shape: {reconstructed.shape}")

fig, ax = get_vae_pointclouds(reconstructed)
plt.show()
# Convert to NumPy
# reconstructed_np = reconstructed.numpy()

# # Plot using Matplotlib
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot
# ax.scatter(reconstructed_np[:, 0], reconstructed_np[:, 1], reconstructed_np[:, 2],
#            c='r', marker='o', s=1)

# # Set labels
# ax.set_title('Reconstructed Point Cloud')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')