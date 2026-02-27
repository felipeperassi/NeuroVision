from config import DATA_VOXELS_ST, DATA_VOXELS, IDX_TRAIN
import numpy as np

def compute_stats(data_path : str, idx_img_path : str) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the mean and standard deviation of the data across trials.
        Args:
            - data_path (str): Path to the .npy file containing voxel data of shape (trials, voxel_dim).
            - idx_img_path (str): Path to the .npy file containing training indices.
        Returns:
            - tuple[np.ndarray, np.ndarray]: Mean and standard deviation of voxel data.
    """
    data = np.load(data_path)
    train_indices = np.load(idx_img_path)
    
    mean = np.mean(data[train_indices], axis=0)
    std = np.std(data[train_indices], axis=0) + 1e-12  # Add small value to avoid division by zero

    print(f"Mean shape: {mean.shape} | Std shape: {std.shape}")
    return mean, std

def save_stats(mean: np.ndarray, std: np.ndarray, stats_path: str) -> None:
    """
    Saves the computed mean and standard deviation to .npy files.
        Args:
            - mean (np.ndarray): Mean of voxel data.
            - std (np.ndarray): Standard deviation of voxel data.
            - stats_path (str): Path to save the .npz file containing mean and std.
    """
    np.savez(stats_path, mean=mean, std=std)

if __name__ == "__main__":
    print(f"Computing voxel statistics from {DATA_VOXELS} using training indices from {IDX_TRAIN}...")
    mean, std = compute_stats(DATA_VOXELS, IDX_TRAIN)
    save_stats(mean, std, DATA_VOXELS_ST)
    print(f"Voxel stats computed and saved to {DATA_VOXELS_ST}.")