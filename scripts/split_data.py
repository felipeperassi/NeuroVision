from config import DIR_IDX, IDX_IMGS, IDX_TRAIN, IDX_TEST, SEED
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(idx_dir : str, idx_img_path : str, idx_train_path : str, idx_test_path : str, seed : int) -> None:
    """
    Splits the dataset into training and testing sets based on unique image indices.
        Args:
            - idx_dir (str): Directory where the index files will be saved.
            - idx_img_path (str): Path to the .npy file containing image indices for each trial.
            - idx_train_path (str): Path where the training indices will be saved as a .npy file.
            - idx_test_path (str): Path where the testing indices will be saved as a .npy file.
            - seed (int): Random seed for reproducibility of the train-test split.
    """
    img_idx     = np.load(idx_img_path).flatten()
    unique_imgs = np.unique(img_idx)

    train_imgs, test_imgs = train_test_split(unique_imgs, test_size=0.1, random_state=seed)

    train_indices = np.where(np.isin(img_idx, train_imgs))[0]
    test_indices  = np.where(np.isin(img_idx, test_imgs))[0]

    if not idx_dir.exists():
        idx_dir.mkdir(parents=True)
        
    np.save(idx_train_path, train_indices)
    np.save(idx_test_path,  test_indices)

    print(f"Train trials: {len(train_indices)} | Test trials: {len(test_indices)}")
    print(f"Train imgs: {len(train_imgs)} | Test imgs: {len(test_imgs)}")

if __name__ == "__main__":
    print(f"Splitting data using image indices from {IDX_IMGS} with seed {SEED}...")
    split_data(DIR_IDX, IDX_IMGS, IDX_TRAIN, IDX_TEST, SEED)
    print(f"Data split completed. Train and test indices saved to {DIR_IDX}.")