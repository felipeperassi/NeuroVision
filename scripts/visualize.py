from config import DATA_IMAGES, IDX_TEST
from inference import inference, process_inference

import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

def visualize(cnn_name : str, idx_image : int) -> None:
    """
    Visualizes the structural & inference results for a given CNN model and image index.
        Args:
            - cnn_name (str): Name of the CNN model ("VAE" or "VGG") to determine which processing pipeline to use.
            - idx_image (int): Index of the image to visualize.
    """
    if cnn_name.upper() not in ["VAE", "VGG"]:
        raise ValueError(f"Invalid CNN name: {cnn_name}. Must be 'VAE' or 'VGG'.")

    structural_img = process_inference(cnn_name, idx_image)
    inference_img = inference(cnn_name, idx_image)
    
    idx_test = np.load(IDX_TEST)
    idx = idx_test[idx_image]
    with h5py.File(DATA_IMAGES, 'r') as f:
        original_img = f['imgBrick'][idx]
    
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title(f'GT (trial {idx})')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(structural_img)
    plt.title('Structural Image')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(inference_img)
    plt.title('MLP + CNN Inference')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Master visualize script.')
    parser.add_argument('--cnn', type=str, required=True, 
                        choices=['VAE', 'VGG'], help='Select model: VAE or VGG.')
    parser.add_argument('--idx', type=int, required=True, 
                        help='Index of the image to be processed.', 
                        choices=range(3000))
    
    args = parser.parse_args()
    visualize(cnn_name=args.cnn, idx_image=int(args.idx))