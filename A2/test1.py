import numpy as np
from scipy.interpolate import RegularGridInterpolator
from PIL import Image 
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    """Converting RGB image to greyscale.
    Args: rgb: numpy array of shape (H, W, 3)
    Returns: gray: numpy array of shape (H, W)
    """
    H,W = rgb.shape[:2]
    gray = np.zeros((H, W), dtype=np.float64)
    weights = [0.2126, 0.7152, 0.0722]
    for i in range(H):
         for j in range(W):
			# execute dot product of r,g,b values and their weights.
            gray[i][j] = np.dot(rgb[i, j, :], weights)
    return gray

def load_data(i0_path, i1_path, gt_path):
    """Loading data.
    Args: i0_path: path to the first image
          i1_path: path to the second image
          gt_path: path to the disparity image
    Returns: i_0: numpy array of shape (H, W)
             i_1: numpy array of shape (H, W)
             g_t: numpy array of shape (H, W)
    """
    i_0 = np.array(Image.open(i0_path), dtype=np.float64)/255
    i_1 = np.array(Image.open(i1_path), dtype=np.float64)/255
    g_t = np.array(Image.open(gt_path), dtype=np.float64)
    return i_0, i_1, g_t

def constant_disparity(disparity_size, a):
    """
    Args: disparity_size: tuple containg height and width (H, W)
          a: value to initialize with
    Returns: disparity_map: numpy array of shape (H, W)
    """
    disparity_map = np.full(disparity_size, a, dtype=np.float64)
    return disparity_map

def shift_interpolated_disparity(im1, d):
    """Shift im1 by d.
    Since disparity can now be continuous, use interpolation.
    Args: im1: numpy.float 2d-array  input image
          d: numpy.float 2d-array  disparity
    Returns: im1_shifted
    """
    x = np.arange(im1.shape[1])  # column
    y = np.arange(im1.shape[0])  # row 

    # Interpolation function
    interpolator = RegularGridInterpolator((y, x), im1, bounds_error=False, fill_value=0.0)

    coords = []
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            coords.append([i-d[i,j], j])
    coords = np.array(coords)

    im1_shifted = interpolator(coords).reshape(im1.shape)
    
    return im1_shifted

# Example usage for testing
if __name__ == "__main__":
    # Generate a sample image
    im0, im1, gt = load_data('./data/i0.png', './data/i1.png', './data/gt.png')
    im0, im1 = rgb2gray(im0), rgb2gray(im1)

    d = constant_disparity(gt.shape, 6)

    # Shift the image
    im1_shifted = shift_interpolated_disparity(im1, d)

    plt.figure(figsize=(8, 8))
    plt.imshow(im1_shifted, cmap='gray')
    plt.title('Shifted Image', fontsize=15)
    plt.show()
