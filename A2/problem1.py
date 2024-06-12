from PIL import Image
import numpy as np
np.random.seed(seed=2024)

def load_data(gt_path):
    g_t = np.array(Image.open(gt_path), dtype=np.float64)
    return g_t

def random_disparity(disparity_size):
    """
    Args: disparity_size: tuple containg height and width (H, W)
    Returns: disparity_map: numpy array of shape (H, W)
    """
    disparity_map=np.random.randint(0,16,size=disparity_size).astype(np.float64)
    return disparity_map

def constant_disparity(disparity_size, a):
    """
    Args: disparity_size: tuple containg height and width (H, W)
          a: value to initialize with
    Returns: disparity_map: numpy array of shape (H, W)
    """
    disparity_map = np.full(disparity_size, a, dtype=np.float64)
    return disparity_map

def log_gaussian(x, mu, sigma):
    """Compute the log gaussian of x.
    Args: x: numpy array of shape (H, W) (np.float64)
          mu: float
          sigma: float
    Returns: result: numpy array of shape (H, W) (np.float64)
    """
    result= -((x - mu) *(x - mu)) / (2 * sigma * sigma)
    return result


def mrf_log_prior(x, mu, sigma):
    """Compute the log of the unnormalized MRF prior density.
    Args: x: numpy array of shape (H, W) (np.float64)
        mu: float
        sigma: float
    Returns: logp: float
    """
    H, W = x.shape
    logp = 0.0

    # W
    for i in range(H):
        for j in range(W - 1):
            logp += log_gaussian(x[i, j + 1] - x[i, j], mu, sigma)

    # H
    for i in range(H - 1):
        for j in range(W):
            logp += log_gaussian(x[i + 1, j] - x[i, j], mu, sigma)

    logp = np.sum(logp)

    return logp

# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code

def main():

    gt = load_data('./data/gt.png')

    # Display log prior of GT disparity map
    logp = mrf_log_prior(gt, mu=0, sigma=1.1)
    print("Log Prior of GT disparity map:", logp)

    # Display log prior of random disparity ma
    random_disp = random_disparity(gt.shape)
    logp = mrf_log_prior(random_disp, mu=0, sigma=1.1)
    print("Log-prior of noisy disparity map:",logp)

    # Display log prior of constant disparity map
    constant_disp = constant_disparity(gt.shape, 6)
    logp = mrf_log_prior(constant_disp, mu=0, sigma=1.1)
    print("Log-prior of constant disparity map:", logp)

if __name__ == "__main__":
	main()

"""
1. Compare values of log-prior density for three disparity maps
Result: Log Prior of GT disparity map: -50685.12396694026
        Log-prior of noisy disparity map: -3896875.619835017
        Log-prior of constant disparity map: 0.0

1-1. GT Disparity Map
has the lowest (least negative) log-prior density among the three maps. 
This indicates that the MRF model considers the GT disparity map to be relatively consistent and plausible. 
The low negative value suggests that the GT disparity map has small differences between neighboring pixels, which aligns well with the Gaussian potentials used in the model.

1-2. Noisy disparity map
Significantly higher (more negative) log-prior density.
MRF model finds the noisy disparity map to be highly inconsistent and implausible. 
The large negative value indicates substantial differences between neighboring pixels, which the Gaussian potentials penalize heavily, reflecting the noise and inconsistency in the disparity values.

1-3. Constant Disparity Map
There are no differences between neighboring pixels, leading to no penalization by the Gaussian potentials. 
MRF model considers this map to be perfectly consistent.

2. Increasing σ of the Gaussian potentials
Increasing σ means that it penalizes the disparity less.

3. Reducing the range of the noise map, e.g. to [0,4]
Reducing the range of noise means that the possible disparity values that neighboring pixels can take become more constrained.
Smaller differences between neighboring pixels result in smaller penalties imposed by the Gaussian potentials in the MRF model. 
Penalties are calculated based on the squared differences between neighboring pixels divided by the variance (σ²) of the Gaussian potentials. 
With smaller disparities due to reduced noise range, these penalties diminish, leading to a less negative log-prior density.
MRF model perceives disparity map with reduced noise range as more consistent and plausible because neighboring disparities are closer to each other. 
This perception aligns with model's expectation that neighboring pixels in a disparity map should have similar values, reflecting smooth transitions in depth across the image.
"""

