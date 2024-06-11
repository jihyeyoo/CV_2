from PIL import Image
import numpy as np
np.random.seed(seed=2024)

def load_data(gt_path):
    """Loading the data.
    The same as loading the disparity from Assignment 1 
    (not graded here).

    Args:
        gt_path: path to the disparity image
    
    Returns:
        g_t: numpy array of shape (H, W)
    """
    return g_t

def random_disparity(disparity_size):
    """Compute a random disparity map.

    Args:
        disparity_size: tuple containg height and width (H, W)

    Returns:
        disparity_map: numpy array of shape (H, W)

    """
    return disparity_map

def constant_disparity(disparity_size, a):
    """Compute a constant disparity map.

    Args:
        disparity_size: tuple containg height and width (H, W)
        a: the value to initialize with

    Returns:
        disparity_map: numpy array of shape (H, W)

    """
    return disparity_map


def log_gaussian(x, mu, sigma):
    """Compute the log gaussian of x.

    Args:
        x: numpy array of shape (H, W) (np.float64)
        mu: float
        sigma: float

    Returns:
        result: numpy array of shape (H, W) (np.float64)

    """
    return result


def mrf_log_prior(x, mu, sigma):
    """Compute the log of the unnormalized MRF prior density.

    Args:
        x: numpy array of shape (H, W) (np.float64)
        mu: float
        sigma: float

    Returns:
        logp: float

    """
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

