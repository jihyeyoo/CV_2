import numpy as np

from scipy import interpolate   # Use this for interpolation
from scipy import signal        # Feel free to use convolutions, if needed
from scipy import optimize      # For gradient-based optimisation
from PIL import Image           # For loading images
from scipy.interpolate import interp2d
from scipy.optimize import minimize

# for experiments with different initialisation
from problem1 import random_disparity
from problem1 import constant_disparity

def rgb2gray(rgb):
    """Converting RGB image to greyscale.
    Args: rgb: numpy array of shape (H, W, 3)
    Returns: gray: numpy array of shape (H, W)
    """
    height, width = rgb.shape[:2]
    gray = np.zeros((height, width), dtype=np.float64)
    weights = [0.2126, 0.7152, 0.0722]
    for i in range(height):
         for j in range(width):
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
    i_0 = np.array(Image.open(i0_path), dtype=np.float64)/255.0
    i_1 = np.array(Image.open(i1_path), dtype=np.float64)/255.0
    g_t = np.array(Image.open(gt_path), dtype=np.float64)
    return i_0, i_1, g_t

def log_gaussian(x,  mu, sigma):
    """Calcuate value & gradient w.r.t. x of Gaussian log-density
    Args: x : numpy.float 2d-array
          mu & sigma : scalar parameters of Gaussian
    Returns: value: value of the log-density
             grad: gradient of the log-density w.r.t. x
    """
    value = -((x - mu) *(x - mu)) / (2 * sigma *sigma)
    grad = -(x - mu) / (sigma *sigma)
    return value, grad

def stereo_log_prior(x, mu, sigma):
    """Evaluate gradient of pairwise MRF log prior with Gaussian distribution
    Args: x: numpy.float 2d-array (disparity)
    Returns: value: value of the log-prior
             grad: gradient of the log-prior w.r.t. x
    """
    # log prior
    value = 0.0
    # array to store grad
    grad = np.zeros(x.shape)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if i > 0: # except first row 
                diff=x[i, j] - x[i-1, j]
                v, g = log_gaussian(diff, mu, sigma)
                value += v
                grad[i, j] += g
                grad[i-1, j] -= g # for top pixel 
            if j > 0: # except first column
                diff=x[i, j] - x[i, j-1]
                v, g = log_gaussian(diff, mu, sigma)
                value += v
                grad[i, j] += g
                grad[i, j-1] -= g # for left pixel
    
    return  value, grad

def shift_interpolated_disparity(im1, d):
    """Shift im1 by d.
    Since disparity can now be continuous, use interpolation.
    Args: im1: numpy.float 2d-array  input image
          d: numpy.float 2d-array  disparity
    Returns:  im1_shifted
    """
    x = np.arange(im1.shape[1])  #column
    y = np.arange(im1.shape[0])  #row
    x_new = x - d  # shifted columns based on disparity
    
    # Create interpolation function
    interpolator = interp2d(x, y, im1, kind='linear', fill_value=0.0)
    
    # Perform interpolation
    im1_shifted = interpolator(x_new, y)
    return im1_shifted

def stereo_log_likelihood(x, im0, im1, mu, sigma):
    """Evaluate gradient of log likelihood.=> logp(im0∣im1,d)
    Args: x: numpy.float 2d-array of the disparity
          im0: numpy.float 2d-array of image #0
          im1: numpy.float 2d-array of image #1
    Returns: value: value of the log-likelihood
             grad: gradient of the log-likelihood w.r.t. x
    Hint: Use shift_interpolated_disparity and log_gaussian
    """
    shifted_im1 = shift_interpolated_disparity(im1, x)
    diff = im0 - shifted_im1
    value, grad = log_gaussian(diff, mu, sigma)
    return value, grad

def stereo_log_posterior(d, im0, im1, mu, sigma, alpha):
    """Computes value & gradient of log-posterior=>logp(d∣im0,im1)∝logp(im0∣im1,d)+αlogp(d)
    Args: d: numpy.float 2d-array of the disparity
          im0: numpy.float 2d-array of image #0
          im1: numpy.float 2d-array of image #1
    Returns:
        value: value of the log-posterior
        grad: gradient of the log-posterior w.r.t. x
    """
    log_prior_value, log_prior_grad = stereo_log_prior(d, mu, sigma)
    log_likelihood_value, log_likelihood_grad = stereo_log_likelihood(d, im0, im1, mu, sigma)
    
    log_posterior = log_likelihood_value + alpha * log_prior_value
    log_posterior_grad = log_likelihood_grad + alpha * log_prior_grad

    return log_posterior, log_posterior_grad

def optim_method():
    """Simply returns the name (string) of the method 
    accepted by scipy.optimize.minimize, that you found
    to work well.
    This is graded with 1 point unless the choice is arbitrary/poor.
    """
    return 'L-BFGS-B'

def stereo(d0, im0, im1, mu, sigma, alpha, method=optim_method()):
    """Estimate disparity map
    Args: d0: numpy.float 2d-array initialisation of the disparity
          im0: numpy.float 2d-array of image #0
          im1: numpy.float 2d-array of image #1
    Returns: d: numpy.float 2d-array estimated value of the disparity
    """
    # Define the objective function for minimize
    def objective_function(d):
        return stereo_log_posterior(d, im0, im1, mu, sigma, alpha)

    # Initial guess of the disparity map
    initial_guess = d0.flatten()

    # Minimize the objective function using the specified method
    result = minimize(objective_function, initial_guess, method=method, jac=True)

    # Reshape the optimized result back to 2D array
    d_optimized = result.x.reshape(d0.shape)

    return d_optimized
"""
def coarse2fine(d0, im0, im1, mu, sigma, alpha, num_levels):
    Coarse-to-fine estimation strategy. Basic idea:
        1. create an image pyramid (of size num_levels)
        2. starting with the lowest resolution, estimate disparity
        3. proceed to the next resolution using the estimated 
        disparity from the previous level as initialisation

    Args:
        d0: numpy.float 2d-array initialisation of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        pyramid: a list of size num_levels containing the estimated
        disparities at each level (from finest to coarsest)
        Sanity check: pyramid[0] contains the finest level (highest resolution)
                      pyramid[-1] contains the coarsest level
    
    pyramid = []
    current_d = d0
    
    for level in range(num_levels):
        # Perform stereo estimation at current resolution
        current_d = stereo(current_d, im0, im1, mu, sigma, alpha)
        pyramid.append(current_d)
        
        # Downsample images for the next level
        im0 = downsample(im0)
        im1 = downsample(im1)
        
        # Upscale current disparity map for initialization at the next level
        current_d = upscale(current_d)
    return pyramid
    """

# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():

    # these are the same functions from Assignment 1
    im0, im1, gt = load_data('./data/i0.png', './data/i1.png', './data/gt.png')
    im0, im1 = rgb2gray(im0), rgb2gray(im1)

    mu = 0.0
    sigma = 1.0

    # experiment with other values of alpha
    alpha = 1.0

    # initial disparity map
    # experiment with constant/random values
    d0 = gt
    #d0 = random_disparity(gt.shape)
    #d0 = constant_disparity(gt.shape, 6)

    # Display stereo: Initialized with noise
    disparity = stereo(d0, im0, im1, mu, sigma, alpha)

    # Pyramid
    # num_levels = 3
    # pyramid = coarse2fine(d0, im0, im1, mu, sigma, num_levels)

if __name__ == "__main__":
    main()
