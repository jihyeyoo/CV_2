import numpy as np

from scipy import interpolate   # Use this for interpolation
from scipy import signal        # Feel free to use convolutions, if needed
from scipy import optimize      # For gradient-based optimisation
from PIL import Image           # For loading images

# for experiments with different initialisation
from problem1 import random_disparity
from problem1 import constant_disparity

import matplotlib.pyplot as plt

def rgb2gray(rgb):
    """Converting RGB image to greyscale.
    The same as in Assignment 1 (no graded here).

    Args:
        rgb: numpy array of shape (H, W, 3)

    Returns:
        gray: numpy array of shape (H, W)

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
    """Loading the data.
    The same as in Assignment 1 (not graded here).

    Args:
        i0_path: path to the first image
        i1_path: path to the second image
        gt_path: path to the disparity image
    
    Returns:
        i_0: numpy array of shape (H, W)
        i_1: numpy array of shape (H, W)
        g_t: numpy array of shape (H, W)
    """
    # read the image data and divide the i_0 and i_1 values by 255 to normalize to [0,1]
    i_0 = np.array(Image.open(i0_path), dtype=np.float64)/255.0
    i_1 = np.array(Image.open(i1_path), dtype=np.float64)/255.0
    g_t = np.array(Image.open(gt_path), dtype=np.float64)

    return i_0, i_1, g_t

def log_gaussian(x,  mu, sigma):
    """Calcuate the value and the gradient w.r.t. x of the Gaussian log-density

    Args:
        x: numpy.float 2d-array
        mu and sigma: scalar parameters of the Gaussian

    Returns:
        value: value of the log-density
        grad: gradient of the log-density w.r.t. x
    """
    # return the value and the gradient
    value = -((x - mu) ** 2) / (2 * sigma ** 2)
    grad = -(x - mu) / (sigma ** 2)
    return value, grad

def stereo_log_prior(x, mu, sigma):
    """Evaluate gradient of pairwise MRF log prior with Gaussian distribution

    Args:
        x: numpy.float 2d-array (disparity)

    Returns:
        value: value of the log-prior
        grad: gradient of the log-prior w.r.t. x
    """
    dh = x[:, 1:] - x[:, :-1]
    dv = x[1:, :] - x[:-1, :]

    fh, gradient_h = log_gaussian(dh, mu, sigma)
    fv, gradient_v = log_gaussian(dv, mu, sigma)

    value = np.sum(fh) + np.sum(fv)
    grad = np.zeros(x.shape)

    grad[:, :-1] += gradient_h
    grad[:, 1:] += gradient_h
    grad[:-1, :] += gradient_v
    grad[1:, :] += gradient_v

    return  value, grad

def shift_interpolated_disparity(im1, d):
    """Shift image im1 by the disparity value d.
    Since disparity can now be continuous, use interpolation.

    Args:
        im1: numpy.float 2d-array  input image
        d: numpy.float 2d-array  disparity

    Returns:
        im1_shifted: Shifted version of im1 by the disparity value.
    """
    H, W = im1.shape
    coords = np.array(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), dtype=np.float64)
    coords[1] -= d
    interpolator = interpolate.RegularGridInterpolator((np.arange(H), np.arange(W)), im1, bounds_error=False, fill_value=0)
    shifted_im1 = interpolator(coords.transpose(1, 2, 0))

    return shifted_im1

def stereo_log_likelihood(x, im0, im1, mu, sigma):
    """Evaluate gradient of the log likelihood.

    Args:
        x: numpy.float 2d-array of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        value: value of the log-likelihood
        grad: gradient of the log-likelihood w.r.t. x

    Hint: Make use of shift_interpolated_disparity and log_gaussian
    """
    shifted_im1 = shift_interpolated_disparity(im1, x)
    log_likelihood, grad_log_likelihood = log_gaussian(im0 - shifted_im1, mu, sigma)

    sobel = [
        [0, 0, 0],
        [0.5, 0, -0.5],
        [0, 0, 0]
    ]
    im1_x_drv = signal.convolve(shifted_im1, sobel, mode='same')
    grad = grad_log_likelihood * im1_x_drv * -1
    value = log_likelihood.sum()

    return value, grad


def stereo_log_posterior(d, im0, im1, mu, sigma, alpha):
    """Computes the value and the gradient of the log-posterior

    Args:
        d: numpy.float 2d-array of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        value: value of the log-posterior
        grad: gradient of the log-posterior w.r.t. x
    """
    log_prior, grad_prior = stereo_log_prior(d, mu, sigma)
    log_likelihood, grad_likelihood = stereo_log_likelihood(d, im0, im1, mu, sigma)
    log_posterior = log_likelihood + alpha * log_prior
    log_posterior_grad = grad_likelihood + alpha * grad_prior
    
    return log_posterior, log_posterior_grad


def optim_method():
    """Simply returns the name (string) of the method 
    accepted by scipy.optimize.minimize, that you found
    to work well.
    This is graded with 1 point unless the choice is arbitrary/poor.
    """
    return 'L-BFGS-B'

def stereo(d0, im0, im1, mu, sigma, alpha, method=optim_method()):
    """Estimating the disparity map

    Args:
        d0: numpy.float 2d-array initialisation of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        d: numpy.float 2d-array estimated value of the disparity
    """
    def objective(d):
        d = d.reshape(d0.shape)
        log_posterior, grad = stereo_log_posterior(d, im0, im1, mu, sigma, alpha)
        return -log_posterior, -grad.flatten()
    
    result = optimize.minimize(objective, d0.flatten(), jac=True, method=method)
    d = result.x.reshape(d0.shape)

    return d

def coarse2fine(d0, im0, im1, mu, sigma, alpha, num_levels):
    """Coarse-to-fine estimation strategy. Basic idea:
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
    """
    from scipy import ndimage

    def downsample2(img):

        # convolve image and gauss filter
        # smooth_img = signal.convolve(img, f, mode='same')
        smooth_img = ndimage.gaussian_filter(img, sigma = 1.5, mode = 'constant', cval = 0)
        # take every other row/column
        downsampled = np.zeros((int(img.shape[0] / 2), int(img.shape[1] / 2)))
        h, w = downsampled.shape

        for i in range(h):
            for j in range(w):
                downsampled[i, j] = smooth_img[i * 2, j * 2]

        return downsampled

    # fill the pyramid first
    pyr_d, pyr_im0, pyr_im1 = [d0], [im0], [im1]
    d, i0, i1 = d0, im0, im1

    for i in range(num_levels - 1):
        # d_f = ndimage.gaussian_filter(d, 1.5, mode='reflect')
        # i0_f = ndimage.gaussian_filter(i0, 1.5, mode='reflect')
        # i1_f = ndimage.gaussian_filter(i1, 1.5, mode='reflect')

        d, i0, i1 = downsample2(d), downsample2(i0), downsample2(i1)

        pyr_d.append(d)
        pyr_im0.append(i0)
        pyr_im1.append(i1)

    # do the iteration
    d = pyr_d[num_levels - 1]
    for i in range(num_levels-1, -1, -1):
        d = stereo(d, pyr_im0[i], pyr_im1[i], mu, sigma, alpha)
        pyr_d[i] = d
        # upscale the disparity map
        if i > 0:
            d = ndimage.zoom(d, 2, order=3)
            d = d[:pyr_d[i - 1].shape[0], :pyr_d[i - 1].shape[1]]
        
    return pyr_d

# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():

    # these are the same functions from Assignment 1
    # (no graded in this assignment)
    im0, im1, gt = load_data('./data/i0.png', './data/i1.png', './data/gt.png')
    im0, im1 = rgb2gray(im0), rgb2gray(im1)

    mu = 0.0
    sigma = 1.0

    # experiment with other values of alpha
    alpha = 1.0

    d_gt_init = gt
    d_const_init =  constant_disparity(gt.shape, 8)
    d_rand_init = random_disparity(gt.shape)

    disparity_gt = stereo(d_gt_init, im0, im1, mu, sigma, alpha)
    disparity_const = stereo(d_const_init, im0, im1, mu, sigma, alpha)
    disparity_rand = stereo(d_rand_init, im0, im1, mu, sigma, alpha)

    num_levels = 3
    pyramid_gt = coarse2fine(d_gt_init, im0, im1, mu, sigma, alpha, num_levels)
    pyramid_const = coarse2fine(d_const_init, im0, im1, mu, sigma, alpha, num_levels)
    pyramid_rand = coarse2fine(d_rand_init, im0, im1, mu, sigma, alpha, num_levels)

    d_gt = pyramid_gt[0]
    d_const = pyramid_const[0]
    d_rand = pyramid_rand[0]

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    axs[0, 0].imshow(gt, cmap='gray')
    axs[0, 0].set_title('Ground Truth Disparity')

    axs[0, 1].imshow(d_gt, cmap='gray')
    axs[0, 1].set_title('Estimated Disparity (GT Init)')

    axs[0, 2].imshow(np.abs(gt - d_gt), cmap='gray')
    axs[0, 2].set_title('Difference (GT)')

    axs[1, 0].imshow(gt, cmap='gray')
    axs[1, 0].set_title('Ground Truth Disparity')

    axs[1, 1].imshow(d_const, cmap='gray')
    axs[1, 1].set_title('Estimated Disparity (Const Init)')

    axs[1, 2].imshow(np.abs(gt - d_const), cmap='gray')
    axs[1, 2].set_title('Difference (Const)')

    axs[2, 0].imshow(gt, cmap='gray')
    axs[2, 0].set_title('Ground Truth Disparity')

    axs[2, 1].imshow(d_rand, cmap='gray')
    axs[2, 1].set_title('Estimated Disparity (Rand Init)')

    axs[2, 2].imshow(np.abs(gt - d_rand), cmap='gray')
    axs[2, 2].set_title('Difference (Rand)')

    plt.show()

if __name__ == "__main__":
    main()
