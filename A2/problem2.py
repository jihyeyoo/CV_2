import numpy as np

from scipy import interpolate   # Use this for interpolation
from scipy import signal        # Feel free to use convolutions, if needed
from scipy import optimize      # For gradient-based optimisation
from PIL import Image           # For loading images
import matplotlib.pyplot as plt
# for experiments with different initialisation
from problem1 import random_disparity
from problem1 import constant_disparity


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
            gray[i][j] = np.dot(rgb[i, j, :], weights)
        # execute dot product of r,g,b values and their weights.

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
    i_0 = np.array(Image.open(i0_path), dtype=np.float64) / 255
    i_1 = np.array(Image.open(i1_path), dtype=np.float64) / 255
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
    value = - (x - mu)**2 / (2*sigma**2)
    grad = - (x - mu) / sigma**2
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
    im1_shifted = shift_interpolated_disparity(im1, x)
    llh, llh_grad = log_gaussian(im0 - im1_shifted, mu, sigma)

    sobel = [
        [0, 0, 0],
        [0.5, 0, -0.5],
        [0, 0, 0]
    ]
    im1_x_drv = signal.convolve(im1_shifted, sobel, mode='same')
    grad = llh_grad * im1_x_drv * -1
    value = llh.sum()
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

    llh, llh_grad = stereo_log_likelihood(d, im0, im1, mu, sigma)
    prior, prior_grad = stereo_log_prior(d, mu, sigma)

    log_posterior = llh + alpha * prior
    log_posterior_grad = llh_grad + alpha * prior_grad

    return log_posterior, log_posterior_grad


def optim_method():
    """Simply returns the name (string) of the method 
    accepted by scipy.optimize.minimize, that you found
    to work well.
    This is graded with 1 point unless the choice is arbitrary/poor.
    """
    return 'L-BFGS-B'

def negative_log_posterior(d, im0, im1, mu, sigma, alpha):

    d = d.reshape(im0.shape)
    log_post, log_post_grad = stereo_log_posterior(d, im0, im1, mu, sigma, alpha)

    return -log_post, -log_post_grad.flatten()

def stereo(d0, im0, im1, mu, sigma, alpha, method=optim_method()):
    """Estimating the disparity map

    Args:
        d0: numpy.float 2d-array initialisation of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        d: numpy.float 2d-array estimated value of the disparity
    """

    result = optimize.minimize(
        fun=negative_log_posterior,
        x0=d0.flatten(),
        args=(im0, im1, mu, sigma, alpha),
        method=method,
        jac=True
    )
        
    d = result.x.reshape(d0.shape)
    return d

def downsample(image, new_size):

    resized_img = Image.fromarray(image).resize(new_size)
    return np.array(resized_img)


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

    # fill the pyramid first
    pyr_d = [d0]
    pyr_im0 = [im0]
    pyr_im1 = [im1]
    d = d0
    i0 = im0
    i1 = im1

    for i in range(num_levels - 1):
        h, w = d.shape
        d = downsample(d, (int(d.shape[0]/2), int(d.shape[1]/2)))
        i0 = downsample(i0, (int(i0.shape[0] / 2), int(i0.shape[1] / 2)))
        i1 = downsample(i1, (int(i1.shape[0] / 2), int(i1.shape[1] / 2)))

        pyr_d.append(d)
        pyr_im0.append(i0)
        pyr_im1.append(i1)

    # do the iteration from lowest resolution
    d = pyr_d[num_levels - 1]
    for i in range(num_levels-1, -1, -1):
        d = stereo(d, pyr_im0[i], pyr_im1[i], mu, sigma, alpha)
        pyr_d[i] = d
        # upsample the disparity map and scale by 2, since disparity should be double as image size gets double.
        d = ndimage.zoom(d, 2, order=3) * 2

    return pyr_d

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

    # Initial disparity maps
    d_gt_init = gt
    d_const_init = np.full_like(gt, 8)
    d_rand_init = np.random.uniform(0, 14, gt.shape)

    use_pyramid = False


    if not use_pyramid:
        # Display stereo: Initialized with noise
        d_gt = stereo(d_gt_init, im0, im1, mu, sigma, alpha)
        d_const = stereo(d_const_init, im0, im1, mu, sigma, alpha)
        d_rand = stereo(d_rand_init, im0, im1, mu, sigma, alpha)

    else:
        # Pyramid
        num_levels = 3
        pyramid_gt = coarse2fine(d_gt_init, im0, im1, mu, sigma, alpha, num_levels)
        pyramid_const = coarse2fine(d_const_init, im0, im1, mu, sigma, alpha, num_levels)
        pyramid_rand = coarse2fine(d_rand_init, im0, im1, mu, sigma, alpha, num_levels)

    # Final disparities from the finest level of the pyramid
        d_gt = pyramid_gt[0]
        d_const = pyramid_const[0]
        d_rand = pyramid_rand[0]

    mse_gt = np.mean((gt - d_gt) ** 2)
    mae_gt = np.mean(np.abs(gt - d_gt))
    print(f'MSE (GT Init): {mse_gt}')
    print(f'MAE (GT Init): {mae_gt}')

    mse_const = np.mean((gt - d_const) ** 2)
    mae_const = np.mean(np.abs(gt - d_const))
    print(f'MSE (Const Init): {mse_const}')
    print(f'MAE (Const Init): {mae_const}')

    mse_rand = np.mean((gt - d_rand) ** 2)
    mae_rand = np.mean(np.abs(gt - d_rand))
    print(f'MSE (Rand Init): {mse_rand}')
    print(f'MAE (Rand Init): {mae_rand}')

    # Visualize results
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
