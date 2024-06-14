import numpy as np
from scipy import interpolate  

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

# Test
a = np.array([
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9]
], dtype=np.float64)

interpolator = interpolate.RegularGridInterpolator((np.arange(a.shape[0]), np.arange(a.shape[1])), a, bounds_error=False, fill_value=0)

value_at_0_3_5 = interpolator([0, 3.5])
value_at_1_0_3 = interpolator([1, 0.3])

print(f"Interpolated value at [0, 3.5]: {value_at_0_3_5}")
print(f"Interpolated value at [1, 0.3]: {value_at_1_0_3}")