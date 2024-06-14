import numpy as np
from scipy.interpolate import RegularGridInterpolator

def interpolate_values(a):
    """Interpolate values in array `a` at specific coordinates."""
    x = np.arange(a.shape[1])  # columns
    y = np.arange(a.shape[0])  # rows

    # Interpolation function
    interpolator = RegularGridInterpolator((y, x), a, bounds_error=False, fill_value=0.0)

    # Coordinates for interpolation
    coords1 = [[0, 3.5]]  # Coordinates for a[0, 3.5]
    coords2 = [[1, 0.3]]  # Coordinates for a[1, 0.3]

    # Interpolate at coordinates
    value1 = interpolator(coords1)[0]
    value2 = interpolator(coords2)[0]

    return value1, value2

# Test data
a = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

# Test interpolate_values function
value1, value2 = interpolate_values(a)

print("Interpolated value at a[0, 3.5]:", value1)
print("Interpolated value at a[1, 0.3]:", value2)
