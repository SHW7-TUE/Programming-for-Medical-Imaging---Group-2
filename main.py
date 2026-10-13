# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import matplotlib.pyplot as plt



""""
pixel data

boolean mask

sd from gaussian kernel

 M×(G(M×I, σ) / G(M, σ)), where I is the image

G(X, sigma) = gaussian smoothing filter
X = image

Multiplication and devision is pixelwise

May use numpu.nan_to_num()

Add test() function

API (module,function, argument names), add docstrings :), hide helper functions from the public interface)
""""
from scipy.ndimage import gaussian_filter
import numpy as np
sigma = 1
def masked_gaussian_filter(pixel_data, mask, sigma):
        """"
    Gaussian smoothing restricted to a mask

    Computes M * (G(M * I, sigma) / G(M, sigma))

    """"
   
    pixel_data = pixel_data.astype(np.float32) # Convert it to a float

    boolean_mask = mask.astype(bool).boolean_mask = mask.astype(np.float32) #TODO check if I need to convert to a boolean
    #Check if the dimensions are the same
    if pixel_data.shape != boolean_mask.shape: #Checks if the shapes of the dataset and the mask are consistent, otherwise raises an error
           raise ValueError("Image and mask Inconsistent!")
    masked_data = pixel_data * boolean_mask
    

    masked_data = masked_data.astype(np.float32)
    masked = boolean_mask.astype(np.float32)
    smoothed = gaussian_filter(input = masked_data, sigma=2)
    masked = gaussian_filter(input=boolean_mask, sigma=2)
    
    out = (smoothed / masked) * boolean_mask 
    out = np.nan_to_num(out, nan=0.0)
    
    return out

from .filtering import masked_gaussian_filter
__all__ = ["masked_gaussian_filter"]

import numpy as np
from pmi_lib import masked_gaussian_filter

img = np.random.rand(64, 64)
mask = np.zeros((64, 64), dtype=bool)
mask[16:48, 16:48] = True

out = masked_gaussian_filter(img, mask, (2.0, 2.0))
print(out.dtype, out.shape)   # float32 (64, 64)
assert np.all(out[~mask] == 0)

# These should raise ValueError
# masked_gaussian_filter(img, mask[:-1], (2.0, 2.0))   # shape mismatch
# masked_gaussian_filter(img, mask, (2.0,))            # wrong sigma length