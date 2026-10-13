import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
sigma = 1
def masked_gaussian_filter(pixel_data, mask, sigma):
    """
    Gaussian smoothing restricted to a mask

    Converts the image mask to an array and float and then multiplies them

    Computes M * (G(M * I, sigma) / G(M, sigma))

    """
    pixel_data = np.array(pixel_data)
    pixel_data = pixel_data.astype(np.float32) # Convert it to a float
    boolean_mask = np.array(boolean_mask)
    boolean_mask = mask.astype(bool).boolean_mask = mask.astype(np.float32) 
    #Check if the dimensions are the same
    if pixel_data.shape != boolean_mask.shape: #Checks if the shapes of the dataset and the mask are consistent, otherwise raises an error
           raise ValueError("Image and mask Inconsistent!")
    masked_data = pixel_data * boolean_mask
    

    masked_data = masked_data.astype(np.float32)
    masked = boolean_mask.astype(np.float32)
    smoothed = gaussian_filter(input = masked_data, sigma = sigma)
    masked = gaussian_filter(input=boolean_mask, sigma = sigma)
    
    out = (smoothed / masked) * boolean_mask 
    out = np.nan_to_num(out, nan=0.0, inf = 0.0)
    
    return out

# def test(file_path):

