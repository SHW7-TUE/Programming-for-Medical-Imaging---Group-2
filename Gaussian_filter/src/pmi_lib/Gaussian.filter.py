import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy import ndimage as ndi

def body_mask(slice_hu, threshold=-300):
    mask = slice_hu > threshold
    mask = ndi.binary_opening(mask, iterations=2)    # removes thin structures such as the table
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, labels, range(1, n + 1))
    mask = labels == (np.argmax(sizes) + 1)          # keep the largest connected component (the body)
    return ndi.binary_fill_holes(mask)               # fills the lungs, so the mask is the full body
mask = body_mask("slice_hu")
def pad_to_match(img, ref):
    """
    Zero-pad img to the same, W as the reference image (The image where the mask should be applied on)
    """
    th,tw = ref.shape[:2]
    h,w = img.shape[:2]
    dh, dw = th-h, tw-w
    if dh <0 or dw < 0:
         raise ValueError(f"img {img.shape[:2]} is larger than {ref.shape[:2]}")
    top, left = dh//2, dw//2
    pad = [(top,dh-top), (left,dw-left)] + [(0,0)]* (img.ndim -2)
    return np.pad(img, pad, mode='constant', constant_values=0)
def masked_gaussian_filter(pixel_data, mask, sigma):
    """
    Gaussian smoothing restricted to a mask

    Converts the image mask to an array and float and then multiplies them

    Computes M * (G(M * I, sigma) / G(M, sigma))

    """
    
    pixel_data = pixel_data.astype(np.float32) # Convert it to a float

    boolean_mask = boolean_mask.astype(np.float32) 
    #Check if the dimensions are the same
    if pixel_data.shape != boolean_mask.shape: #Checks if the shapes of the dataset and the mask are consistent, otherwise raises an error
           raise ValueError("Image and mask Inconsistent!")
    if pixel_data.shape!= boolean_mask.shape:
        boolean_mask = pad_to_match(boolean_mask, pixel_data) # If the mask and the image are not the same size, zero padding is applied to fill up the empty space
    masked_data = pixel_data * boolean_mask
    
    #Convertes the arrays to floats
    masked_data = masked_data.astype(np.float32)
    masked = boolean_mask.astype(np.float32)

    # Applies a gaussian filter
    smoothed = gaussian_filter(input = masked_data, sigma = sigma)
    masked = gaussian_filter(input=boolean_mask, sigma = sigma)
    
    out = (smoothed / masked) * boolean_mask # Calculates M * (G(M * I, sigma) / G(M, sigma))
    out = np.nan_to_num(out, nan=0.0, inf = 0.0) # Switches nan and inf for zeroes
    
    return out

