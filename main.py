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
    boolean_mask = np.array(mask)
    boolean_mask = boolean_mask.astype(np.float32) 
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

def test(pixel_data, sigma = 3):
    mask = pixel_data > pixel_data.mean()


    # Numeric checks
    assert masked.dtype == np.float32
    assert np.all(masked[~mask] == 0)
    assert not np.isnan(masked).any()

    # Constant image should stay constant inside the image
    const = masked_gaussian_filter(np.full_like(img, 5.0), mask, sigma)
    np.testing.assert_allclose(const[mask], plain, rtol=5.0, atol=1e-5)
      
    # Mapping

    fig,ax = plt.subplots(1, 5, figsize=(18,4))
    panels = [(img, "Image I"), (mask, "Mask M"),
              (plain, "Plain G(I)"), (masked, "Masked filter"),
              ((plain - masked) * mask, "Plain - masked (inside M)")]
    for a, (data, title) in zip(ax, panels):
        a.imshow(data, cmap="gray")
        a.set_title(title)
        a.axis("off")
    plt.show()

test("your_image.png")