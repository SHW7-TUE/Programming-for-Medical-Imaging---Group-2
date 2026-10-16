
import os
import nibabel as nib
import numpy as np
from pydicom import dcmread
from pydicom.pixels import apply_rescale
from scipy.ndimage import gaussian_filter
from pmi_viewer import view

__all__ = ["dicom_to_nifti"]

def _get_z_position(dicom_image) -> float:
    """Helper function that returns the z-coordinate of the Image Position Patient DICOM attribute."""
    return dicom_image.ImagePositionPatient[2]


def _get_spacing(dicom_list) -> tuple[float, ...]:
    """Return DICOM rows, columns, and optional slice spacing in millimetres for multiple DICOM files."""

    n = len(dicom_list)                     #The number of slices in the list
    dx, dy = dicom_list[0].PixelSpacing     #Read the X and Y pixel spacing from the first slice.

    if n > 1:
        z_first = np.array(dicom_list[0].ImagePositionPatient)
        z_last = np.array(dicom_list[-1].ImagePositionPatient)
        dz = np.linalg.norm(z_last - z_first) / (n - 1)
        #Spacing between slices is calculated as distance from the first to the last slice centre, divided by the number of gaps (n slices = n-1 gaps)"""
        return [dx, dy, dz]
    else:
        return [dx, dy]                     #Since a single slice has no neighbouring slice, only 2D spacing exists.


def read_dicom(file_path) -> tuple[np.ndarray, tuple[float, ...]]:
    """Read one DICOM image or a directory of images."""

    dicom_images = []

    if os.path.isfile(file_path):                   #Single DICOM file
        dicom_images.append(dcmread(file_path))
    elif os.path.isdir(file_path):                  #Directory with multiple DICOM files
        for file_name in os.listdir(file_path):
            full_name = os.path.join(file_path, file_name)
            if os.path.isfile(full_name):
                dicom_images.append(dcmread(full_name))
    else: raise ValueError("DICOM path does not exist")

    series_uids = set()
    for image in dicom_images:
        series_uids.add(image.SeriesInstanceUID)
    if len(series_uids) > 1:
    #A set only stores unique values, so if there are more UIDs in the set, the slices belong to different scans."""
        raise ValueError("DICOM files have different Series Instance UIDs.")

    if len(dicom_images) > 1:
        dicom_images.sort(key=_get_z_position) #Slices are sorted in anatomical order along z

    pixel_values = []
    for image in dicom_images:
        pixel_values.append(apply_rescale(image.pixel_array, image)) #Convert the pixel values to real values

    if len(pixel_values) == 1:
        data = pixel_values[0]
    else:
        data = pixel_values

    return data, _get_spacing(dicom_images)


def read_nifti(file_path):
    """
    Reads a Nifti file and returns the data and spacing.
    """

    nifti_image = nib.load(file_path)  # Read the image.
    transposed_nifti = np.transpose(nifti_image.get_fdata())
    mirrored_nifti = np.flip(transposed_nifti, axis = (-1, -2))
    #Nifti uses (x, y, z) axis, while DICOM uses (z, y, x), so we transpose and mirror the NIFTI to be consistent with DICOM.

    spacing = nifti_image.header.get_zooms()
    return mirrored_nifti, spacing

def write_nifti(pixel_data, spacing, file_path) -> None:
    pixel_data = np.asarray(pixel_data, dtype=float)            #Raises ValueError if not a number.

    if pixel_data.ndim not in (2, 3):
        raise ValueError("Pixel data must be 2D or 3D.")
    if len(spacing) != pixel_data.ndim:
        raise ValueError("Spacing must have one value per dimension of the pixel data.")
    if any(s <= 0 for s in spacing):                            #A voxel cannot have zero or negative value.
        raise ValueError("Spacing values must be positive.")

    mirrored = np.flip(pixel_data, axis= (-1, -2))                #Undo mirroring and transposing done in read_nifti()
    transposed = np.transpose(mirrored)

    affine = np.eye(4)
    affine[0, 0] = spacing[0]                                   #Set the diagonal values of the affine matrix to the pixel spacing values.
    affine[1, 1] = spacing[1]
    if len(spacing) == 3:
        affine[2, 2] = spacing[2]

    nib.save(nib.Nifti1Image(transposed, affine), file_path)

def dicom_to_nifti(dicom_path, nifti_path) -> None:
    """Convert a DICOM file or folder of DICOM files to a NIfTI file.

    Stored DICOM pixels are converted to physical values.
    A directory must contain one series of dicom files, the slices are automatically sorted by patient z-position.
    """
    data, spacing = read_dicom(dicom_path)
    write_nifti(data, spacing, nifti_path)

def _masked_gaussian_filter(pixel_data, mask, sigma) -> np.ndarray:
    """
    Gaussian smoothing restricted to a mask
    Converts the image mask to an array and float and then multiplies them
    Computes M * (G(M * I, sigma) / G(M, sigma))
    """
    # Check if the dimensions are the same
    if pixel_data.shape != mask.shape:  # Checks if the shapes of the dataset and the mask are consistent, otherwise raises an error
        raise ValueError("Image and mask are inconsistent!")

    #Check if the mask is a boolean mask
    if mask.dtype != np.bool_:
        raise ValueError("The mask must be a boolean mask!")

    pixel_data = pixel_data.astype(np.float32)  # Convert it to a float
    mask = mask.astype(np.float32)

    #Apply a gaussian filter
    smoothed = gaussian_filter(input=(pixel_data * mask), sigma=sigma)
    masked = gaussian_filter(input=mask, sigma=sigma)

    out = mask * (smoothed / masked)  # Calculates M * (G(M * I, sigma) / G(M, sigma))
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)  # Switches nan and inf for zeroes
    return out

#import matplotlib.pyplot as plt

def _test (file_path):
    #Test the read_dicom function
    data, spacing = read_dicom(file_path)
    view(data)

    #Test the Gaussian filter function
    data = np.asarray(data, dtype=np.float32)
    mask = data > -300
    data.ndim
    sigma = (2.0, ) * data.ndim #Give a standard deviation for each dimension of the image
    filtered_image = _masked_gaussian_filter(data, mask, sigma=sigma)
    view(filtered_image)

    #Test the write_nifti and read_nifti function
    nifti_path = "test_nifti.nii.gz"
    write_nifti(data, spacing, nifti_path)
    nifti_image, nifti_spacing = read_nifti(nifti_path)
    view(nifti_image)

    #Test the dicom_to_nifti function
    dicom_to_nifti(file_path, nifti_path)
    nifti_image, nifti_spacing = read_nifti(nifti_path)
    view(nifti_image)

# actually run the stuff
if __name__ == "__main__":
    print("Please enter the path to a DICOM image / DICOM images.")
    _test(input().strip())