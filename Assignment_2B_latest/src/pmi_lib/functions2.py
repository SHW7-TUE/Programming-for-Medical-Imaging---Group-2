from numbers import Real

import os
import nibabel as nib
import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.pixels import apply_rescale
from scipy.ndimage import gaussian_filter
from pmi_viewer import view
from typing import Any


__all__ = ["dicom_to_nifti"]


class Image:
    """Class to store image data and spacings in millimetres for each axis."""

    def __init__(self, data: np.ndarray, spacing: tuple[float, ...] | None = None, ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("Image data must be a NumPy array.")
        if spacing is None:
            spacing = (1.0,) * data.ndim
        if not isinstance(spacing, tuple):
            raise TypeError("Spacing must be a tuple of numbers.")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in spacing):
            raise TypeError("Spacing values must be real numbers.")
        if len(spacing) != data.ndim:
            raise ValueError("Spacing must have one value per image dimension.")

        spacing = tuple(float(value) for value in spacing)
        if any(not np.isfinite(value) or value <= 0 for value in spacing):
            raise ValueError("Spacing values must be finite and positive.")

        self._data = data
        self.spacing = spacing

    def __str__(self) -> str:
        return (f"A {self._data.dtype} image with shape {self._data.shape} " f"and a spacing of {self.spacing} mm.")

    def get_pixel_area(self) -> float:
        """Return the area of a pixel in square millimetres."""
        if self._data.ndim < 2:
            raise ValueError("Pixel area requires at least two dimensions.")
        return self.spacing[0] * self.spacing[1]

    def get_voxel_volume(self) -> float:
        """Return the volume of a voxel in cubic millimetres."""
        if self._data.ndim < 3:
            raise ValueError("Voxel volume requires at least three dimensions.")
        return self.spacing[0] * self.spacing[1] * self.spacing[2]

    def __len__(self) -> int:
        return len(self._data)     # Allows len(image) to work. It returns the size of the outer (first) dimension.

    def __array__(
            self,
            dtype: np.typing.DTypeLike | None = None,
            *,
            copy: bool | None = None,
    ) -> np.ndarray:
                            # Lets NumPy convert an image to an array. It casts the pixel data to the dtype specified.
        return self._data.astype(dtype, copy=copy)

    def __getitem__(self, key: Any) -> np.ndarray:
                            # indexes and slices an image like an array.
        return self._data[key]


class DicomImage(Image):
    def __init__(self, data: np.ndarray, metadata: dict, spacing: tuple[float, ...] | None = None, ):
        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary")
        super().__init__(data, spacing)
        self.metadata = metadata


def _get_z_position(dicom_image: Dataset) -> float:
    """Helper function that returns the z-coordinate of the Image Position Patient DICOM attribute."""
    return dicom_image.ImagePositionPatient[2]


def _get_spacing(dicom_list: list[Dataset]) -> tuple[float, ...]:
    """Return DICOM rows, columns, and optional slice spacing in millimetres for multiple DICOM files."""

    n = len(dicom_list)  # The number of slices in the list
    dx, dy = dicom_list[0].PixelSpacing  # Read the X and Y pixel spacing from the first slice.

    if n > 1:
        z_first = np.array(dicom_list[0].ImagePositionPatient)
        z_last = np.array(dicom_list[-1].ImagePositionPatient)
        dz = np.linalg.norm(z_last - z_first) / (n - 1)
        # Spacing between slices is calculated as distance from the first to the last slice centre, divided by the number of gaps (n slices = n-1 gaps)"""
        return (float(dx), float(dy), float(dz))
    else:
        return (float(dx), float(dy))  # Since a single slice has no neighbouring slice, only 2D spacing exists.


def read_dicom(file_path: str) -> DicomImage:
    """Read one DICOM image or a directory of images."""

    dicom_images = []
    if os.path.isfile(file_path):  # Single DICOM file
        dicom_images.append(dcmread(file_path))
    elif os.path.isdir(file_path):  # Directory with multiple DICOM files
        for file_name in os.listdir(file_path):
            full_name = os.path.join(file_path, file_name)
            if os.path.isfile(full_name):
                dicom_images.append(dcmread(full_name))
    else:
        raise ValueError("DICOM path does not exist")

    if not dicom_images:
        raise ValueError(f"No DICOM files found")

    series_uids = set()
    for image in dicom_images:
        series_uids.add(image.SeriesInstanceUID)
    if len(series_uids) > 1:
        # A set only stores unique values, so if there are more UIDs in the set, the slices belong to different scans."""
        raise ValueError("DICOM files have different Series Instance UIDs.")

    if len(dicom_images) > 1:
        dicom_images.sort(key=_get_z_position)  # Slices are sorted in anatomical order along z

    pixel_values = []
    for image in dicom_images:
        pixel_values.append(apply_rescale(image.pixel_array, image))  # Convert the pixel values to real values

    if len(pixel_values) == 1:
        data = pixel_values[0]
    else:
        data = np.stack(pixel_values, axis=1)

    metadata = {item.keyword: item.value for item in dicom_images[0] if item.keyword and item.keyword != 'PixelData'}
    # for each item in the dataset of the image, add the keyword and value to the metadata, except for 'PixelData'

    return DicomImage(data=data, metadata=metadata, spacing=_get_spacing(dicom_images))


def read_nifti(file_path: str) -> Image:
    """
    Reads a Nifti file and returns the data and spacing.
    """
    nifti = nib.load(file_path)  # Read the image.
    transposed_nifti = np.transpose(nifti.get_fdata())
    mirrored_nifti = np.flip(transposed_nifti, axis=(-1, -2))
    # Nifti uses (x, y, z) axis, while DICOM uses (z, y, x), so we transpose and mirror the NIFTI to be consistent with DICOM.

    spacing = nifti.header.get_zooms()
    nifti_image = Image(mirrored_nifti, spacing)
    return nifti_image


def write_nifti(image: Image, file_path: str) -> None:
    if not isinstance(image, Image):
        raise TypeError("The image must be an Image object.")
    if np.ndim(image) not in (2, 3):
        raise ValueError("Pixel data must be 2D or 3D.")

    pixel_data = np.asarray(image, dtype=float)
    mirrored = np.flip(pixel_data, axis=(-1, -2))  # Undo mirroring and transposing done in read_nifti()
    transposed = np.transpose(mirrored)

    affine = np.eye(4)
    affine[0, 0] = image.spacing[0]  # Set the diagonal values of the affine matrix to the pixel spacing values.
    affine[1, 1] = image.spacing[1]
    if len(image.spacing) == 3:
        affine[2, 2] = image.spacing[2]

    nib.save(nib.Nifti1Image(transposed, affine), file_path)


def dicom_to_nifti(dicom_path: str, nifti_path: str) -> None:
    """Convert a DICOM file or folder of DICOM files to a NIfTI file.

    Stored DICOM pixels are converted to physical values.
    A directory must contain one series of dicom files, the slices are automatically sorted by patient z-position.
    """
    image = read_dicom(dicom_path)
    write_nifti(image, nifti_path)


def _masked_gaussian_filter(image: Image, mask: np.ndarray, sigma: tuple[float, ...]) -> Image:
    """
    Gaussian smoothing restricted to a mask
    Converts the image mask to an array and float and then multiplies them
    Computes M * (G(M * I, sigma) / G(M, sigma))
    """
    if not isinstance(image, Image):
        raise TypeError("The image must be an Image object.")
    if not isinstance(mask, np.ndarray):
        raise TypeError("The mask must be a NumPy array.")
    if np.shape(image) != mask.shape:  # Checks if the shapes of the dataset and the mask are consistent, otherwise raises an error
        raise ValueError("Image and mask are inconsistent!")
    if mask.dtype != np.bool_:
        raise ValueError("The mask must be a boolean mask!")

    sigma_pixels = tuple(
        value / spacing
        for value, spacing in zip(sigma, image.spacing)
    )
    pixel_data = np.asarray(image, dtype=np.float32)  # Convert it to a float
    mask = mask.astype(np.float32)

    # Apply a gaussian filter
    smoothed = gaussian_filter(input=(pixel_data * mask), sigma=sigma_pixels)
    masked = gaussian_filter(input=mask, sigma=sigma_pixels)

    filtered_image = mask * (smoothed / masked)  # Calculates M * (G(M * I, sigma) / G(M, sigma))
    filtered_image = np.nan_to_num(filtered_image, nan=0.0, posinf=0.0, neginf=0.0)  # Switches nan and inf for zeroes
    return Image(filtered_image, image.spacing)


def _test(file_path: str) -> None:
    # Test the read_dicom function
    image = read_dicom(file_path)
    #view(image.data, title="Result of read_dicom funcion")
    view(np.asarray(image), title="Result of read_dicom funcion")

    #Tests the NumPy conversion
    print(len(image), np.shape(image), np.ndim(image))
    print(image[0].shape)
    print(np.asarray(image, dtype=np.float32).dtype)

    # Test the Gaussian filter function
    #data = np.asarray(image.data, dtype=np.float32)
    data = np.asarray(image, dtype=np.float32)
    mask = data > -300
    data.ndim
    sigma = (2.0,) * data.ndim  # Give a standard deviation for each dimension of the image
    filtered_image = _masked_gaussian_filter(image, mask, sigma=sigma)
    view(np.asarray(filtered_image), title="Result of _masked_gaussian_filter function")

    # Test the write_nifti and read_nifti function
    nifti_path = "test_nifti.nii.gz"
    write_nifti(image, nifti_path)
    nifti_image = read_nifti(nifti_path)
    view(np.asarray(nifti_image), title="Result of write_nifti and read_nifti functions")

    # Test the dicom_to_nifti function
    dicom_to_nifti(file_path, nifti_path)
    nifti_image = read_nifti(nifti_path)
    view(np.asarray(nifti_image), title="Result of dicom_to_nifti and read_nifti functions")


# actually run the stuff
if __name__ == "__main__":
    print("Please enter the path to a DICOM image / DICOM images.")
    _test(input().strip())