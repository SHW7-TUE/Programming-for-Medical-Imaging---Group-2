"""This script combines our group solutions for assignment 2A and 2B. Image pixel data and spacing 
always use the same order: (row, column) for a 2D image and
(row, column, slice) for a 3D volume. The conversion to NIFTI uses the same simple affine matrix
as the exercises we have done in previous weeks.
"""

from numbers import Real
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from pmi_viewer import view
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.pixels import apply_rescale
from scipy.ndimage import gaussian_filter

__all__ = ["Image", "DicomImage", "read_dicom", "read_nifti", "write_nifti", "dicom_to_nifti", "masked_gaussian_filter", "test"]


class Image:
    """Class to store image data and spacings in millimetres for each axis."""

    def __init__(self, data: np.ndarray, spacing: tuple[float, ...] | None = None,) -> None:
        """Store an array with one spacing per dimension."""
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
        # The len(image) line here returns the size of the outer dimension.
        return len(self._data)

    def __array__(self, dtype: np.typing.DTypeLike | None = None, *, copy: bool | None = None,) -> np.ndarray:
        # numpy obtains the pixel data, optionally casting it for compatibility.
        if dtype is None:
            dtype = self._data.dtype
        if copy is False and np.dtype(dtype) != self._data.dtype:
            raise ValueError("Changing the pixel datatype requires a copy.")
        return self._data.astype(dtype, copy=copy is True)

    def __getitem__(self, key: Any) -> np.ndarray:
        # Let image[key] be an index and return it. This also ensures further compatibility since any
        # type is accepted. 
        return self._data[key]


class DicomImage(Image):
    """A class containing an Image with a dictionary of DICOM attributes."""

    def __init__(self, data: np.ndarray, spacing: tuple[float, ...] | None, metadata: dict,) -> None:
        """Store the image data, spacing, and DICOM metadata."""
        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary.")
        super().__init__(data, spacing)
        self.metadata = metadata

def _get_z_position(dicom_image: Dataset) -> float:
    """Return the z-coordinate of a DICOM slice."""
    return float(dicom_image.ImagePositionPatient[2])


def _get_spacing(dicom_images: list[Dataset]) -> tuple[float, ...]:
    """Return row, column, and optional slice spacing in millimetres."""
    if dicom_images[0].get("PixelSpacing") is None:
        raise ValueError("DICOM image must provide PixelSpacing in millimetres.")
    row_spacing, column_spacing = (float(value) for value in dicom_images[0].PixelSpacing)
    if len(dicom_images) == 1:
        return row_spacing, column_spacing

    # Linear interpolation for the optional z-spacing
    first_position = np.asarray(dicom_images[0].ImagePositionPatient, dtype=float)
    last_position = np.asarray(dicom_images[-1].ImagePositionPatient, dtype=float)
    distance = np.linalg.norm(last_position - first_position)
    slice_spacing = distance / (len(dicom_images) - 1)
    return row_spacing, column_spacing, float(slice_spacing)

def read_dicom(file_path: str | Path) -> DicomImage:
    """Read one DICOM file or a folder containing a series of DICOM slices."""
    source = Path(file_path)
    dicom_images: list[Dataset] = []

    if source.is_file():
        dicom_images.append(dcmread(source))
    elif source.is_dir():
        for path in source.iterdir():
            if path.is_file():
                dicom_images.append(dcmread(path))
    else:
        raise ValueError(f"DICOM path does not exist: {source}")

    if not dicom_images:
        raise ValueError(f"No DICOM files found in: {source}")

    series_uids = {image.get("SeriesInstanceUID") for image in dicom_images}
    if len(series_uids) > 1:
        raise ValueError("DICOM files have different Series Instance UIDs.")

    if len(dicom_images) > 1:
        dicom_images.sort(key=_get_z_position)

    pixel_values = [apply_rescale(image.pixel_array, image) for image in dicom_images]
    if len(pixel_values) == 1:
        data = pixel_values[0]
    else:
        # Rows and columns first, and then add the slice axis last.
        data = np.stack(pixel_values, axis=-1)

    # These lines are for storing the DICOM attributes from the DICOM class that we defined earier.
    # Each attributes keyword is used as its dictionary key, skipping any unnamed attributes and the PixelData.
    metadata = {
        item.keyword: item.value
        for item in dicom_images[0]
        if item.keyword and item.keyword != "PixelData"
    }
    return DicomImage(data, _get_spacing(dicom_images), metadata)

def read_nifti(file_path: str | Path) -> Image:
    """Read a 2D or 3D NIfTI image using the same simple transpose logic."""
    nifti_image = nib.load(file_path)
    if len(nifti_image.shape) not in (2, 3):
        raise ValueError("NIfTI data must be 2D or 3D.")

    # Undo the NIFTI axis swap and axis flips for NIFTI reading.
    data = np.swapaxes(nifti_image.get_fdata(), 0, 1)
    data = np.flip(data, axis=(0, 1))
    nifti_spacing = nifti_image.header.get_zooms()
    spacing = (nifti_spacing[1], nifti_spacing[0], *nifti_spacing[2:])
    return Image(data, tuple(float(value) for value in spacing))


def write_nifti(image: Image, file_path: str | Path) -> None:
    """Write a 2D or 3D Image as a NIfTI file."""
    if not isinstance(image, Image):
        raise TypeError("The image must be an Image object.")
    if np.ndim(image) not in (2, 3):
        raise ValueError("NIfTI data must be 2D or 3D.")

    # Convert from (row, column, slice) to NIfTI (x, y, z).
    data = np.asarray(image, dtype=float)
    data = np.swapaxes(data, 0, 1)
    data = np.flip(data, axis=(0, 1))

    affine = np.eye(4)
    affine[0, 0] = image.spacing[1]  # NIfTI x column spacing
    affine[1, 1] = image.spacing[0]  # NIfTI y row spacing
    if np.ndim(image) == 3:
        affine[2, 2] = image.spacing[2]

    nib.save(nib.Nifti1Image(data, affine), file_path)


def dicom_to_nifti(dicom_path: str | Path, nifti_path: str | Path) -> None:
    """Convert a DICOM file or a directory of slices to NIfTI."""
    image = read_dicom(dicom_path)
    write_nifti(image, nifti_path)


def masked_gaussian_filter(image: Image, mask: np.ndarray, sigma: tuple[float, ...],) -> Image:
    """Smooth an Image within a boolean mask with sigma in millimetres."""
    if not isinstance(image, Image):
        raise TypeError("The image must be an Image object.")
    if not isinstance(mask, np.ndarray):
        raise TypeError("The mask must be a NumPy array.")
    if mask.shape != np.shape(image):
        raise ValueError("Image and mask must have the same shape.")
    if mask.dtype != np.bool_:
        raise ValueError("The mask must be a boolean array.")
    if not isinstance(sigma, tuple):
        raise TypeError("Sigma must be a tuple of numbers.")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in sigma):
        raise TypeError("Sigma values must be real numbers.")
    if len(sigma) != np.ndim(image):
        raise ValueError("Sigma must have one value per image dimension.")

    sigma = tuple(float(value) for value in sigma)
    if any(not np.isfinite(value) or value < 0 for value in sigma):
        raise ValueError("Sigma values must be finite and non-negative.")

    # Convert sigma in millimetres to pixels for calculations.
    sigma_pixels = tuple(value / spacing for value, spacing in zip(sigma, image.spacing))
    pixel_data = np.asarray(image, dtype=np.float32)
    weights = mask.astype(np.float32)
    smoothed_image = gaussian_filter(pixel_data * weights, sigma=sigma_pixels)
    smoothed_mask = gaussian_filter(weights, sigma=sigma_pixels)

    # Calculate M * G(M * I) / G(M) without dividing by zero to prevent any errors that we got
    # in our first few attempts.
    filtered = np.zeros_like(smoothed_image)
    valid_pixels = mask & (smoothed_mask > 0)
    np.divide(smoothed_image, smoothed_mask, out=filtered, where=valid_pixels)
    return Image(filtered, image.spacing)


def _check_nifti_round_trip(original: Image, file_path: Path) -> None:
    """Check that saving and reading files maintains the pixel values and spacing between file formats."""
    restored = read_nifti(file_path)
    np.testing.assert_allclose(np.asarray(restored), np.asarray(original))
    np.testing.assert_allclose(restored.spacing, original.spacing)


def _view_image(image: Image, title: str) -> None:
    """Adapt the Image axes and spacing to the PMI viewer expected order. Otherwise weird flips and 
    translations can happen."""
    data = np.asarray(image)
    if data.ndim == 3:
        data = np.moveaxis(data, -1, 0)  # PMI Viewer expects (slice, row, column).
    viewer_spacing = (image.spacing[1], image.spacing[0], *image.spacing[2:])
    view(data, spacing=viewer_spacing, title=title)


def test(file_path: str | Path) -> None:
    """Check and display the combined functions using a DICOM file or series."""
    image = read_dicom(file_path)
    assert isinstance(image, Image)
    assert isinstance(image, DicomImage)
    assert isinstance(image.metadata, dict)
    assert "PixelData" not in image.metadata
    print(image)
    print(f"DICOM metadata: {len(image.metadata)} attributes")
    print(f"Pixel area: {image.get_pixel_area():.4f} mm²")
    if np.ndim(image) >= 3:
        print(f"Voxel volume: {image.get_voxel_volume():.4f} mm³")

    # Further checks to ensure that the image data is actually correct in shape and dimensionality.
    data = np.asarray(image)
    assert len(image) == data.shape[0]
    assert np.shape(image) == data.shape
    assert np.ndim(image) == data.ndim
    np.testing.assert_array_equal(image[0], data[0])
    assert np.asarray(image, dtype=np.float32).dtype == np.float32
    print(f"NumPy: length {len(image)}, shape {np.shape(image)}, dimensions {np.ndim(image)}")

    # Code that defines where the outputs of the testing function are written, by default this is
    # in the same directory as the script under outputs/
    output_directory = Path.cwd() / "outputs"
    output_directory.mkdir(exist_ok=True)
    source_name = Path(file_path).stem
    round_trip_path = output_directory / f"{source_name}_round_trip.nii.gz"
    converted_path = output_directory / f"{source_name}_converted.nii.gz"

    # Test the individual write/read functions before the completed conversion
    write_nifti(image, round_trip_path)
    _check_nifti_round_trip(image, round_trip_path)
    print(f"Verified write/read round trip: {round_trip_path}")

    dicom_to_nifti(file_path, converted_path)
    _check_nifti_round_trip(image, converted_path)
    print(f"Verified DICOM conversion: {converted_path}")

    # We use an example threshhold here of -300 in Hounsfield Units, but this is an arbitrary choice.
    # Same for the strength of sigma.
    mask = data > -300
    sigma = (2.0,) * data.ndim
    filtered = masked_gaussian_filter(image, mask, sigma)
    filtered_data = np.asarray(filtered)
    assert filtered_data.dtype == np.float32
    assert filtered.spacing == image.spacing
    assert np.all(np.isfinite(filtered_data))
    np.testing.assert_array_equal(filtered_data[~mask], 0)
    print(f"Masked Gaussian filter: sigma {sigma} mm")

    _view_image(image, "Original DICOM image")
    _view_image(filtered, "Masked Gaussian image")

if __name__ == "__main__":
    print("Please enter the path to a DICOM image or directory of images.")
    test(input().strip())
