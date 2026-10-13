"""This script converts DICOM images to NIfTI using the same logic and conventions we used in the week 3 exercises."""

from pathlib import Path

import nibabel as nib
import numpy as np
from pydicom import dcmread
from pydicom.pixels import apply_rescale

__all__ = ["dicom_to_nifti"]


def dicom_to_nifti(dicom_path: str | Path, nifti_path: str | Path) -> None:
    """Convert a DICOM file or folder of DICOM files to a NIfTI file.

    Stored DICOM pixels are converted to physical values. A directory must
    contain one series of dicom files, the slices are automatically sorted by patient z-position.
    """
    data, spacing = _read_dicom(Path(dicom_path))
    _write_nifti(data, spacing, Path(nifti_path))


def _read_dicom(source: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    """Read one DICOM image or a directory of images."""
    images = []

    if source.is_file():
        images.append(dcmread(source))
    elif source.is_dir():
        for path in source.iterdir():
            if path.is_file():
                images.append(dcmread(path))
    else:
        raise ValueError(f"DICOM path does not exist: {source}")

    if not images:
        raise ValueError(f"No DICOM files found in: {source}")

    if source.is_dir() and not _is_single_series(images):
        raise ValueError("DICOM files belong to different series.")

    if len(images) > 1:
        images.sort(key=_get_z_position)

    pixels = []
    for image in images:
        pixels.append(apply_rescale(image.pixel_array, image))

    if len(pixels) == 1:
        data = pixels[0]
    else:
        data = np.stack(pixels)

    return data, _get_spacing(images)


def _is_single_series(images) -> bool:
    """Helper function that returns whether all images have the same series identifier."""
    series_uid = images[0].get("SeriesInstanceUID")
    for image in images[1:]:
        if image.get("SeriesInstanceUID") != series_uid:
            return False
    return True


def _get_z_position(image) -> float:
    """Helper function that returns an image z-position for slice sorting."""
    return float(image.ImagePositionPatient[2])


def _get_spacing(images) -> tuple[float, ...]:
    """Return DICOM rows, columns, and optional slice spacing in millimetres for multiple DICOM files."""
    pixel_spacing = images[0].PixelSpacing
    row_spacing = float(pixel_spacing[0])
    column_spacing = float(pixel_spacing[1])

    if len(images) == 1:
        return row_spacing, column_spacing

    first = np.asarray(images[0].ImagePositionPatient, dtype=float)
    last = np.asarray(images[-1].ImagePositionPatient, dtype=float)
    slice_spacing = np.linalg.norm(last - first) / (len(images) - 1)
    return row_spacing, column_spacing, float(slice_spacing)


def _write_nifti(data: np.ndarray, spacing: tuple[float, ...], destination: Path) -> None:
    """Write data with the same transpose and orientational flip as used in week 3."""
    nifti_data = np.transpose(data)
    nifti_data = np.flip(nifti_data, axis=(0, 1))

    affine = np.eye(4)
    affine[0, 0] = spacing[1]
    affine[1, 1] = spacing[0]
    if len(spacing) == 3:
        affine[2, 2] = spacing[2]

    nib.save(nib.Nifti1Image(nifti_data, affine), destination)
