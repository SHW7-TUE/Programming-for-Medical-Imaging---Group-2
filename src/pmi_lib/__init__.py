import os

import nibabel as nib
import numpy as np
from pydicom import dcmread


def get_z_position(dicom_image):
    return dicom_image.ImagePositionPatient[2]    #note: sorting by z only works for axial scans


def get_spacing(dicom_list):
    n = len(dicom_list)
    dx, dy = dicom_list[0].PixelSpacing    # DICOM standard: [row spacing, column spacing]
                                           # for square pixels order doesnt matter

    if n > 1:                              #spacing between slices (dz) is calculated as distance
                                           # from the first to the last slice centre, divided by
                                           # the number of gaps (n slices = n - 1 gaps).
        z_first = np.array(dicom_list[0].ImagePositionPatient)
        z_last = np.array(dicom_list[-1].ImagePositionPatient)
        dz = np.linalg.norm(z_last - z_first) / (n - 1)
        return [dx, dy, dz]
    else:
        return [dx, dy]                 #since a single slice has no neighbouring slice, only 2D spacing exists.


def read_dicom(file_path):
    dicom_list = []

    if os.path.isfile(file_path):                   # single DICOM file
        dicom_list.append(dcmread(file_path))
    else:                                           # directory with multiple DICOM files
        for file_name in os.listdir(file_path):
            full_name = os.path.join(file_path, file_name)
            if os.path.isfile(full_name):
                dicom_list.append(dcmread(full_name))

                                                    # All files must belong to the same 3D volume.
                                                    # Different UIDs mean the slices belong to different scans.
    series_uids = set()
    for dicom_image in dicom_list:
        series_uids.add(dicom_image.SeriesInstanceUID)
    if len(series_uids) > 1:
        raise ValueError("DICOM files have different Series Instance UIDs.")

    sorted_list = sorted(dicom_list, key=get_z_position)    #slices are sorted in anatomical order along z

    sorted_images = []                              #this linear transformation converts raw integer values
                                                    # in the file into meaningful values:
                                                    # physical = stored * RescaleSlope + RescaleIntercept
    for dicom_image in sorted_list:
        slope = float(dicom_image.RescaleSlope)
        intercept = float(dicom_image.RescaleIntercept)
        sorted_images.append(dicom_image.pixel_array * slope + intercept)
                                                    # pixel_array is the 2D NumPy array of stored integer pixel
                                                    #values with shape (rows, columns)

    spacing_list = get_spacing(sorted_list)

    spacing = []                                    # convert spacing values to float, since they may be
                                                    # stored as strings in the DICOM header
    for s in spacing_list:
        spacing.append(float(s))
    spacing = tuple(spacing)

    if len(sorted_images) == 1:                     # for single files: 2D array, matching the 2D spacing.
        return sorted_images[0], spacing

    return np.array(sorted_images), spacing         # for multiple slices, a 3D numpy array with axes
                                                    # (slice, row, column) is stored with (dx, dy, dz) spacing


def read_nifti(file_path):
    nifti_image = nib.load(file_path)  # Read the image.
    transposed_nifti = nifti_image.get_fdata().T        # nifti uses (x, y, z) axes, while DICOM uses (z,y, x).
    mirrored_nifti = np.flip(transposed_nifti, axis=0)  # so we transpose and mirror the nifti to be
                                                        # consistent with DICOM

    spacing = tuple(float(s) for s in nifti_image.header.get_zooms())   # we need to match the order of
                                                                        # (dx, dy, dz) from get_spacing().

    return mirrored_nifti, spacing


def write_nifti(pixel_data, spacing, file_path):
    pixel_data = np.asarray(pixel_data, dtype=float)            # Raises ValueError if not a number.
    spacing = [float(s) for s in spacing]

    if pixel_data.ndim not in (2, 3):
        raise ValueError("Pixel data must be 2D or 3D.")
    if len(spacing) != pixel_data.ndim:
        raise ValueError("Spacing must have one value per dimension of the pixel data.")
    if any(s <= 0 for s in spacing):                           #  A voxel cannot have zero or negative value.
        raise ValueError("Spacing values must be positive.")

                                                    # Undo mirroring and transposing done in read_nifti()
    mirrored = np.flip(pixel_data, axis=0)          # so the data goes back to (x, y, z), what NIfTI expects.
    transposed = mirrored.T

    affine = np.diag(spacing + [1.0] * (4 - len(spacing)))    # 4x4 affine from spacing maps voxel indices into
    nib.save(nib.Nifti1Image(transposed, affine), file_path)  # coordinates in mm.
                                                              # Missing dimensions for 2D files are padded with 1.


def dicom_to_nifti(dicom_path, nifti_path):
    result = read_dicom(dicom_path)
    pixel_data = result[0]
    spacing = result[1]
    write_nifti(pixel_data, spacing, nifti_path)