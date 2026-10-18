"""Checks for the combined Assignment 2A/2B working script."""

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
from scipy.ndimage import gaussian_filter

import functions


class ImageTests(unittest.TestCase):
    def test_default_spacing_and_string(self):
        image = functions.Image(np.zeros((2, 3), dtype=np.float32))

        self.assertEqual(image.spacing, (1.0, 1.0))
        self.assertEqual(
            str(image),
            "A float32 image with shape (2, 3) and a spacing of (1.0, 1.0) mm.",
        )
        self.assertEqual(image.get_pixel_area(), 1.0)
        with self.assertRaises(ValueError):
            image.get_voxel_volume()

    def test_area_and_volume_use_first_spatial_axes(self):
        image = functions.Image(np.zeros((2, 3, 4, 5)), (0.8, 0.8, 5, 2))

        self.assertEqual(image.spacing, (0.8, 0.8, 5.0, 2.0))
        self.assertAlmostEqual(image.get_pixel_area(), 0.64)
        self.assertAlmostEqual(image.get_voxel_volume(), 3.2)

        with self.assertRaises(ValueError):
            functions.Image(np.zeros(3)).get_pixel_area()

    def test_invalid_image_types_and_spacing(self):
        with self.assertRaises(TypeError):
            functions.Image([[1, 2]])

        data = np.zeros((2, 3))
        for spacing in ([1, 2], (1, "2"), (1, True)):
            with self.subTest(spacing=spacing), self.assertRaises(TypeError):
                functions.Image(data, spacing)
        for spacing in ((1,), (1, 0), (1, -1), (1, np.nan), (1, np.inf)):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                functions.Image(data, spacing)


class GaussianTests(unittest.TestCase):
    def test_sigma_is_in_millimetres_and_output_is_masked(self):
        data = np.zeros((9, 9, 5), dtype=np.float32)
        data[4, 4, 2] = 100
        image = functions.Image(data, (0.5, 1.0, 2.0))
        mask = np.ones(data.shape, dtype=bool)
        mask[:2, :, :] = False
        original = data.copy()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = functions.masked_gaussian_filter(image, mask, (2.0, 2.0, 2.0))

        # The chosen physical sigma corresponds to (4, 2, 1) pixels.
        weights = mask.astype(np.float32)
        numerator = gaussian_filter(data * weights, sigma=(4.0, 2.0, 1.0))
        denominator = gaussian_filter(weights, sigma=(4.0, 2.0, 1.0))
        expected = np.zeros_like(data)
        valid = mask & (denominator > 0)
        expected[valid] = numerator[valid] / denominator[valid]

        self.assertFalse(caught)
        self.assertEqual(result.data.dtype, np.float32)
        self.assertEqual(result.spacing, image.spacing)
        np.testing.assert_allclose(result.data, expected)
        np.testing.assert_array_equal(result.data[~mask], 0)
        np.testing.assert_array_equal(image.data, original)

    def test_empty_mask_and_zero_sigma(self):
        image = functions.Image(np.arange(12).reshape(3, 4), (0.5, 2.0))
        empty_mask = np.zeros(image.data.shape, dtype=bool)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = functions.masked_gaussian_filter(image, empty_mask, (2.0, 2.0))
        self.assertFalse(caught)
        np.testing.assert_array_equal(result.data, 0)

        full_mask = np.ones(image.data.shape, dtype=bool)
        result = functions.masked_gaussian_filter(image, full_mask, (0.0, 0.0))
        np.testing.assert_array_equal(result.data, image.data)

    def test_invalid_mask_and_sigma(self):
        image = functions.Image(np.zeros((2, 3)))
        mask = np.ones((2, 3), dtype=bool)
        with self.assertRaises(TypeError):
            functions.masked_gaussian_filter(image.data, mask, (1, 1))
        with self.assertRaises(TypeError):
            functions.masked_gaussian_filter(image, [[True]], (1, 1))
        for sigma in ([1, 1], (True, 1), ("1", 1)):
            with self.subTest(sigma=sigma), self.assertRaises(TypeError):
                functions.masked_gaussian_filter(image, mask, sigma)
        with self.assertRaises(ValueError):
            functions.masked_gaussian_filter(image, np.ones((3, 2), dtype=bool), (1, 1))
        with self.assertRaises(ValueError):
            functions.masked_gaussian_filter(image, np.ones((2, 3)), (1, 1))
        for sigma in ((1,), (-1, 1), (np.nan, 1), (np.inf, 1)):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                functions.masked_gaussian_filter(image, mask, sigma)


class NiftiAndDicomTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="pmi-functions-")
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.series_uid = generate_uid()
        self.pixels = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)

    def _write_slice(self, path, z):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        image = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        image.SOPClassUID = meta.MediaStorageSOPClassUID
        image.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        image.SeriesInstanceUID = self.series_uid
        image.Rows, image.Columns = self.pixels.shape
        image.SamplesPerPixel = 1
        image.PhotometricInterpretation = "MONOCHROME2"
        image.BitsAllocated = image.BitsStored = 16
        image.HighBit = 15
        image.PixelRepresentation = 1
        image.PixelSpacing = [0.75, 1.25]
        image.ImagePositionPatient = [0, 0, z]
        image.RescaleSlope = 2
        image.RescaleIntercept = -1024
        image.PixelData = (self.pixels + 10 * z).astype("<i2").tobytes()
        image.save_as(path, enforce_file_format=True)

    def test_nifti_round_trip_with_unequal_spacing(self):
        for shape, spacing in (((2, 3), (0.75, 1.25)), ((2, 3, 4), (0.75, 1.25, 2.5))):
            with self.subTest(shape=shape):
                data = np.arange(np.prod(shape)).reshape(shape)
                image = functions.Image(data, spacing)
                output = self.root / f"{len(shape)}d.nii.gz"
                functions.write_nifti(image, output)

                stored = nib.load(output)
                expected_data = np.flip(np.swapaxes(data, 0, 1), axis=(0, 1))
                np.testing.assert_array_equal(stored.get_fdata(), expected_data)
                np.testing.assert_allclose(
                    stored.header.get_zooms(), (spacing[1], spacing[0], *spacing[2:])
                )
                restored = functions.read_nifti(output)
                np.testing.assert_array_equal(restored.data, data)
                np.testing.assert_allclose(restored.spacing, spacing)

    def test_nifti_rejects_unsupported_dimensions(self):
        image = functions.Image(np.zeros((2, 2, 2, 2)))
        with self.assertRaises(ValueError):
            functions.write_nifti(image, self.root / "4d.nii.gz")
        nib.save(nib.Nifti1Image(image.data, np.eye(4)), self.root / "4d.nii.gz")
        with self.assertRaises(ValueError):
            functions.read_nifti(self.root / "4d.nii.gz")

    def test_dicom_volume_is_sorted_rescaled_and_written(self):
        source = self.root / "series"
        source.mkdir()
        self._write_slice(source / "last.dcm", z=2)
        self._write_slice(source / "first.dcm", z=0)

        image = functions.read_dicom(source)
        self.assertEqual(image.data.shape, (2, 3, 2))
        self.assertEqual(image.spacing, (0.75, 1.25, 2.0))
        np.testing.assert_array_equal(image.data[:, :, 0], self.pixels * 2 - 1024)
        np.testing.assert_array_equal(image.data[:, :, 1], (self.pixels + 20) * 2 - 1024)

        output = self.root / "combined.nii.gz"
        functions.dicom_to_nifti(source, output)
        actual = nib.load(output)
        expected_data = np.flip(np.swapaxes(image.data, 0, 1), axis=(0, 1))
        np.testing.assert_array_equal(actual.get_fdata(), expected_data)
        np.testing.assert_allclose(actual.affine, np.diag([1.25, 0.75, 2.0, 1.0]))

    def test_demo_saves_separate_outputs_and_adapts_viewer_axes(self):
        source = self.root / "series"
        source.mkdir()
        self._write_slice(source / "first.dcm", z=0)
        self._write_slice(source / "last.dcm", z=2)

        with patch.object(functions, "__file__", str(self.root / "functions.py")):
            with patch.object(functions, "view") as viewer:
                functions.test(source)

        self.assertTrue((self.root / "outputs" / "series_round_trip.nii.gz").exists())
        self.assertTrue((self.root / "outputs" / "series_converted.nii.gz").exists())
        self.assertEqual(viewer.call_count, 2)
        self.assertEqual(viewer.call_args_list[0].args[0].shape, (2, 2, 3))
        self.assertEqual(viewer.call_args_list[0].kwargs["spacing"], (1.25, 0.75, 2.0))


if __name__ == "__main__":
    unittest.main()
