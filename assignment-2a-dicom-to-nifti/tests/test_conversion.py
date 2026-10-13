"""Basic checks for the DICOM-to-NIfTI converter."""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from click.testing import CliRunner
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from pmi_lib.cli import dicom_to_nifti as cli
from pmi_lib.conversion import dicom_to_nifti


class ConversionTests(unittest.TestCase):
    """Check the required file, directory, series, and CLI behaviour."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="pmi-test-")
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.series_uid = generate_uid()
        self.pixels = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)

    def _write_slice(self, path, *, z=0, series_uid=None):
        """Write a small, uncompressed CT slice."""
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        image = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        image.SOPClassUID = meta.MediaStorageSOPClassUID
        image.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        image.SeriesInstanceUID = series_uid or self.series_uid
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

    def test_single_file_is_rescaled_and_saved(self):
        source = self.root / "slice.dcm"
        output = self.root / "slice.nii.gz"
        self._write_slice(source)

        dicom_to_nifti(source, output)
        image = nib.load(output)
        self.assertEqual(image.shape, (3, 2))
        np.testing.assert_array_equal(
            image.get_fdata(), [[-1012, -1018], [-1014, -1020], [-1016, -1022]]
        )
        np.testing.assert_allclose(image.header.get_zooms(), [1.25, 0.75])

    def test_directory_is_sorted(self):
        source = self.root / "series"
        source.mkdir()
        self._write_slice(source / "last.dcm", z=2)
        self._write_slice(source / "first.dcm", z=0)

        output = self.root / "volume.nii"
        dicom_to_nifti(source, output)
        image = nib.load(output)
        self.assertEqual(image.shape, (3, 2, 2))
        np.testing.assert_array_equal(image.get_fdata()[0, 0], [-1012, -972])

    def test_mixed_series_are_rejected(self):
        source = self.root / "series"
        source.mkdir()
        self._write_slice(source / "first.dcm")
        self._write_slice(source / "second.dcm", z=2, series_uid=generate_uid())

        with self.assertRaises(ValueError):
            dicom_to_nifti(source, self.root / "volume.nii")

    def test_cli_converts_a_file(self):
        source = self.root / "slice.dcm"
        output = self.root / "slice.nii.gz"
        self._write_slice(source)

        result = CliRunner().invoke(cli, [str(source), str(output)])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(output.exists())

    def test_cli_help_explains_inputs_and_examples(self):
        result = CliRunner().invoke(cli, ["--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("DICOM_PATH", result.output)
        self.assertIn("NIFTI_PATH", result.output)
        self.assertIn("existing DICOM file or a directory of slices", result.output)
        self.assertIn("outputs/jaw_volume.nii.gz", result.output)
        self.assertIn("-h, --help", result.output)


if __name__ == "__main__":
    unittest.main()
