"""Checks for the Assignment 2B command-line converter."""

import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from click.testing import CliRunner

from Assignment_2B import functions
from Assignment_2B.cli import dicom_to_nifti


class CliTests(unittest.TestCase):
    def test_help_explains_inputs_and_examples(self):
        result = CliRunner().invoke(dicom_to_nifti, ["--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("DICOM_PATH", result.output)
        self.assertIn("NIFTI_PATH", result.output)
        self.assertIn("dicom-to-nifti-2b", result.output)

    def test_command_calls_the_2b_conversion_function(self):
        with tempfile.TemporaryDirectory(prefix="pmi-2b-cli-") as directory:
            source = Path(directory) / "slice.dcm"
            source.touch()
            output = Path(directory) / "slice.nii.gz"

            with patch.object(functions, "dicom_to_nifti") as convert:
                result = CliRunner().invoke(dicom_to_nifti, [str(source), str(output)])

            self.assertEqual(result.exit_code, 0, result.output)
            convert.assert_called_once_with(source, output)
            self.assertIn(str(output), result.output)

    def test_command_ignores_another_functions_module(self):
        with tempfile.TemporaryDirectory(prefix="pmi-2b-cli-") as directory:
            source = Path(directory) / "slice.dcm"
            source.touch()
            output = Path(directory) / "slice.nii.gz"
            other_functions = ModuleType("functions")
            other_functions.dicom_to_nifti = Mock()

            with patch.dict("sys.modules", {"functions": other_functions}):
                with patch.object(functions, "dicom_to_nifti") as convert:
                    result = CliRunner().invoke(dicom_to_nifti, [str(source), str(output)])

            self.assertEqual(result.exit_code, 0, result.output)
            convert.assert_called_once_with(source, output)
            other_functions.dicom_to_nifti.assert_not_called()

    def test_missing_input_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="pmi-2b-cli-") as directory:
            source = Path(directory) / "missing.dcm"
            output = Path(directory) / "slice.nii.gz"
            result = CliRunner().invoke(dicom_to_nifti, [str(source), str(output)])

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("does not exist", result.output)
            self.assertFalse(output.exists())

    def test_conversion_error_is_reported(self):
        with tempfile.TemporaryDirectory(prefix="pmi-2b-cli-") as directory:
            source = Path(directory) / "slice.dcm"
            source.touch()
            output = Path(directory) / "slice.nii.gz"
            with patch.object(functions, "dicom_to_nifti", side_effect=ValueError("Invalid image")):
                result = CliRunner().invoke(dicom_to_nifti, [str(source), str(output)])

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Invalid image", result.output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
