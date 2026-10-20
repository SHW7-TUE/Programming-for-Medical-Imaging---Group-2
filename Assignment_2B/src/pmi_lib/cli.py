"""Click command for the Assignment 2A DICOM-to-NIfTI converter."""
from pathlib import Path
import click


from pmi_lib.functions import dicom_to_nifti as _convert

@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Convert a DICOM file, or a folder containing a series of DICOM files, "
        "to a NIfTI file.\n\n"
        "DICOM_PATH must be an existing DICOM file or a directory of slices "
        "from one series. NIFTI_PATH must be a destination filename ending in "
        ".nii or .nii.gz. The output directory must already exist!"
    ),
    epilog=(
        "\b\n"
        "Examples:\n"
        "  dicom-to-nifti ../documents/images/ct_jaw_slice.dcm "
        "outputs/jaw_slice.nii.gz\n"
        "  dicom-to-nifti ../documents/images/ct_jaw outputs/jaw_volume.nii.gz\n\n"
        "Directory slices are sorted by patient z-position."
    ),
)

@click.argument(
    "dicom_path",
    type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path),
    required=True,
)

@click.argument(
    "nifti_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
)

def dicom_to_nifti(dicom_path: Path, nifti_path: Path) -> None:
    """Convert DICOM_PATH (a file or directory) to NIFTI_PATH (.nii or .nii.gz)."""
    try:
        _convert(dicom_path, nifti_path)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        raise click.ClickException("Could not convert the supplied DICOM input") from exc
    click.echo(f"Saved NIfTI file to {nifti_path}")

if __name__ == "__main__":
    dicom_to_nifti()
