# Assignment 2B — work in progress

This is a standalone working copy of the combined Assignment 2A/2B script.
It contains the `Image` class, DICOM and NIfTI functions, and the masked
Gaussian filter. Image data and spacing use `(row, column[, slice])` order;
Gaussian sigma is specified in millimetres.

The script uses the same simplified NIfTI geometry as Assignment 2A. This is
not yet the final group package: for example, the `DicomImage` subclass and
the other group members' parts are still to be integrated.

## Set up

From this directory, create a Python 3.14 virtual environment and install the
project and its dependencies:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -e .
```

On Windows, use `py -3.14 -m venv .venv` and
`.venv\Scripts\python -m pip install -e .` instead. The JPEG decoder is
included for the supplied compressed DICOM images. For the commands below on
Windows, replace `.venv/bin/python` with `.venv\Scripts\python`.

On Debian/Ubuntu, if creating the environment reports that `ensurepip` is
unavailable, install the matching Python venv package first (for example,
`sudo apt install python3.14-venv`), then retry the commands above.

For now the project can also be installed into the existing 2A environment.
From the PMI root directory, run:

```sh
Assignment_2A/.venv/bin/python -m pip install -e Assignment_2B
```

This makes both commands available in that environment: `dicom-to-nifti`
uses 2A, and `dicom-to-nifti-2b` uses 2B. In Bash, run `hash -r` if the shell
does not immediately find a newly installed command.

When importing the 2B functions in Python, use `Assignment_2B.functions` so
they are clearly distinguished from the separate 2A working script.

## Run and test

To convert a DICOM file or series without running the demonstration, use the
2B command. For example, from this directory with its environment active:

```sh
dicom-to-nifti-2b --help
dicom-to-nifti-2b ../documents/images/ct_jaw_slice.dcm outputs/jaw_slice.nii.gz
```

The output directory must already exist. The command writes one NIfTI file; it
does not run the Gaussian filter or open the viewer. Choose an output filename
that you are happy to replace if it already exists.

Run the demonstration and enter the path to a DICOM file or a directory
containing one DICOM series when prompted:

```sh
.venv/bin/python functions.py
```

For example, from this directory, enter `../documents/images/ct_jaw_slice.dcm`
for the supplied slice or `../documents/images/ct_jaw` for the series. The
script checks the NIfTI round trip and the DICOM conversion, then opens the
original and filtered images in the viewer. The viewer needs a graphical
desktop; its windows open one after the other.

The two NIfTI files are saved under `outputs/`, named after the input. Running
the demonstration again with the same input replaces those files. The filtered
image is displayed but is not saved.

Run the independent synthetic tests with:

```sh
.venv/bin/python -m unittest discover -s tests -t . -v
```

The tests use temporary DICOM and NIfTI files and mock the graphical viewer,
so the course data and a desktop are not needed to run them.
