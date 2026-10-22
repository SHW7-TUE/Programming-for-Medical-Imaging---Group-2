# PMI library — Group 2
# Werner Huiskes, Soufyen Farhosh, Lucia van den Boogaart, Daphne Ploeg

This ZIP file contains our combined work for Assignments 2A and 2B. The installed python package 
can read DICOM and NIfTI images, store image data and spacing, convert DICOM slices
to NIfTI file format, and apply Gaussian smoothing with a mask.

The python script that performs these actions is `src/pmi_lib/functions.py`. The script
for the command terminal interface is in `src/pmi_lib/cli.py`. 

# Setup in a new directory

Copy the entire project folder to the new location and open a terminal in the
folder containing `pyproject.toml`. Python 3.14 or newer is needed, as well
as a DICOM file or a folder containing a series of slices. 

Create and activate a virtual Python environment, then install the necessary packages. 
Our library also installs all package dependencies.

Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Linux/macOS:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

# Commandline interface (CLI)

Installing this package also creates the necessary `dicom-to-nifti` command,
implemented with Click as needed for the assignment. It also requires an input 
DICOM file or a series of DICOM slices in a folder. Output file names should
end in `.nii` or `.nii.gz`.

After activation, the following code can be used to run the CLI command.
Replace `path/to/DICOM` below with your input path. The output directory must
already exist for the command to work.

```sh
mkdir outputs
dicom-to-nifti "path/to/DICOM" outputs/converted.nii.gz
dicom-to-nifti --help
```

The command saves one NIfTI file and prints its location. 

# How to Run the Full Library

With the Python environment active, run:

```sh
python -m pmi_lib.functions
```
The script will prompt you to enter the path to a DICOM slice or a folder.
Paths may be absolute or relative to your current directory. Quotation marks
are not needed for the path since we use a strip function. The image
data does not have to be inside the project folder!

Our test(path) function then prints image information and executes all the
necessary functions. It writes `<input>_round_trip.nii.gz` and `<input>_converted.nii.gz` under
`outputs/` in the current directory. It then displays the original and filtered
images one after another in PMI viewer. Close the first to continue to the next. 
Repeated runs will replace the same output files.

## License

MIT (See: `LICENSE.md`).
