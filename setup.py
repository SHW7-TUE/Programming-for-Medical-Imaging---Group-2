import platform
import shutil
import subprocess
import sys

from pathlib import Path

from setuptools import Command, setup
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.errors import SetupError  # Prints nicely without traceback


# ANSI markup escape sequences.
BOLD = "\033[1m"
RED = "\033[31m"
RESET = "\033[0m"

HELP_WINDOWS = f"""
    * {BOLD}Install Chocolatey{RESET} as explained here:  https://docs.chocolatey.org/en-us/choco/setup/
    * Close and reopen the Administrative shell in which you installed Chocolatey, and
        {BOLD}install MinGW{RESET} by running:  choco install mingw -y
    * Close and reopen this terminal and try building again
"""

HELP_DARWIN = f"""
    * {BOLD}Install Clang{RESET} by running:  xcode-select --install
    * Try building again
"""

HELP_UNIX = f"""
    * {BOLD}Update the package list{RESET} by running:  sudo apt update
    * Then {BOLD}install GCC{RESET} by running:  sudo apt install build-essential
    * Try building again
"""

_package_root = Path(__file__).parent / "src" / "pmi_lib"


def _get_input_files():
    return [str(file) for file in _package_root.rglob("*.c")]


# BuildNative.run() is executed when
class BuildNative(Command):
    description = "compile the native library"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        input_files = _get_input_files()

        # We don't need to build anything if filters.c is not present.
        if not input_files:
            return

        # Check if GCC is present and display the help text if it isn't.
        gcc = shutil.which("gcc")
        if not gcc:
            if platform.system() == "Windows":
                help = HELP_WINDOWS
            elif platform.system() == "Darwin":  # macOS
                help = HELP_DARWIN
            else:  # Linux / Unix
                help = HELP_UNIX
            raise SetupError(f"{BOLD}GCC was not found.{RESET} Please install it first:\n" + help)

        # Set common build arguments.
        build_args = ['-shared', '-O2', '-Wall', '-Wextra', '-Werror', '--std=c99', '-fdiagnostics-color=always']

        # Set platform specific build arguments.
        if platform.system() == "Windows":
            output_file = _package_root / "pmi.dll"
            cmd = ["gcc", *build_args, "-o", str(output_file), *input_files]
        elif platform.system() == "Darwin":  # macOS
            output_file = _package_root / "libpmi.dylib"
            cmd = ["gcc", *build_args, "-fPIC", "-o", str(output_file), *input_files]
        else:  # Linux / Unix
            output_file = _package_root / "libpmi.so"
            cmd = ["gcc", *build_args, "-fPIC", "-o", str(output_file), *input_files]

        # Run the build command. Raise an error when it fails.
        result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        if result.returncode != 0:
            raise SetupError(f"{BOLD}Failed to compile source file(s){RESET}")


# BuildPy is used for all builds.
class BuildPy(build_py):
    def run(self):
        self.run_command("build_native")  # First try to compile the native code.
        super().run()  # Then build everything else.


# BdistWheel.get_tag is called when has_ext_modules returns True. This tag is used for naming .whl archives.
class BdistWheel(bdist_wheel):
    def get_tag(self):
        _, _, platform = super().get_tag()
        return "py3", "none", platform  # Our native libraries do not depend on a specific Python version.


# EditableWheel is used for editable installs (python -m pip install -e .).
class EditableWheel(editable_wheel):
    def run(self):
        self.run_command("build_native")  # First try to compile the native code.
        super().run()  # Then install as usual.


setup(
    packages=["src/pmi_lib"],
    package_data={
        "src/pmi_lib": [
            "*.dll",
            "*.so",
            "*.dylib",
        ],
    },
    has_ext_modules=lambda: any(_get_input_files()),
    cmdclass={
        "build_native": BuildNative,
        "build_py": BuildPy,
        "editable_wheel": EditableWheel,
        "bdist_wheel": BdistWheel,
    },
)
