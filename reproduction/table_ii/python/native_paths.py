"""Native library names shared by the cold-replay build and runtime."""
import os
from pathlib import Path
from sys import platform


def library_name(stem):
    if os.name == "nt":
        extension = ".dll"
    elif platform == "darwin":
        extension = ".dylib"
    else:
        extension = ".so"
    return stem + extension


def library_path(stem):
    return Path(__file__).with_name(library_name(stem))
