# Polar Low-Weight Spectrum

English · [简体中文](README.zh-CN.md)

This directory contains the Python and C++ counting kernel used to evaluate the exact low-weight spectra of the paper's codes. The kernel's authors and software version are listed in [CITATION.cff](CITATION.cff). Its [MIT license](LICENSE) applies to this dependency.

From the repository root, run `python tools/build_native.py` to build the native library. The [spectrum interface](../../polar_subcodes/spectrum/backend.py) reads a code's checks and calls the kernel with a weight cutoff. The [repository guide](../../README.md) includes a complete example for the code in Fig. 5.
