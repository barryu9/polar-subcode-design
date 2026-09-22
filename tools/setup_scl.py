"""Build the optional min-sum SCL extension in the reader package."""
import os
from pathlib import Path
import numpy as np
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
setup(name='polar-subcode-native-scl', version='0.1.0',
      ext_modules=[Extension('polar_subcodes.decoding._scl_step_decoder_cpp',
          sources=['cpp/scl_step_decoder_cpp.cpp'], include_dirs=[np.get_include()],
          language='c++', extra_compile_args=['/O2','/std:c++17'] if os.name=='nt' else ['-O3','-std=c++17'])])
