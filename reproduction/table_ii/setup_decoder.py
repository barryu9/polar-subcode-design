"""Build the historical SCL design-sample tie convention for this replay only."""
import os
from pathlib import Path
import numpy as np
from setuptools import Extension, setup

replay = Path(__file__).resolve().parent
repository = replay.parents[1]
os.chdir(replay / "python")
setup(
    name="polar-table-ii-design-decoder", version="1.0",
    packages=["polar_code"],
    ext_modules=[Extension(
        "polar_code._scl_step_decoder_cpp",
        [str(repository / "cpp/scl_step_decoder_cpp.cpp")],
        include_dirs=[np.get_include()],
        define_macros=[("POLAR_PAPER_REPLAY_TIES", "1")],
        extra_compile_args=["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"],
        language="c++",
    )],
)
