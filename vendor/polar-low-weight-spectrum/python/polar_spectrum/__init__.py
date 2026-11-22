"""Exact low-weight Polar/PAC spectra: reference Python and native C++."""
from .algorithm import algorithm1, WorkLimitError

__version__ = "0.1.0"
__all__ = ["algorithm1", "WorkLimitError"]
