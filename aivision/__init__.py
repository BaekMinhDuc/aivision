
__version__ = "8.4.75"

import importlib
import os
import sys
from typing import TYPE_CHECKING

# Pickle compatibility: resolve old ultralytics module path to aivision
sys.modules.setdefault("ultralytics", sys.modules[__name__])

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from aivision.utils import ASSETS, SETTINGS
from aivision.utils.checks import check_aivi as checks
from aivision.utils.downloads import download

settings = SETTINGS

MODELS = ("AIVI", "AIVIWorld", "AIVIE", "NAS", "SAM", "FastSAM", "RTDETR")

__all__ = (
    "__version__",
    "ASSETS",
    *MODELS,
    "checks",
    "download",
    "settings",
)

if TYPE_CHECKING:
    # Enable hints for type checkers
    from aivision.models import AIVI, AIVIWorld, AIVIE, NAS, SAM, FastSAM, RTDETR  # noqa


def __getattr__(name: str):
    """Lazy-import model classes on first access."""
    if name in MODELS:
        return getattr(importlib.import_module("aivision.models"), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """Extend dir() to include lazily available model names for IDE autocompletion."""
    return sorted(set(globals()) | set(MODELS))


if __name__ == "__main__":
    print(__version__)
