"""Always load the SPS-capable standard backend built inside this repository."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1] / "submodules" / "gaussian-rasterization"
sys.path.insert(0, str(_root))
try:
    import diff_gaussian_rasterization as backend
except ImportError as error:
    raise ImportError("Build the local standard renderer first: cd {} && python setup.py build_ext --inplace".format(_root)) from error
if _root not in Path(backend.__file__).resolve().parents:
    raise ImportError("A foreign diff_gaussian_rasterization was imported before the local SPS backend")
GaussianRasterizationSettings = backend.GaussianRasterizationSettings
GaussianRasterizer = backend.GaussianRasterizer
SparseGaussianAdam = getattr(backend, "SparseGaussianAdam", None)
