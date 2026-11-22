"""Run a muFFTTO example unchanged, optionally forcing muGrid.FFTEngine onto the GPU."""
import sys, runpy, traceback
import matplotlib; matplotlib.use("Agg")
import muGrid

device = sys.argv[1]          # "cpu" or "gpu"
script = sys.argv[2]
if device == "gpu":
    _orig = muGrid.FFTEngine.__init__
    def _init(self, *a, **kw):
        kw["device"] = "gpu"
        _orig(self, *a, **kw)
    muGrid.FFTEngine.__init__ = _init
sys.argv = [script]
try:
    runpy.run_path(script, run_name="__main__")
except Exception:
    traceback.print_exc()
    sys.exit(1)
