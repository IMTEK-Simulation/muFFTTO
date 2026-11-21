"""Run one muFFTTO example on CPU or (forced) GPU.

usage: python sweep_run.py {cpu|gpu} SCRIPT [script args...]

GPU mode:
  * forces muGrid.FFTEngine(device="gpu")
  * imports muFFTTO from the patched scratch copy (muFFTTO_gpu/) before the
    example can put the repo copy on sys.path
  * instead of failing on the first host/device mismatch, it converts and
    LOGS every implicit transfer (cupy->numpy, numpy written into a cupy
    array, NuMPI reduction of cupy data), so one run shows all of them.
At exit, a summary of transfer sites (file:line, count) is written to stderr.
"""
import collections
import faulthandler
import os
import runpy
import sys
import traceback

faulthandler.enable()
import matplotlib

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
device, script = sys.argv[1], os.path.abspath(sys.argv[2])
script_args = sys.argv[3:]

sites = collections.Counter()


def _site():
    """first stack frame in muFFTTO or the example script"""
    for fr in reversed(traceback.extract_stack()):
        if fr.filename == os.path.abspath(__file__):
            continue
        if "/muFFTTO/muFFTTO/" in fr.filename or "muFFTTO_gpu/muFFTTO/" in fr.filename \
                or fr.filename == script:
            name = fr.filename
            if "muFFTTO_gpu/" in name:
                name = "muFFTTO/" + name.split("muFFTTO_gpu/muFFTTO/")[-1]
            elif name == script:
                name = os.path.basename(name)
            return f"{name}:{fr.lineno} ({fr.name})"
    return "<other>"


if device == "gpu":
    sys.path.insert(0, os.path.join(HERE, "muFFTTO_gpu"))
    import muFFTTO  # noqa: F401  (cache the patched copy in sys.modules)
    import muFFTTO.domain  # noqa: F401
    assert "muFFTTO_gpu" in muFFTTO.__file__, muFFTTO.__file__

    import cupy as cp
    import numpy as np
    import muGrid

    _orig_init = muGrid.FFTEngine.__init__

    def _init(self, *a, **kw):
        kw["device"] = "gpu"
        _orig_init(self, *a, **kw)

    muGrid.FFTEngine.__init__ = _init

    def _array(self, dtype=None, copy=None):
        sites["to_host  " + _site()] += 1
        a = self.get()
        return a.astype(dtype) if dtype is not None else a

    cp.ndarray.__array__ = _array

    _orig_setitem = cp.ndarray.__setitem__

    def _setitem(self, key, value):
        if isinstance(value, np.ndarray) and value.ndim > 0:
            sites["to_dev   " + _site()] += 1
            value = cp.asarray(value)
        return _orig_setitem(self, key, value)

    cp.ndarray.__setitem__ = _setitem

    # mixed numpy/cupy arithmetic: numpy-initiated (np_arr + cp_arr -> __array_ufunc__)
    _orig_ufunc = cp.ndarray.__array_ufunc__

    def _conv(args, tag):
        out = []
        for a in args:
            if isinstance(a, np.ndarray) and a.ndim > 0:
                sites[f"mixed    {_site()} [{tag}]"] += 1
                a = cp.asarray(a)
            out.append(a)
        return out

    def _array_ufunc(self, ufunc, method, *inputs, **kw):
        out = kw.get("out")
        if out is not None and any(isinstance(o, np.ndarray) for o in out):
            # numpy array updated in place with cupy data (np_arr += cp_arr):
            # compute on device, copy result back into the numpy output
            sites[f"to_host  {_site()} [in-place {ufunc.__name__} into numpy]"] += 1
            kw = dict(kw)
            del kw["out"]
            res = _orig_ufunc(self, ufunc, method, *_conv(inputs, ufunc.__name__), **kw)
            out[0][...] = res.get()
            return out[0]
        return _orig_ufunc(self, ufunc, method, *_conv(inputs, ufunc.__name__), **kw)

    cp.ndarray.__array_ufunc__ = _array_ufunc

    # ... and cupy-initiated (cp_arr + np_arr -> cupy dunder methods)
    for _op in ("add", "sub", "mul", "truediv", "floordiv", "pow", "matmul", "mod"):
        for _name in (f"__{_op}__", f"__r{_op}__", f"__i{_op}__"):
            if hasattr(cp.ndarray, _name):
                def _mk(orig, nm):
                    def f(self, other):
                        (other,) = _conv((other,), nm)
                        return orig(self, other)
                    return f
                setattr(cp.ndarray, _name, _mk(getattr(cp.ndarray, _name), _name))

    from NuMPI.Tools import Reduction

    def _wrap(name):
        orig = getattr(Reduction, name)

        def f(self, arr, *a, **kw):
            if isinstance(arr, cp.ndarray):
                sites[f"reduce   {_site()} [Reduction.{name}]"] += 1
                arr = arr.get()
            return orig(self, arr, *a, **kw)

        setattr(Reduction, name, f)

    for _n in ("sum", "max", "min", "mean", "any", "all", "dot"):
        if hasattr(Reduction, _n):
            _wrap(_n)

sys.argv = [script] + script_args
rc = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    rc = e.code or 0
except Exception:
    traceback.print_exc()
    rc = 1
finally:
    if device == "gpu":
        print("\n=== implicit host/device transfers (site, count) ===", file=sys.stderr)
        for k, v in sorted(sites.items(), key=lambda kv: -kv[1]):
            print(f"{v:8d}  {k}", file=sys.stderr)
        print("=== end transfers ===", file=sys.stderr)
sys.exit(rc)
