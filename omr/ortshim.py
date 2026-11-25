"""A stand-in for the `onnxruntime` Python package inside Pyodide (the browser).

homr calls onnxruntime synchronously; here every model call goes to onnxruntime-web in
JavaScript (window.stmOrt) and waits with pyodide.ffi.run_sync (needs JSPI). Tensors stay
in JavaScript until Python asks for .numpy(), so the decoder's key/value cache doesn't
travel back and forth every step.
"""
import os

import numpy as np
from pyodide.ffi import run_sync, to_js
import js

__version__ = "web"

_DT = {np.dtype(np.float32): "float32", np.dtype(np.float16): "float16", np.dtype(np.int64): "int64",
       np.dtype(np.int32): "int32", np.dtype(np.bool_): "bool", np.dtype(np.uint8): "uint8",
       np.dtype(np.int8): "int8"}
_NP = {v: k for k, v in _DT.items()}


def _to_tensor(a):
    a = np.ascontiguousarray(a)
    return js.stmOrt.tensor(_DT[a.dtype], to_js(a.tobytes()), to_js([int(d) for d in a.shape]))


def _from_tensor(t):
    raw = js.stmOrt.bytes(t).to_bytes()
    dims = [int(d) for d in t.dims.to_py()]
    return np.frombuffer(raw, dtype=_NP[str(t.type)]).reshape(dims).copy()


class OrtValue:
    def __init__(self, t):
        self._t = t

    @staticmethod
    def ortvalue_from_numpy(a, device_type="cpu", device_id=0):
        return OrtValue(_to_tensor(a))

    def numpy(self):
        return _from_tensor(self._t)

    def shape(self):
        return [int(d) for d in self._t.dims.to_py()]


class _Arg:
    def __init__(self, name):
        self.name = name


class IOBinding:
    def __init__(self, sess):
        self._sess = sess
        self._in = {}
        self._out = []
        self._results = []

    def bind_cpu_input(self, name, arr):
        self._in[name] = _to_tensor(arr)

    def bind_input(self, name, device_type, device_id, element_type, shape, buffer_ptr):
        raise NotImplementedError

    def bind_ortvalue_input(self, name, ov):
        self._in[name] = ov._t

    def bind_output(self, name, *args, **kwargs):
        if name not in self._out:
            self._out.append(name)

    def clear_binding_inputs(self):
        self._in = {}

    def clear_binding_outputs(self):
        self._out = []

    def get_outputs(self):
        return self._results

    def copy_outputs_to_cpu(self):
        return [r.numpy() for r in self._results]


class InferenceSession:
    def __init__(self, path, sess_options=None, providers=None, **kwargs):
        self._name = os.path.basename(str(path))
        self._s = run_sync(js.stmOrt.load(self._name))
        self._inputs = [_Arg(n) for n in self._s.inputNames.to_py()]
        self._outputs = [_Arg(n) for n in self._s.outputNames.to_py()]

    def __del__(self):
        # free the network's memory in JavaScript as soon as homr drops it (phones are tight on memory)
        try:
            js.stmOrt.release(self._s)
        except Exception:
            pass

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def io_binding(self):
        return IOBinding(self)

    def _run(self, feeds, names):
        res = run_sync(js.stmOrt.run(self._s, to_js(feeds, dict_converter=js.Object.fromEntries), to_js(names)))
        return [OrtValue(res[i]) for i in range(len(names))]

    def run(self, output_names, input_feed, run_options=None):
        names = output_names or [o.name for o in self._outputs]
        outs = self._run({k: _to_tensor(v) for k, v in input_feed.items()}, names)
        return [o.numpy() for o in outs]

    def run_with_iobinding(self, iobinding, run_options=None):
        names = iobinding._out or [o.name for o in self._outputs]
        iobinding._results = self._run(dict(iobinding._in), names)


class SessionOptions:
    def __init__(self):
        self.graph_optimization_level = None
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0
        self.log_severity_level = 3

    def add_session_config_entry(self, *a):
        pass


class GraphOptimizationLevel:
    ORT_DISABLE_ALL = 0
    ORT_ENABLE_BASIC = 1
    ORT_ENABLE_EXTENDED = 2
    ORT_ENABLE_ALL = 99


def get_available_providers():
    return ["CPUExecutionProvider"]


def get_device():
    return "CPU"


def set_default_logger_severity(level):
    pass


def preload_dlls(*a, **k):
    pass
