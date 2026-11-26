"""A stand-in for `pytesseract` inside Pyodide (the phone's music reader).

The engine's chord-name and lyric reading (stm_scan/text_layers.py) calls pytesseract.image_to_data;
here the image goes to tesseract.js in JavaScript (self.stmTess in worker.mjs), the same Tesseract
engine and English data as on the PC, and Python waits for it with pyodide.ffi.run_sync (JSPI).
"""
import json
import re

import cv2
from pyodide.ffi import run_sync, to_js
import js


class Output:
    DICT = "dict"
    STRING = "string"
    BYTES = "bytes"


class _Cmd:
    tesseract_cmd = "tesseract.js"


pytesseract = _Cmd()
FIELDS = ("text", "left", "top", "width", "height", "conf")


def image_to_data(image, lang=None, config="", nice=0, output_type=Output.DICT, timeout=0, pandas_config=None):
    m = re.search(r"--psm\s+(\d+)", config or "")
    psm = int(m.group(1)) if m else 3
    ok, png = cv2.imencode(".png", image)
    if not ok:
        return {k: [] for k in FIELDS}
    words = json.loads(run_sync(js.stmTess.recognize(to_js(png.tobytes()), psm)))
    return {k: [w[k] for w in words] for k in FIELDS}


def image_to_string(image, lang=None, config="", nice=0, output_type=Output.STRING, timeout=0):
    return " ".join(t for t in image_to_data(image, config=config)["text"] if t)


def get_tesseract_version():
    return "5"
