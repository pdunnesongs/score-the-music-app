"""The PC engine's scan pipeline, run on the phone (unpacked into Pyodide as stm_device).

stm_scan is the engine's own scan code (engine/scan: omr, text_layers, melody_chords, backing,
tab_image), so a phone reads a sheet the same way the PC does: straightened pages, printed tab,
chord names and lyrics (tesseract.js through tessshim), a Melody part with chords worked out from
the printed parts, and the chord backing. Only the page reader (homr) runs in-process here instead
of as a separate program.
"""
import contextlib
import io
import sys
from pathlib import Path

import js
import stm_scan.omr as omr
import stm_scan.text_layers as text_layers
import homr.main


class _Tee(io.TextIOBase):
    def __init__(self, a):
        self.a, self.buf = a, io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self.a.write(s)

    def flush(self):
        self.a.flush()


_last_error: list = [None]
_homr_run = homr.main.run_homr


def _run_and_keep_error(*a, **k):
    # homr's main() turns any error into exit code 1; keep the error itself (e.g. "No staffs found")
    try:
        return _homr_run(*a, **k)
    except Exception as e:
        _last_error[0] = e
        raise


homr.main.run_homr = _run_and_keep_error


def _run_homr_here(image: Path, timeout: int = 600) -> str:
    out = image.with_suffix(".musicxml")
    tee = _Tee(sys.stderr)
    _last_error[0] = None
    sys.argv = ["homr", "--no-title", "--write-staff-positions", str(image)]
    try:
        with contextlib.redirect_stderr(tee):
            homr.main.main()
    except SystemExit as e:
        if e.code:
            raise RuntimeError(f"Music recognition failed on {image.name}: {_last_error[0] or ''} {tee.buf.getvalue()[-2000:]}")
    except Exception as e:
        raise RuntimeError(f"Music recognition failed on {image.name}: {e}")
    if not out.exists():
        raise RuntimeError(f"Music recognition failed on {image.name}: {_last_error[0] or ''} {tee.buf.getvalue()[-2000:]}")
    return out.read_text(encoding="utf-8")


omr._run_homr = _run_homr_here


def read(paths, title: str, text: bool) -> dict:
    """Read the pages (PNG paths) and return dict(xml, warnings), as the engine's /scan does."""
    text_layers.tesseract_available = (lambda: True) if text else (lambda: False)
    r = omr.recognise_images([Path(p) for p in paths], title,
                             progress=lambda step, f: js.stmProgress(step, f),
                             melody_chords=True, source_name=title)
    if not text:
        r.warnings[:] = [w for w in r.warnings if "Tesseract" not in w]
        r.warnings.insert(0, "Chord names and words printed on the page were not read (the text reader "
                             "couldn't start on this device).")
    return {"xml": r.musicxml, "warnings": list(r.warnings)}
