// Score The Music: read printed sheet music on the device itself, with no engine.
// Runs homr (the engine's music reader, Python) in Pyodide, and its three neural networks in
// onnxruntime-web. homr calls the networks synchronously; ortshim.py bridges that to
// onnxruntime-web's promises with pyodide.ffi.run_sync, which needs JSPI (Chrome/Android 137+).
//
// in:  { type: 'scan', pages: [Uint8Array PNG], title }
// out: { type: 'progress', step, fraction } | { type: 'done', xml, seconds } | { type: 'error', message }
import * as ort from './ort/ort.wasm.min.mjs';
import { loadPyodide } from './pyodide/pyodide.mjs';

export const MODEL_CACHE = 'stm-omr-models';
const base = new URL('./', import.meta.url).href;
const post = (m) => self.postMessage(m);
const progress = (step, fraction) => post({ type: 'progress', step, fraction });
// Every step and log line goes to the app too, so a failure on a phone says where it stopped.
const log = (line) => post({ type: 'log', line: String(line).slice(0, 300) });
self.addEventListener('error', (e) => post({ type: 'error', message: `The music reader stopped (${e.message || e.type} ${e.filename ? `in ${e.filename.split('/').pop()}:${e.lineno}` : ''})` }));
self.addEventListener('unhandledrejection', (e) => post({ type: 'error', message: `The music reader stopped (${e.reason?.message || e.reason})` }));

ort.env.wasm.wasmPaths = base + 'ort/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

// The model files come from the engine once and live in Cache Storage. homr asks for them by
// file name; match on what kind of model it is, so a newer homr on the PC still fits.
const role = (name) => (/segnet/.test(name) ? 'segnet' : /encoder/.test(name) ? 'encoder' : /decoder/.test(name) ? 'decoder' : name);
async function modelBytes(name) {
  const cache = await caches.open(MODEL_CACHE);
  for (const req of await cache.keys()) {
    if (role(decodeURIComponent(req.url.split('/').pop())) === role(name)) {
      return new Uint8Array(await (await cache.match(req)).arrayBuffer());
    }
  }
  throw new Error('The music reader is not downloaded yet. Connect to your PC once and choose "Get scanning ready".');
}

let onStderr = () => {};
let sessionsMade = false;
const stderr = (line) => { log(line); onStderr(line); };
self.stmOrt = {
  async load(name) {
    sessionsMade = true;
    log(`loading model ${role(name)} (threads ${ort.env.wasm.numThreads}, isolated ${self.crossOriginIsolated})`);
    const s = await ort.InferenceSession.create(await modelBytes(name), { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    log(`model ${role(name)} ready`);
    return s;
  },
  release(sess) { try { sess.release(); } catch { /* already gone */ } },
  tensor(type, bytes, dims) {
    const b = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const A = { float32: Float32Array, float16: Uint16Array, int64: BigInt64Array, int32: Int32Array, bool: Uint8Array, uint8: Uint8Array, int8: Int8Array }[type];
    return new ort.Tensor(type, new A(b), dims);
  },
  bytes(t) { const d = t.data; return new Uint8Array(d.buffer, d.byteOffset, d.byteLength); },
  async run(sess, feeds, names) { const r = await sess.run(feeds, names); return names.map((n) => r[n]); },
};

let pyReady = null;
async function setup() {
  progress('Starting the music reader', 0.01);
  log(`starting Python (memory ${navigator.deviceMemory ?? '?'} GB, cores ${navigator.hardwareConcurrency})`);
  const py = await loadPyodide({ indexURL: base + 'pyodide/', stderr, stdout: stderr });
  progress('Starting the music reader', 0.04);
  log('loading numpy, opencv, pillow');
  await py.loadPackage(['numpy', 'opencv-python', 'pillow'], { messageCallback: () => {}, errorCallback: log });
  log('loading homr');
  const zip = new Uint8Array(await (await fetch(base + 'homr.zip')).arrayBuffer());
  py.unpackArchive(zip, 'zip', { extractDir: '/lib/python3.13/site-packages' });
  py.FS.mkdirTree('/lib/python3.13/site-packages/onnxruntime');
  py.FS.writeFile('/lib/python3.13/site-packages/onnxruntime/__init__.py', await (await fetch(base + 'ortshim.py')).text());
  await py.runPythonAsync(`
import sys, types
for m in ('rapidocr', 'pypdfium2', 'requests'):
    sys.modules[m] = types.ModuleType(m)
sys.modules['rapidocr'].RapidOCR = object
lz = types.ModuleType('lzma'); lz.LZMAError = Exception; sys.modules['lzma'] = lz  # only homr's cache uses it
import homr.segmentation.config as sc, homr.transformer.configs as tc
for p in (sc.segnet_path_onnx, tc.default_config.filepaths.encoder_path, tc.default_config.filepaths.decoder_path):
    open(p, 'w').close()   # placeholders: ortshim loads the real models from Cache Storage
import homr.main
`);
  progress('Starting the music reader', 0.08);
  return py;
}

async function scan(pages) {
  const t0 = performance.now();
  pyReady ??= setup();
  const py = await pyReady;
  const n = pages.length;
  for (const f of py.FS.readdir('/tmp')) if (/^page/.test(f)) py.FS.unlink('/tmp/' + f);
  const paths = pages.map((bytes, i) => { const p = `/tmp/page${String(i + 1).padStart(3, '0')}.png`; py.FS.writeFile(p, bytes); return p; });
  // Progress from homr's own log lines: each page is found ("Processing"), then each staff is read.
  let page = 0, staves = 0;
  const pageShare = 0.9 / n;
  const at = (inPage) => 0.08 + pageShare * (page - 1 + inPage);
  const where = () => (n > 1 ? `page ${page} of ${n}` : 'the page');
  onStderr = (line) => {
    let m;
    if (line.startsWith('Processing ')) { page += 1; staves = 0; progress(`Finding the staves on ${where()}`, at(0.02)); }
    else if ((m = line.match(/^Found \d+ connected staffs.*\[(.*)\]/))) {
      staves = m[1].split(',').reduce((a, b) => a + (parseInt(b, 10) || 0), 0);
      progress(`Reading ${where()}`, at(0.45));
    } else if ((m = line.match(/^Running TrOmr inference on staff image (\d+)/)) && staves) {
      const k = +m[1] + 1;
      progress(`Reading line ${Math.min(k, staves)} of ${staves} on ${where()}`, at(0.45 + 0.5 * (k - 1) / staves));
    }
  };
  py.globals.set('stm_paths', py.toPy(paths));
  await py.runPythonAsync(`
import sys
sys.argv = ['homr', '--no-title', *stm_paths]
try:
    homr.main.main()
except SystemExit as e:
    if e.code:
        raise RuntimeError('The music could not be read.')
`);
  const out = paths[0].replace(/\.png$/, '.musicxml');
  if (!py.FS.analyzePath(out).exists) throw new Error('No music could be recognised. Try a sharper, flatter, well-lit photo.');
  const xml = py.FS.readFile(out, { encoding: 'utf8' });
  py.FS.unlink(out);
  return { xml, seconds: (performance.now() - t0) / 1000 };
}

self.onmessage = async (e) => {
  // fewer threads use less memory: the app asks for 1 when a first try ran out (must be set
  // before the first network is loaded)
  if (e.data?.threads && !sessionsMade) ort.env.wasm.numThreads = Math.min(ort.env.wasm.numThreads, e.data.threads);
  if (e.data?.type === 'warm') {   // start Python while the app is still preparing the pages
    pyReady ??= setup();
    pyReady.catch(() => { pyReady = null; });
    return;
  }
  if (e.data?.type !== 'scan') return;
  try {
    if (typeof WebAssembly.Suspending !== 'function') {
      throw new Error("This browser can't run the music reader. Use Chrome 137 or newer, or scan with the engine on your PC.");
    }
    const r = await scan(e.data.pages.map((p) => new Uint8Array(p)));
    post({ type: 'done', ...r });
  } catch (err) {
    post({ type: 'error', message: String(err?.message || err).split('\n').filter(Boolean).slice(-1)[0] });
  }
};
