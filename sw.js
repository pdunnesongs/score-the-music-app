// Score The Music service worker: keeps the app itself (code, music font, sound bank)
// so it opens with no connection. Saved songs already live in the device's IndexedDB.
// Calls to the engine (transcribe, export, scan, health) are never cached.
const CACHE = 'stm-app-v5';
const OMR_RUNTIME = 'stm-omr-runtime';   // kept by src/offline/phoneScan.ts, never versioned
const SCOPE = new URL(self.registration.scope).pathname;           // e.g. /app/

self.addEventListener('install', (e) => {
  e.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    const shell = ['./', 'manifest.webmanifest', 'icons/icon-192.png', 'icons/icon-512.png',
      'font/Bravura.woff2', 'font/Bravura.woff', 'fonts/PetalumaScript.otf', 'soundfont/sonivox.sf2'];
    await Promise.allSettled(shell.map((u) => cache.add(u)));
    // every hashed bundle, including chunks loaded later (asset-list.json is written by the build),
    // and anything else named in index.html
    try {
      const html = await (await fetch('./', { cache: 'no-store' })).text();
      const urls = new Set([...html.matchAll(/(?:src|href)="(\.\/assets\/[^"]+)"/g)].map((m) => m[1]));
      try { for (const u of await (await fetch('asset-list.json', { cache: 'no-store' })).json()) urls.add(u); } catch { /* older build */ }
      await Promise.allSettled([...urls].map((u) => cache.add(u)));
    } catch { /* offline during install: runtime caching fills in later */ }
    self.skipWaiting();
  })());
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    // only old copies of the app: the phone's music reader models (stm-omr-models) stay
    for (const k of await caches.keys()) if (k.startsWith('stm-app-') && k !== CACHE) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  const url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== location.origin || !url.pathname.startsWith(SCOPE)) return;
  if (req.mode === 'navigate') {
    // the page: newest from the network, the saved copy when offline
    e.respondWith(fetch(req).then((r) => {
      const copy = r.clone(); caches.open(CACHE).then((c) => c.put('./', copy)); return isolate(r);
    }).catch(async () => isolate(await caches.match('./'))));
    return;
  }
  // The phone's music reader (omr/): newest from the network, else the copy that "Get phone
  // scanning ready" put in its own cache (stm-omr-runtime), which survives app updates.
  if (url.pathname.startsWith(SCOPE + 'omr/')) {
    e.respondWith(fetch(req).then(async (r) => {
      if (r.ok && (await caches.has(OMR_RUNTIME))) (await caches.open(OMR_RUNTIME)).put(req.url, r.clone());
      return isolate(r);
    }).catch(async () => isolate(await caches.match(req.url, { cacheName: OMR_RUNTIME, ignoreVary: true, ignoreSearch: true }))));
    return;
  }
  // everything else in the app: from the cache, refreshed in the background
  e.respondWith(caches.open(CACHE).then(async (cache) => {
    // ignoreVary: the engine sends 'Vary: Origin', and module scripts are requested with an Origin
    // header that the stored copy wasn't, so without it dynamic imports miss the cache offline
    const hit = await cache.match(req, { ignoreVary: true });
    const net = fetch(req).then((r) => { if (r.ok) cache.put(req, r.clone()); return r; }).catch(() => hit);
    return isolate(hit || await net);   // workers need the same headers as the page
  }));
});

// Reading sheet music on an Android phone (omr/worker.mjs) runs on all CPU cores only when the
// page is cross-origin isolated, so on Android every app response gets these headers.
// 'credentialless' still lets the app fetch from an engine on another address.
function isolate(r) {
  if (!r || !/Android/i.test(self.navigator.userAgent) || r.type === 'opaqueredirect') return r;
  const h = new Headers(r.headers);
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Embedder-Policy', 'credentialless');
  return new Response(r.body, { status: r.status, statusText: r.statusText, headers: h });
}
