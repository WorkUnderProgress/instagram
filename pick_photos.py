"""Pick carousel photos in the browser: paste a folder, tick the scoring models (time shown), watch it score, pick.

One file, Windows or Mac. Once: python3 -m pip install pyiqa==0.1.16 aesthetic-predictor-v2-5 open_clip_torch pillow-heif certifi "setuptools<81"
Then: python3 pick_photos.py [FOLDER] (Windows: python instead of python3)
Options: [--out DIR] [--port N] [--no-browser] [--similar 0.83]. The browser page opens by itself.
Submit copies the picked originals to --out (default FOLDER/Picked_<date>); Quit writes nothing.
"""
import argparse, datetime, http.server, io, json, os, pathlib, re, shutil, sys, threading, time, urllib.parse, webbrowser
import cv2, numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

if sys.stdout is None: # pythonw: no console, so libraries that print must not crash
    sys.stdout = sys.stderr = open(os.devnull, "w")
register_heif_opener() # iPhone / Mac HEIC photos
os.environ.setdefault("SSL_CERT_FILE", __import__("certifi").where()) # model downloads work without the Mac "Install Certificates" step
SIMILAR, MIN_SHARP, DARK, BRIGHT = 0.83, 40, 30, 230 # stack cosine; hide below/above these
MODELS = { # key: (name, AVA SRCC, s/photo, load s on this laptop), best first
    "topiq_iaa": ("TOPIQ-IAA", 0.791, 1.7, 15), "musiq-ava": ("MUSIQ-AVA", 0.745, 0.8, 5), "nima": ("NIMA", 0.712, 0.4, 10),
    "v25": ("Aesthetic V2.5 (SigLIP)", 0.710, 5.3, 60), "laion_aes": ("LAION V2 (ViT-L/14)", 0.665, 1.5, 30)}
BASE = ("CLIP B/32: stacks, sharpness (always on)", None, 0.15, 40)
CACHE = ".pick_photos_cache.npz"
NAMES = {m: v[0] for m, v in ({"base": BASE} | MODELS).items()}
S, stop, worker, picks, root, files, keys, cache, order = {"n": 0, "N": 0, "done": False, "errors": {}, "models": []}, threading.Event(), None, None, None, [], [], {}, []
clean = lambda s: pathlib.Path(s.strip().strip('"\'').strip()).expanduser()
photos = lambda f: sorted(p for p in f.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")
                          and not any(d.startswith(("Picked_", ".")) for d in p.relative_to(f).parts[:-1]))
key = lambda p: (p.parent, f"{p.name}|{p.stat().st_size}|{p.stat().st_mtime_ns}")
missing = lambda c, k, m: m not in c.get(k, {}) or (m != "base" and np.isnan(c[k][m]))
group = lambda i: " - ".join(files[i].parent.relative_to(root).parts) # "" for the folder itself


def load(p, size):
    im = Image.open(p)
    im.draft("RGB", (size, size))
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((size, size))
    return im


def shot(p): # shooting order: EXIF capture time, then natural file-name order (IMG_5 < IMG_6 < IMG_10)
    try:
        t = Image.open(p).getexif().get_ifd(0x8769).get(36867) or ""
    except Exception: # unreadable photo: sorted by name only, reported when it is scored
        t = ""
    return t, [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def read_cache(fs): # {(folder, name|size|mtime): {"base": [sharp, bright, *embedding], model: score}}
    c = {}
    for d in {p.parent for p in fs}:
        z = dict(np.load(d / CACHE)) if (d / CACHE).is_file() else {} # dict: each array is read once
        if "base" in z: # older cache files have no per-model scores: rebuilt
            c |= {(d, k): {m: z[m][j] for m in z if m != "keys"} for j, k in enumerate(z["keys"])}
    return c


def write_cache():
    for d in {k[0] for k in keys}:
        ks = [k for k in keys if k[0] == d and "base" in cache.get(k, {})]
        ks and np.savez(d / CACHE, keys=np.array([k[1] for k in ks]), base=np.stack([cache[k]["base"] for k in ks]),
                        **{m: np.array([cache[k].get(m, np.nan) for k in ks]) for m in MODELS})


def count(path):
    f = clean(path)
    if not path.strip() or not f.is_dir():
        return {"error": "not a folder"}
    fs = photos(f)
    if not fs:
        return {"error": "no photos"}
    c, ks = read_cache(fs), [key(p) for p in fs]
    return {"photos": len(fs), "folders": len({p.parent for p in fs}),
            "models": [{"key": m, "name": n, "ava": ava, "s": s, "load": l, "todo": int(sum(missing(c, k, m) for k in ks))}
                       for m, (n, ava, s, l) in {"base": BASE, **MODELS}.items()]}


def make(m): # model key -> f(PIL image at 1024 px) -> score (or the base vector)
    import torch
    if m == "base":
        import open_clip
        net, _, pre = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        net.eval()

        def f(im):
            im = im.copy()
            im.thumbnail((640, 640))
            g = np.asarray(im.convert("L"), dtype=np.float32)
            e = net.encode_image(pre(im)[None])
            return np.r_[cv2.Laplacian(g, cv2.CV_32F).var(), g.mean(), (e / e.norm(dim=-1, keepdim=True))[0].numpy()]
        return f
    if m == "v25":
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        net, pre = convert_v2_5_from_siglip(low_cpu_mem_usage=True, trust_remote_code=True)
        net.eval()
        return lambda im: float(net(pre(images=im, return_tensors="pt").pixel_values).logits.squeeze())
    import pyiqa
    net = pyiqa.create_metric(m, device="cpu")
    return lambda im: float(net(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)[None]))


def run(folder, ms): # background: score every photo in shooting order with CLIP + the ticked models
    global root, files, keys, cache, order
    import torch
    root, files = folder, photos(folder)
    keys, cache = [key(p) for p in files], read_cache(files)
    order = sorted(range(len(files)), key=lambda i: shot(files[i]))
    S.update(N=len(files), models=ms)
    f = {}
    for m in ms + ["base"]:
        if any(missing(cache, k, m) for k in keys):
            try:
                f[m] = make(m)
            except Exception as e:
                S["errors"][m] = f"{NAMES[m]} failed to load: {e}"
    if "base" in S["errors"] or not set(ms) - set(S["errors"]):
        return S.update(done=True)
    work, t0, worked = sum(any(missing(cache, k, m) for m in f) for k in keys), time.time(), 0
    try:
        with torch.inference_mode():
            for n, i in enumerate(order):
                if stop.is_set():
                    break
                need = [m for m in f if missing(cache, keys[i], m) and m not in S["errors"]]
                try:
                    im = need and load(files[i], 1024)
                except Exception:
                    S["errors"][files[i].name], need = f"could not read {files[i].name}: skipped", []
                for m in need:
                    try:
                        cache.setdefault(keys[i], {})[m] = f[m](im)
                    except Exception as e:
                        S["errors"][m] = f"{NAMES[m]} failed on {files[i].name}: {e}"
                worked += bool(need)
                S["left_s"] = (time.time() - t0) / max(worked, 1) * (work - worked)
                S["n"] = n + 1
                n % 20 == 19 and write_cache()
    except Exception as e: # never leave the page waiting on a dead thread
        S["errors"]["run"] = f"scoring stopped: {e}"
    finally:
        write_cache()
        S.update(done=True, left_s=0)


def state():
    idx, ok = order[:S["n"]], [m for m in S["models"] if m not in S["errors"]]
    out = {k: S.get(k, 0) for k in ("n", "N", "done", "left_s")} | {"errors": list(S["errors"].values()), "stacks": [], "hidden": []}
    idx = [i for i in idx if "base" in cache.get(keys[i], {})]
    if not idx:
        return out
    b = lambda i: cache[keys[i]]["base"]
    Z = np.array([[cache[keys[i]].get(m, np.nan) for m in ok] for i in idx]).reshape(len(idx), len(ok))
    with np.errstate(all="ignore"): # z-score each model over the photos scored so far, then average the models
        score = dict(zip(idx, np.nan_to_num(np.nanmean((Z - np.nanmean(Z, 0)) / np.nanstd(Z, 0), 1)) if ok else np.zeros(len(idx))))
    hid = lambda i: b(i)[0] < MIN_SHARP or not DARK <= b(i)[1] <= BRIGHT
    stacks = []
    for i in idx: # shooting order; a shot that looks like the one just before it joins its stack
        if not hid(i):
            stacks[-1].append(i) if stacks and b(i)[2:] @ b(stacks[-1][-1])[2:] >= a.similar else stacks.append([i])
    if S["done"]:
        stacks = sorted((sorted(s, key=lambda i: -score[i]) for s in stacks), key=lambda s: -score[s[0]])
    info = lambda i: {"i": int(i), "name": files[i].name, "group": group(i),
                      "label": f"{score[i]:+.1f}" + (" dark" if b(i)[1] < DARK else " bright" if b(i)[1] > BRIGHT else " blurry" if b(i)[0] < MIN_SHARP else "")}
    return out | {"stacks": [[info(i) for i in s] for s in stacks], "hidden": [info(i) for i in idx if hid(i)]}


class H(http.server.BaseHTTPRequestHandler):
    log_message = lambda *x: None

    def send(self, body, ctype="application/json", cache=False):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        cache and self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

    def do_GET(self):
        u, S["seen"] = urllib.parse.urlparse(self.path), time.time()
        if u.path.startswith("/img/"):
            b = io.BytesIO()
            load(files[int(u.path[5:])], 600).save(b, "JPEG", quality=85)
            return self.send(b.getvalue(), "image/jpeg", cache=True)
        if u.path in ("/count", "/state"):
            return self.send(count(urllib.parse.parse_qs(u.query).get("path", [""])[0]) if u.path == "/count" else state())
        page = PAGE
        self.send(page.replace("__DATA__", json.dumps({"path": a.folder, "started": worker is not None})).encode(), "text/html; charset=utf-8")

    def do_POST(self):
        global worker, picks, out
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/start":
            if worker is None:
                worker = threading.Thread(target=run, args=(clean(body["path"]), body["models"]), daemon=True)
                worker.start()
            return self.send("ok")
        picks, out = body, a.out or (root or clean(a.folder)) / f"Picked_{datetime.date.today()}"
        stop.set()
        if picks:
            out.mkdir(parents=True, exist_ok=True)
            for i in picks:
                shutil.copy2(files[i], out / " - ".join(filter(None, [group(i), files[i].name])))
        self.send({"msg": f"Copied {len(picks)} photo(s) to {out}." if picks else "Nothing written."})
        threading.Thread(target=server.shutdown).start()


PAGE = r"""<!doctype html><meta charset="utf-8"><title>pick photos</title>
<style>
body{font:14px system-ui,sans-serif;margin:0;background:#111;color:#eee}
header{position:sticky;top:0;z-index:1;background:#222;padding:10px 14px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
section{padding:24px}input[type=text]{width:min(90vw,800px);padding:8px;font:inherit}label{display:block;margin:8px 0;cursor:pointer}
.grid{display:flex;flex-wrap:wrap;gap:10px;padding:12px;align-items:flex-start}
.stack{background:#1c1c1c;border-radius:8px;padding:5px}
.stack .more{display:none}.stack.open .more{display:inline-block}.stack.open{outline:1px solid #555}
.t{position:relative;display:inline-block;margin:2px;cursor:pointer;vertical-align:top}
.t img{height:200px;display:block;border:4px solid transparent;border-radius:5px}
.t.on img{border-color:#3c9}
.t span{position:absolute;left:6px;bottom:6px;background:#000b;padding:1px 5px;border-radius:3px;font-size:11px}
.stack .t:first-child span::before{content:"\2605 "}
button{padding:6px 14px;cursor:pointer}.stack>button{display:block;margin:4px auto 0}
summary{padding:0 14px;cursor:pointer;color:#aaa}progress{width:240px}
</style>
<header><span id="run" hidden><b id="n">0 picked</b> <button onclick="send(1)">Submit selection</button> <progress id="pb" max="1" value="0"></progress>
</span>
<button onclick="send(0)">Quit</button><span id="s" style="color:#aaa"></span><span id="er" style="color:#f66"></span></header>
<section id="s1"><h2>Folder to pick photos from</h2><input type="text" id="path" placeholder="paste the folder path" autofocus>
<p id="cnt" style="color:#aaa"></p><button id="nx" disabled onclick="models()">Next</button></section>
<section id="s2" hidden><h2>Scoring models (best first)</h2><div id="ml"></div><p id="tot"></p><button id="go" onclick="start()">Start</button></section>
<div class="grid" id="g"></div>
<details id="hd" hidden><summary>hidden: blurry, dark or blown out (<span id="hn">0</span>)</summary><div class="grid" id="h"></div></details>
<script>
const D = __DATA__, picked = new Set(), opened = new Set(), T = {};
let C;
const mins = s => s < 90 ? Math.round(s) + ' s' : Math.round(s / 60) + ' min';
const cost = m => m.todo ? m.load + m.s * m.todo : 0;
const ticked = () => [...ml.querySelectorAll('input:checked')].map(x => x.value);
path.oninput = () => { clearTimeout(path.t); nx.disabled = true; path.t = setTimeout(check, 300); };
async function check() {
  C = await (await fetch('/count?path=' + encodeURIComponent(path.value))).json();
  cnt.textContent = C.error || `${C.photos} photos in ${C.folders} folder(s)`; nx.disabled = !!C.error;
}
function models() {
  s1.hidden = true; s2.hidden = false;
  ml.innerHTML = C.models.map((m, k) => `<label><input type="checkbox" value="${m.key}"${k < 4 ? ' checked' : ''}${k ? '' : ' disabled'}>
    ${m.name}${m.ava ? ' · AVA ' + m.ava : ''} · ${m.todo ? '~' + mins(cost(m)) : 'all cached'}</label>`).join('');
  ml.onchange = total; total();
}
function total() {
  tot.textContent = 'Total: ~' + mins(C.models.filter(m => ticked().includes(m.key)).reduce((t, m) => t + cost(m), 0));
  go.disabled = ticked().length < 2;
}
async function start() {
  go.disabled = true;
  await fetch('/start', {method: 'POST', body: JSON.stringify({path: path.value, models: ticked().filter(k => k != 'base')})});
  s1.hidden = s2.hidden = true; run.hidden = hd.hidden = false; poll();
}
function tile(p, cls) { // one element per photo, reused on every redraw so images never reload
  const d = T[p.i] ||= Object.assign(document.createElement('div'), {title: (p.group ? p.group + ' / ' : '') + p.name,
    innerHTML: `<img loading="lazy" src="/img/${p.i}"><span></span>`,
    onclick: () => { picked.has(p.i) ? picked.delete(p.i) : picked.add(p.i); d.classList.toggle('on'); n.textContent = picked.size + ' picked'; }});
  d.className = 't ' + cls + (picked.has(p.i) ? ' on' : ''); d.lastChild.textContent = p.label;
  return d;
}
async function poll() {
  const S = await (await fetch('/state')).json();
  pb.max = S.N || 1; pb.value = S.n; er.textContent = S.errors.join(' · ');
  s.textContent = S.done ? `${S.n} photos, ${S.stacks.length} stacks, ${S.hidden.length} hidden · click a photo to pick it, +N opens its stack`
                         : S.n ? `${S.n} / ${S.N} photos scored · ~${mins(S.left_s)} left · you can pick already` : 'loading models (can take a few minutes)...';
  g.replaceChildren(...S.stacks.map(st => {
    const box = document.createElement('div'), id = Math.min(...st.map(p => p.i)), more = () => '+' + (st.length - 1);
    box.className = 'stack' + (opened.has(id) ? ' open' : '');
    box.append(...st.map((p, k) => tile(p, k ? 'more' : '')));
    if (st.length > 1) box.append(Object.assign(document.createElement('button'), {textContent: opened.has(id) ? 'close' : more(),
      onclick: e => { opened.has(id) ? opened.delete(id) : opened.add(id); box.classList.toggle('open'); e.target.textContent = opened.has(id) ? 'close' : more(); }}));
    return box;
  }));
  h.replaceChildren(...S.hidden.map(p => tile(p, ''))); hn.textContent = S.hidden.length;
  if (!S.done) setTimeout(poll, 1000);
}
async function send(ok) {
  if (ok && !picked.size) return alert('Pick at least one photo first.');
  const r = await (await fetch('/done', {method: 'POST', body: JSON.stringify(ok ? [...picked] : null)})).json();
  document.body.innerHTML = `<p style="padding:24px">${r.msg} You can close this tab.</p>`;
}
setInterval(() => fetch('/state'), 10000); // keeps the script alive while this tab is open
path.value = D.path;
if (D.started) { s1.hidden = true; run.hidden = hd.hidden = false; poll(); }
else if (D.path) check().then(() => C.error || models());
</script>
"""


def watchdog(): # tab closed without Submit/Quit: stop, so no invisible pythonw is left behind (open pages ping every 10 s)
    while time.time() - S["seen"] < 180:
        time.sleep(10)
    stop.set()
    server.shutdown()


if __name__ == "__main__": # the model processes import this file too; only the main one runs the page
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default="", help="or paste it in the page")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--port", type=int, default=0, help="default: any free port, so two pickers never share one")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--similar", type=float, default=SIMILAR, help=f"stack photos at least this alike, 0-1 (default {SIMILAR}; lower = bigger stacks)")
    a = ap.parse_args()
    http.server.ThreadingHTTPServer.allow_reuse_address = False # on Windows it lets a second picker share the port
    server = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H)
    a.no_browser or webbrowser.open(f"http://127.0.0.1:{server.server_port}/")
    S["seen"] = time.time()
    threading.Thread(target=watchdog, daemon=True).start()
    print(f"picker: http://127.0.0.1:{server.server_port}/", file=sys.stderr, flush=True)
    server.serve_forever()
    worker and worker.join() # the cache is saved and the model processes are stopped
    if picks:
        print(out)
    sys.exit(0 if picks else 1)