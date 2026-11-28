"""Template reel: an Instagram Reel / YouTube Short's cuts + audio, filled with your own footage. One file, Windows or Mac, CPU only.
    python template_reel.py [--port N] [--no-browser] [--cookies-from-browser safari]        (Mac: python3.14 template_reel.py)
The first run installs the packages it needs into this Python; work is kept next to this file (templates/, .cache/). Needs ffmpeg 7+
(+ deno for YouTube Shorts). The page takes the link, your footage folder, the scoring models, tries and threshold; each shot gets
random clips of random videos until one clears the threshold (else the best, marked low), rendered under the reel's audio. The viewer
replaces clips; Save reel writes <footage folder>/template_reel_output/<reel id>/ (the reel + one clip per shot)."""
import argparse, datetime, hashlib, importlib.util, json, os, random, re, shutil, statistics, subprocess, sys, threading, time, traceback, types, webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote

NEED = {"numpy": "numpy==2.5.3", "cv2": "opencv-python==5.0.0.93", "scenedetect": "scenedetect==0.7.1", "yt_dlp": "yt-dlp==2026.8.19", "torch": "torch==2.14.1",
        "torchvision": "torchvision==0.29.1", "open_clip": "open_clip_torch==3.3.0", "pyiqa": "pyiqa==0.1.16", "transformers": "transformers==5.22.0", "timm": "timm==1.0.30",
        "einops": "einops==0.8.2", "yaml": "pyyaml==6.0.3", "aesthetic_predictor_v2_5": "aesthetic-predictor-v2-5==2024.12.18.1", "certifi": "certifi==2026.7.22",
        "pkg_resources": "setuptools==80.10.2"}  # pyiqa's LAION needs pkg_resources (gone in setuptools 81)
if missing := [p for m, p in NEED.items() if not importlib.util.find_spec(m)]:  # first run: install what's missing
    print("First run: installing ", ", ".join(missing), "(a few minutes)...", flush=True); subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
import numpy as np

os.environ.setdefault("SSL_CERT_FILE", __import__("certifi").where())  # downloads work without the Mac "Install Certificates" step
ROOT = Path(__file__).resolve().parent; TEMPLATES, CACHE = ROOT / "templates", ROOT / ".cache"; MODELS_DIR = CACHE / "models"
os.environ.setdefault("HF_HOME", str(MODELS_DIR)); os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
THRESHOLD, TRIES, MAX_TRIES = 0.40, 10, 15  # page defaults: a clip must score at least THRESHOLD (kept low on purpose); tries slider 1..MAX_TRIES
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mts", ".3gp"}
OUTPUT = "template_reel_output"  # Save reel writes here, inside the footage folder; never searched for footage
# cuts: a colour-grid jump (mean 8x8 change > 12, and 3x the median of 8 frames either side), PySceneDetect content (27), or a wipe
# (a frame with 20+ pixels 40 off its median grey at 144x256, then 2 nearly blank frames: typed text cleared); 1 frame apart = one cut
CUT = {"grid_min_diff": 12.0, "grid_ratio": 3.0, "grid_window": 8, "content_threshold": 27.0, "wipe_detail": 40, "wipe_min_px": 20, "wipe_blank_px": 5, "merge_frames": 1, "min_shot_s": 0.5}
DOVER_CODE, DOVER_URL = "https://github.com/VQAssessment/DOVER/archive/refs/heads/master.zip", "https://github.com/QualityAssessment/DOVER/releases/download/"  # S-Lab licence: non-commercial
LAION_HEAD = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth"
ESRGAN = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-"  # + x4v3.pth (denoises) / wdn-x4v3.pth (keeps the noise); BSD-3
RIFE = "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife426.pth"  # RIFE 4.26 (Practical-RIFE, MIT): Make 4K's in-between frames
TEXTDET = "https://huggingface.co/opencv/text_detection_ppocr/resolve/main/text_detection_en_ppocrv3_2023may.onnx"  # PP-OCRv3 text detector (OpenCV zoo, Apache-2.0): finds the template's on-video text
K4 = {"denoise": 0.3, "ai": 0.6, "grain": 3, "crf": 16}  # Make 4K's AI version: Real-ESRGAN denoise 0..1 (low keeps texture), AI share of the blend with the plain upscale, grain strength, x264 quality
LOCK, SCORE, TEXT, LOADED, COOKIES, DOVER_FRAMES, TEXT_ERR = threading.Lock(), threading.Lock(), threading.Lock(), {}, {}, {}, {}  # SCORE: one scoring job at a time (Slide's score, a replace); TEXT: one text search
THUMBS = CACHE / "thumbs"  # Choose a video's icons: <hash of path+size+mtime>_<duration s>.jpg
state = {"stage": "setup", "tries": TRIES, "threshold": THRESHOLD, "max_tries": MAX_TRIES}
S = {"metas": {}, "readers": {}}  # per process: each video's probe + reader, the reel being edited
STATS = CACHE / "template_reel_stats.json"; stats = json.loads(STATS.read_text()) if STATS.is_file() else {}
calibration = {k: {"mean": m, "sd": s} for k, (m, s) in {  # each model's typical raw score (mean, sd) on 16 clips; your own take over after 30 scores
    "current": (0.65035, 0.15103), "dover-mobile": (0.61001, 0.04063), "musiq-spaq": (62.14513, 16.57662), "musiq-ava": (4.85308, 0.49833),
    "laion_aes": (4.75119, 0.56368), "dover": (0.51219, 0.20494), "aes-v2.5": (4.55521, 0.56719)}.items()}
stamp, last = lambda: f"{datetime.datetime.now():%Y%m%d_%H%M%S}", lambda e: str(e).strip().splitlines()[-1]
busy = lambda msg: state.update(stage="working", msg=msg)


def note(k, x): stats[k] = (stats.get(k, []) + [round(x, 4)])[-1000:]  # timings and scores, for the estimate
def med(k, default): return statistics.median(stats[k]) if stats.get(k) else default
def save_stats(): STATS.parent.mkdir(parents=True, exist_ok=True); STATS.write_text(json.dumps(stats))


# ---------------------------------------------------------------- ffmpeg
def binary(name):  # on PATH, else where winget / Homebrew put it (a fresh install isn't on PATH yet)
    for p in (shutil.which(name), Path(os.environ.get("LOCALAPPDATA", "-")) / f"Microsoft/WinGet/Links/{name}.exe", f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"):
        if p and Path(p).is_file(): return str(p)
    raise RuntimeError(f"{name} not found: install ffmpeg (Mac: brew install ffmpeg · Windows: winget install Gyan.FFmpeg)")


def ff(args, desc="ffmpeg"):  # run ffmpeg quietly; on failure raise with the end of its log
    p = subprocess.run([binary("ffmpeg"), "-hide_banner", "-nostdin", "-y", *map(str, args)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode: raise RuntimeError(f"{desc} failed:\n" + "\n".join(p.stderr.strip().splitlines()[-15:]))


def probe(path):
    p = subprocess.run([binary("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode: raise RuntimeError(f"ffprobe failed on {path}: {p.stderr.strip()}")
    return json.loads(p.stdout)


def fraction(s): n, _, d = (s or "0").partition("/"); return float(n) / float(d) if d and float(d) else 0.0 if d else float(n)


def describe(path):  # a video as seen: rotation and pixel aspect applied; fps, length, HDR
    info = probe(path); v = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    if v is None: raise ValueError("no video stream")
    rot = next((round(float(sd["rotation"])) for sd in v.get("side_data_list", []) if "rotation" in sd), 0) or int(v.get("tags", {}).get("rotate", 0))
    n, _, d = str(v.get("sample_aspect_ratio", "")).partition(":"); sar = float(n) / float(d) if d and float(n) > 0 and float(d) > 0 else 1.0
    w, h = int(v["width"]), int(v["height"]); stored, w = [w, h], round(w * sar); w, h = (h, w) if rot % 180 else (w, h)
    r, avg = fraction(v.get("r_frame_rate")), fraction(v.get("avg_frame_rate"))
    return {"display_size": [w, h], "stored_size": stored, "sar": round(sar, 4), "fps": round(avg or r, 3), "transfer": v.get("color_transfer"), "codec": v.get("codec_name"), "pix_fmt": v.get("pix_fmt"),
            "duration_s": round(float(v.get("duration") or info["format"].get("duration") or 0), 3), "hdr": v.get("color_transfer") in ("arib-std-b67", "smpte2084")}


def normalise(m, fps, W=1080, H=1920):  # the frame as the reel shows it: HDR -> SDR, square pixels, constant fps, cover W x H, centre crop
    tm = [f"zscale=tin={m['transfer']}:min=bt2020nc:pin=bt2020:rin=tv:t=linear:npl=100", "format=gbrpf32le",
          "tonemap=hable:desat=0", "zscale=t=bt709:m=bt709:r=tv"] if m["hdr"] else []
    sq = ["scale='trunc(iw*sar/2)*2':ih", "setsar=1"] if abs(m["sar"] - 1) >= 0.01 else []
    return tm + sq + ["setpts=PTS-STARTPTS", f"fps={fps}", f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos", f"crop={W}:{H}", "setsar=1", "format=yuv420p"]


# ---------------------------------------------------------------- the template: download + cuts
def friendly(e):
    for pat, hint in [(r"age|not a bot|login required|empty media response|rate-limit", "needs a login: start this script with --cookies-from-browser safari (or chrome, edge)"),
                      (r"Private video", "the video is private"), (r"unavailable|not available", "the video is unavailable (removed, region-locked, or a bad link)"),
                      (r"Unsupported URL", "not a Reel / Short link"), (r"getaddrinfo|Failed to resolve|timed out|ConnectionError", "no internet connection, or the site is unreachable"),
                      (r"Requested format is not available|No video formats found", "no video offered: update yt-dlp (python -m pip install -U yt-dlp) and install deno")]:
        if re.search(pat, str(e), re.I): return f"{hint} ({last(e)})"
    return str(e)


def fetch(link):  # Reel / Short link, or a video file -> templates/<id>/video.mp4 + audio.wav + source.json (no repeat download)
    import yt_dlp
    if Path(link).is_file():
        info = {"id": re.sub(r"[^\w.-]+", "_", Path(link).stem)[:80], "title": Path(link).stem, "url": str(Path(link).resolve())}; d = TEMPLATES / info["id"]
        if not (d / "video.mp4").exists(): d.mkdir(parents=True, exist_ok=True); shutil.copyfile(link, d / "video.mp4")
    else:
        quiet = type("Quiet", (), {k: lambda self, m: None for k in ("debug", "info", "warning", "error")})()
        opts = {"quiet": True, "no_warnings": True, "logger": quiet, "noplaylist": True, "noprogress": True, "format": "bv*+ba/b", "merge_output_format": "mp4",
                "outtmpl": str(TEMPLATES / "%(id)s" / "video.%(ext)s"), "ffmpeg_location": str(Path(binary("ffmpeg")).parent), **COOKIES}
        try:
            with yt_dlp.YoutubeDL(opts) as y:
                info = y.extract_info(link, download=False); d = TEMPLATES / info["id"]
                if not (d / "video.mp4").exists(): y.process_ie_result(info, download=True)
        except yt_dlp.utils.DownloadError as e: raise RuntimeError(friendly(e)) from None
        (d / "caption.txt").write_text(info.get("description") or "", encoding="utf-8")  # the post's caption, for Copy caption / Save reel
        info = {k: info.get(k) for k in ("id", "title", "uploader", "webpage_url")} | {"url": link}
    if not (a := d / "audio.wav").exists() or a.stat().st_mtime < (d / "video.mp4").stat().st_mtime:
        s = {x["codec_type"]: x for x in reversed(probe(d / "video.mp4")["streams"])}
        if "audio" not in s: raise RuntimeError(f"no audio track in {d / 'video.mp4'}")
        ff(["-i", d / "video.mp4", "-vn", "-ar", 48000, "-ac", 2, "-c:a", "pcm_s16le", a], "audio extract")
        (d / "source.json").write_text(json.dumps(info | {"fps": fraction(s["video"]["avg_frame_rate"]), "size": [s["video"]["width"], s["video"]["height"]]}, indent=2, ensure_ascii=False), encoding="utf-8")
    return d


def cuts(video):  # three detectors: colour-grid jump, PySceneDetect content, wipe; a cut 2+ find, or a wipe, is sure
    import cv2
    from scenedetect import detect, ContentDetector
    c = CUT; cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS); grid, mass = [], []
    while (f := cap.read()[1]) is not None:
        grid.append(cv2.resize(f, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32))
        g = cv2.cvtColor(cv2.resize(f, (144, 256)), cv2.COLOR_BGR2GRAY); mass.append(int((cv2.absdiff(g, int(np.median(g))) > c["wipe_detail"]).sum()))
    if not grid: raise RuntimeError(f"cannot read video frames: {video}")
    d = np.r_[0, np.abs(np.diff(grid, axis=0)).mean(axis=(1, 2, 3))]; w = c["grid_window"]
    found = {"grid": {i for i in range(1, len(d)) if d[i] > c["grid_min_diff"] and d[i] == d[max(0, i - 2):i + 3].max()
                      and d[i] > c["grid_ratio"] * np.median(np.r_[d[max(1, i - w):i], d[i + 1:i + w + 1]])},
             "content": {s[0].frame_num for s in detect(str(video), ContentDetector(threshold=c["content_threshold"]))[1:]},
             "wipe": {i for i in range(1, len(mass) - 1) if mass[i - 1] >= c["wipe_min_px"] and max(mass[i:i + 2]) <= c["wipe_blank_px"]}}
    groups, out = [], []  # detections within merge_frames of each other are one cut
    for i in sorted(set().union(*found.values())): groups[-1].append(i) if groups and i - groups[-1][-1] <= c["merge_frames"] else groups.append([i])
    for grp in groups:
        by = sorted({k for k, s in found.items() for i in grp if i in s}); frame = next((i for i in grp if i in found["grid"] | found["wipe"]), grp[0])
        out.append({"frame": frame, "t": round(frame / fps, 3), "sure": len(by) > 1 or "wipe" in by, "by": by})
    floor_f = round(c["min_shot_s"] * fps)  # coalesce shots shorter than min_shot_s by dropping the less confident of the two bounding cuts (tie -> drop the right)
    while True:
        edges = [0, *(k["frame"] for k in out), len(grid)]
        b = next((i for i in range(len(edges) - 1) if edges[i + 1] - edges[i] < floor_f), None)
        if b is None or not out: break
        out.pop(b - 1 if b == len(out) or (b > 0 and not out[b - 1]["sure"] and out[b]["sure"]) else b)
    return {"video": Path(video).name, "fps": fps, "frames": len(grid), "min_shot_s": c["min_shot_s"], "cuts": out, "shots": [{"start_f": s, "end_f": e, "len_s": round((e - s) / fps, 3)} for s, e in zip(edges, edges[1:])]}


def caption(d):  # the template post's caption (saved when it's downloaded; an older template reads it again from its link)
    try:
        if not (f := d / "caption.txt").is_file() and (u := json.loads((d / "source.json").read_text(encoding="utf-8")).get("webpage_url")): fetch(u)
        return f.read_text(encoding="utf-8")
    except Exception: return ""


def template(link):  # the template folder + its cuts (detected again if the video or this script is newer, so CUT config changes bust the cache)
    with LOCK:
        d = fetch(link); cj = d / "cuts.json"
        if not cj.exists() or cj.stat().st_mtime < max((d / "video.mp4").stat().st_mtime, Path(__file__).stat().st_mtime): cj.write_text(json.dumps(cuts(d / "video.mp4"), indent=1))
        return d, json.loads(cj.read_text())


def text_layer(d, step=3):  # the template's on-video text, lifted off its frames (exact font, size, place, timing): its pieces as see-through PNGs
    import cv2  # in d/text/: text.json = {size, fps, sprites: [{png, x, y, w, h, frames: [[first, last], ...]}]} in the template's pixels and frames
    j, src = d / "text" / "text.json", d / "video.mp4"
    with TEXT:
        if found(d): return json.loads(j.read_text())
        if not (m := MODELS_DIR / Path(TEXTDET).name).is_file(): m.parent.mkdir(parents=True, exist_ok=True); __import__("urllib.request").request.urlretrieve(TEXTDET, m)
        net = cv2.dnn.TextDetectionModel_DB(cv2.dnn.readNet(str(m))); W, H = 736, 1312
        net.setBinaryThreshold(0.3); net.setPolygonThreshold(0.5); net.setUnclipRatio(2.0); net.setMaxCandidates(200); net.setInputSize((W, H))
        net.setInputMean((123.675, 116.28, 103.53)); net.setInputScale(1 / 255 / np.array([0.229, 0.224, 0.225]))
        def frames():
            cap = cv2.VideoCapture(str(src))
            while (f := cap.read()[1]) is not None: yield f
        dets, fps, cut = {}, cv2.VideoCapture(str(src)).get(cv2.CAP_PROP_FPS), [0] + [c["frame"] for c in json.loads((d / "cuts.json").read_text())["cuts"]]
        shot = lambda k: sum(c <= k for c in cut)  # text usually changes at a cut: a look never spans one (one caption's pieces are joined later)
        def sig(f, b):  # a box's bright strokes, where they are in the frame: the same letters as last time? (a font switch keeps the box, not the strokes)
            x0, y0 = max(0, int(b[0])), max(0, int(b[1])); c = f[y0:int(b[3]), x0:int(b[2])]; g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32); r = int(0.35 * (b[3] - b[1])) | 1
            return x0, y0, (g - cv2.morphologyEx(g, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))) > 40) & (c.min(2) > 150)
        def alike(s, t):  # the share of two sigs' strokes in the same place
            x0, y0 = min(s[0], t[0]), min(s[1], t[1]); m = np.zeros((2, max(s[1] + s[2].shape[0], t[1] + t[2].shape[0]) - y0, max(s[0] + s[2].shape[1], t[0] + t[2].shape[1]) - x0), bool)
            for i, (x, y, a) in enumerate((s, t)): m[i, y - y0:y - y0 + a.shape[0], x - x0:x - x0 + a.shape[1]] = a
            return (m[0] & m[1]).sum() / ((m[0] | m[1]).sum() + 1)
        for k, f in enumerate(frames()):  # 1. text boxes every step frames; tiny ones are signs in the footage
            h, w = f.shape[:2]
            if k % step == 0: dets[k] = [(b, sig(f, b)) for p in map(np.array, net.detect(cv2.resize(f, (W, H)))[0]) if (b := [p[:, 0].min() * w / W, p[:, 1].min() * h / H, p[:, 0].max() * w / W, p[:, 1].max() * h / H])[3] - b[1] > 0.018 * h]
        n = k + 1; ov = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        iou = lambda a, b: ov(a, b) / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ov(a, b) + 1e-9); tr = []
        for k, bs in dets.items():  # 2. lines: a box in about the same place on consecutive detections; a new look when it grows by a letter (typing) or at a cut
            for b, s in bs:
                t = next((t for t in tr if t["f1"] >= k - 3 * step and iou(t["looks"][-1]["b"], b) > 0.5), None)
                if not t: tr.append(t := {"f0": k, "looks": []})
                t["f1"] = k; L = t["looks"][-1] if t["looks"] else None
                if L and shot(L["z"]) == shot(k) and abs((b[2] - b[0]) - (L["b"][2] - L["b"][0])) < 0.3 * (b[3] - b[1]) and alike(s, L["s"]) > 0.5:
                    L["z"], L["s"], L["bz"] = k, s, b; L["mv"] = max(L["mv"], abs(b[1] + b[3] - L["b0"][1] - L["b0"][3]) / 2); L["b"] = [min(L["b"][0], b[0]), min(L["b"][1], b[1]), max(L["b"][2], b[2]), max(L["b"][3], b[3])]
                else: t["looks"].append({"a": k, "z": k, "b": b, "b0": b, "bz": b, "mv": 0, "s": s})  # (a font switching every few frames: a look per font)
        mid = lambda L: (L["b"][1] + L["b"][3]) / 2  # captions keep their line (a new font may sit a little higher or lower); signs in the footage move
        still = lambda t: max(L["mv"] for L in t["looks"]) < 0.01 * h and max(map(mid, t["looks"])) - min(map(mid, t["looks"])) < 0.5 * np.median([L["b"][3] - L["b"][1] for L in t["looks"]])
        looks = [dict(L, line=i) for i, t in enumerate(t for t in tr if t["f1"] - t["f0"] >= 0.4 * fps and still(t)) for L in t["looks"]]
        for L in looks: b, lh = L["b"], L["b"][3] - L["b"][1]; p = 0.3 * lh; L.update(r=[max(0, int(b[0] - p)), max(0, int(b[1] - p)), min(w, int(b[2] + p)), min(h, int(b[3] + p))], lh=lh, crops=[])
        for k, f in enumerate(frames()):  # 3. each look's letters, from up to 30 of its frames (and the ones up to the next detection with the same strokes: a font shown 3 frames gets all 3)
            for L in looks:
                if L["a"] <= k < L["z"] + step and shot(k) == shot(L["a"]) and (k - L["a"]) % (1 + (L["z"] - L["a"]) // 30) == 0 and (k <= L["z"] or alike(sig(f, L["bz"]), L["s"]) > 0.5): x0, y0, x1, y1 = L["r"]; L["crops"].append(f[y0:y1, x0:x1])
        fin = lambda L: L.update(core=(c := L["alpha"] > 0.6), ring=cv2.dilate(c.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2).astype(bool) & ~cv2.dilate(c.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
        for L in looks:  # bright thin strokes over a smooth background, that stay put while the scene behind them moves
            cs = np.stack(L.pop("crops")).astype(np.float32); M = np.median(cs, 0); G = cv2.cvtColor(M, cv2.COLOR_BGR2GRAY); r = int(0.35 * L["lh"]) | 1
            bg = cv2.morphologyEx(G, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))); top = G - bg
            L["raw"] = core = (top > 40) & (M.min(2) > 150) & ((cs.std(0).max(2) < 15) if len(cs) >= 5 else True); L["ok"] = core.sum() > 0.01 * core.size
            if not L["ok"]: continue
            br = core & (G > np.percentile(G[core], 95) - 20); seed = br & (np.abs(M - np.median(M[br], 0)).max(2) < 40)  # the text's own colour (white, cream, pale pink...): its brightest pixels and those like them
            thin = cv2.erode(core.astype(np.uint8), np.ones((3, 3), np.uint8))  # pries apart bits of the scene that only touch a letter
            nl, lab, st, cen = cv2.connectedComponentsWithStats(thin, connectivity=4); hits = np.bincount(lab[seed & thin.astype(bool)], minlength=nl); by = L["b"][1] - L["r"][1]
            good = [i for i in range(1, nl) if hits[i] >= 0.3 * st[i, cv2.CC_STAT_AREA] and by - 0.1 * L["lh"] <= cen[i][1] <= by + 1.1 * L["lh"] and st[i, cv2.CC_STAT_HEIGHT] <= 1.3 * L["lh"]]
            core = core & cv2.dilate(np.isin(lab, good).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)  # letters: shapes mostly in the text's colour, on its line
            if core.sum() <= 0.01 * core.size or max([st[i, cv2.CC_STAT_HEIGHT] for i in good], default=0) < 0.4 * L["lh"]: L["ok"] = False; continue  # (a line's letters fill its box: bright bits behind dark text don't)
            L["C"], CL = np.median(M[seed & core], 0), float(np.percentile(G[core], 90))
            L["alpha"] = np.clip(top / np.maximum(CL - bg, 1), 0, 1) * (cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0)
            fin(L); c = L["core"]; L["Cm"] = float(np.median(G[c]))
            L["ok"] = CL > 230 and (len(cs) < 3 or np.abs(cs.mean(3) - G)[:, c].mean() < 16)  # overlaid letters are solid and steady (a watermark is see-through)
        box = lambda A, B: [min(A["r"][0], B["r"][0]), min(A["r"][1], B["r"][1]), max(A["r"][2], B["r"][2]), max(A["r"][3], B["r"][3])]
        def put(L, k, r):  # L's k (raw / alpha) on the frame area r
            m = np.zeros((r[3] - r[1], r[2] - r[0]), L[k].dtype); m[L["r"][1] - r[1]:L["r"][3] - r[1], L["r"][0] - r[0]:L["r"][2] - r[0]] = L[k]; return m
        def same(A, B, both=False):  # the same letters in the same place: their strokes overlap (both: their cleaned letters, each mostly the other's: not a thin font inside a bold one)
            a, b = put(A, k := "core" if both else "raw", r := box(A, B)), put(B, k, r); return (a & b).sum() > 0.6 * (max if both else min)(a.sum(), b.sum())
        looks = [L for L in looks if L["ok"] and sum(M["z"] - M["a"] + step for M in looks if same(L, M)) < 0.5 * n]  # a handle watermark: the same letters half the reel or more
        joined = []
        for B in sorted(looks, key=lambda L: L["a"]):  # one caption over a cut: its pieces joined, keeping only the strokes seen over both scenes (a speck is one scene's)
            if A := next((A for A in joined if A["line"] == B["line"] and max(A["shots"]) + 1 == shot(B["a"]) and B["a"] - A["z"] <= step and same(A, B, True)), None):
                r = box(A, B); A.update(alpha=np.minimum(put(A, "alpha", r), put(B, "alpha", r)), raw=put(A, "raw", r) & put(B, "raw", r), r=r, z=B["z"], shots=A["shots"] | {shot(B["a"])}, Cm=(A["Cm"] + B["Cm"]) / 2); fin(A)
            else: joined.append(dict(B, shots={shot(B["a"])}))
        looks = joined
        for L in looks: L["frames"] = []
        for k, f in enumerate(frames()):  # 4. each frame, each line: its fullest look whose letters are really there (exact start and end: cuts, typing)
            best = {}
            for L in looks:
                if L["a"] - step <= k < L["z"] + step and shot(k) in L["shots"]:
                    x0, y0, x1, y1 = L["r"]; G = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32); c, g = np.median(G[L["core"]]), np.median(G[L["ring"]])
                    if c - g > 25 and abs(c - L["Cm"]) < 30 and L["core"].sum() > best.get(L["line"], {"core": np.zeros(1)})["core"].sum(): best[L["line"]] = L
            for L in best.values(): L["frames"].append(k)
        shutil.rmtree(j.parent, ignore_errors=True); j.parent.mkdir(parents=True); sprites = []
        for i, L in enumerate(L for L in looks if L["frames"]):  # imencode + write_bytes: cv2.imwrite can't take non-ASCII paths on Windows
            (j.parent / f"s{i}.png").write_bytes(cv2.imencode(".png", np.dstack([np.broadcast_to(L["C"], L["alpha"].shape + (3,)), L["alpha"] * 255]).round().astype(np.uint8))[1].tobytes()); runs = []
            for k in L["frames"]:
                if runs and k == runs[-1][1] + 1: runs[-1][1] = k
                else: runs.append([k, k])
            sprites.append({"png": f"s{i}.png", "x": L["r"][0], "y": L["r"][1], "w": L["r"][2] - L["r"][0], "h": L["r"][3] - L["r"][1], "frames": runs})
        j.write_text(json.dumps(t := {"size": [w, h], "fps": fps, "sprites": sprites})); return t


def found(d): j = d / "text" / "text.json"; return j.is_file() and j.stat().st_mtime >= max((d / "video.mp4").stat().st_mtime, Path(__file__).stat().st_mtime)  # (found again when this script is newer)


def dropped(d): f = d / "text" / "drop.json"; return set(json.loads(f.read_text())) if f.is_file() else set()  # the text pieces you unticked (gone with the pieces when they're found again)


def text_status(d):  # for the viewer: the text's pieces, or "working" while they're found in the background (once per template)
    if found(d): j = d / "text" / "text.json"; return {**json.loads(j.read_text()), "status": "ready", "v": int(j.stat().st_mtime), "drop": sorted(dropped(d))}
    if d.name in TEXT_ERR: return {"status": "error", "msg": TEXT_ERR.pop(d.name)}
    def run():
        try: text_layer(d)
        except Exception as e: traceback.print_exc(); TEXT_ERR[d.name] = last(e)
    if not TEXT.locked(): threading.Thread(target=run, daemon=True).start()
    return {"status": "working"}


def burn(d, on, W, H, first, pre="null", post="null"):  # the on-video text over a W x H video (when on; only the pieces you kept): (ffmpeg inputs, numbered from first; a filter graph [0:v] -> [v])
    t = text_layer(d) if on else {"size": [W, H], "fps": 1, "sprites": []}; (w, h), sp = t["size"], [p for p in t["sprites"] if p["png"] not in dropped(d)]; s = max(W / w, H / h); ox, oy = (w * s - W) / 2, (h * s - H) / 2
    g = f"[0:v]{pre}[v0];" + "".join(f"[{first + i}:v]scale=iw*{s:.5f}:-1[s{i}];[v{i}][s{i}]overlay={p['x'] * s - ox:.1f}:{p['y'] * s - oy:.1f}:format=auto:enable='"
                  + "+".join(f"between(t,{a / t['fps']:.4f},{(b + 0.5) / t['fps']:.4f})" for a, b in p["frames"]) + f"'[v{i + 1}];" for i, p in enumerate(sp))
    return sum((["-i", d / "text" / p["png"]] for p in sp), []), g + f"[v{len(sp)}]{post}[v]"


# ---------------------------------------------------------------- scoring models
class Reader:  # a clip's frames for scoring: 30 fps, 540x960, the reel's 9:16 crop. OpenCV in-process (an ffmpeg launch costs ~5 s); ffmpeg for HDR + HEVC (cv2's swscale on macOS can't convert 10-bit HEVC mid-stream)
    def __init__(self, video, meta):
        import cv2
        self.video, self.meta, self.cap = video, meta, None if meta["hdr"] or meta.get("codec") == "hevc" else cv2.VideoCapture(str(video))
        ok, f = self.cap.read() if self.cap is not None and self.cap.isOpened() else (False, None); (dw, dh), (sw, sh) = meta["display_size"], meta["stored_size"]
        if not ok or ((f.shape[1] > f.shape[0]) != (dw > dh) and dw != dh and sw != sh): self.cap and self.cap.release(); self.cap = None  # OpenCV turns rotated video itself; if not, ffmpeg decodes (release the failed cap so its ffmpeg threads go)
        self.fps = (self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 0) or meta["fps"] or 30.0
        print(f"reader: {Path(video).name} codec={meta.get('codec')} pix_fmt={meta.get('pix_fmt')} dur={meta['duration_s']:.1f}s path={'ffmpeg' if self.cap is None else 'cv2'}", flush=True)

    def frames(self, start, length, w=540, h=960):
        import cv2
        if self.cap is None:
            raw = subprocess.run([binary("ffmpeg"), "-v", "error", "-ss", f"{start:.3f}", "-i", str(self.video), "-t", f"{length:.3f}", "-an",
                                  "-vf", ",".join(normalise(self.meta, 30, w, h)[:-1]), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
            n = len(raw) // (w * h * 3); return np.frombuffer(raw[:n * w * h * 3], np.uint8).reshape(n, h, w, 3)
        every = max(1, round(self.fps / 30)); self.cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0); out = []
        for i in range(max(1, round(length * self.fps))):
            if not self.cap.grab(): break
            if i % every: continue
            ok, f = self.cap.retrieve()
            if not ok: break
            fh, fw = f.shape[:2]; s = max(w / fw, h / fh)  # cover w x h, centre crop
            f = cv2.resize(f, (max(w, round(fw * s)), max(h, round(fh * s))), interpolation=cv2.INTER_AREA); y, x = (f.shape[0] - h) // 2, (f.shape[1] - w) // 2
            out.append(cv2.cvtColor(f[y:y + h, x:x + w], cv2.COLOR_BGR2RGB))
        return np.stack(out) if out else np.zeros((0, h, w, 3), np.uint8)


def pick(f, n): return [f[i] for i in np.linspace(0, len(f) - 1, min(n, len(f))).round().astype(int)]


def measure(frames, fps):  # classic measures, each 0-1: sharpness, exposure, stability, motion, colour; + scene cut / frozen / brightness
    import cv2
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]; bright = np.array([g.mean() for g in grays])
    sharp = np.array([cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]); clipped = np.array([((g > 250) | (g < 5)).mean() for g in grays])
    hists = [cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_RGB2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256]) for f in frames]; flows, gx, gy, cut = [], [], [], []
    for i in range(1, len(grays)):
        a, b = (cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for g in grays[i - 1:i + 1])
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0); m = np.median(flow.reshape(-1, 2), axis=0)
        flows.append(float(np.linalg.norm(flow, axis=2).mean())); gx.append(m[0]); gy.append(m[1]); cut.append(cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA))
    s = np.clip((np.log(np.median(sharp) + 1e-6) - np.log(15)) / (np.log(400) - np.log(15)), 0, 1); b = float(np.median(bright))
    exposure = float(np.clip(1 - max(60 - b, b - 200, 0) / 60 - 2 * float(np.mean(clipped)), 0, 1)); jitter = 0.0
    if len(gx) >= 3:  # shake: camera motion around its own ~0.5 s average (a smooth pan isn't shake)
        k = max(3, round(fps * 0.5) | 1); sm = lambda x: np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
        jitter = float(np.mean(np.hypot(np.array(gx) - sm(np.array(gx)), np.array(gy) - sm(np.array(gy)))))
    m = float(np.median(flows)) if flows else 0.0; rgb = frames.astype(np.float32)
    rg, yb = rgb[..., 0] - rgb[..., 1], 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    return {"sharpness": float(s), "exposure": exposure, "stability": float(np.clip(1 - jitter / 3.0, 0, 1)),
            "motion": max(float(np.clip(m, 0, 1)) * (1 - np.clip((m - 6) / 6, 0, 1)), 0.25),  # a still shot is calm, not worthless
            "colour": float(np.clip((np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())) / 90.0, 0, 1)),
            "scene_cut": bool(cut) and max(cut) > 0.55, "frozen": bool(flows) and max(flows) < 0.02 and float(np.std(bright)) < 0.5, "brightness": b}


def current():  # "Current": CLIP ViT-B/32 (LAION aesthetic head + zero-shot interest) + classic measures, at ~8 fps 270x480
    import torch, open_clip
    from huggingface_hub import constants as hf
    from PIL import Image
    pos = ["an eye-catching, interesting photo", "a beautiful, vibrant photo full of life", "a stunning scenic photo", "a photo of people having fun"]
    neg = ["a boring, dull photo", "a blurry, out of focus photo", "a dark, underexposed photo", "a photo of nothing in particular", "a black screen"]
    hf.HF_HUB_OFFLINE = True  # cached weights; download only when missing
    try: model, _, prep = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    except Exception: hf.HF_HUB_OFFLINE = False; model, _, prep = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    model.eval(); head = MODELS_DIR / "laion_aesthetic_vit_b_32_linear.pth"
    if not head.is_file(): head.parent.mkdir(parents=True, exist_ok=True); torch.hub.download_url_to_file(LAION_HEAD, str(head))
    head = torch.load(head, map_location="cpu", weights_only=True)
    with torch.inference_mode(): t = model.encode_text(open_clip.get_tokenizer("ViT-B-32-quickgelu")(pos + neg)); text = t / t.norm(dim=-1, keepdim=True)
    squares = lambda f: [f[o:o + f.shape[1]] for o in np.linspace(0, f.shape[0] - f.shape[1], 2).round().astype(int)]  # 9:16 -> 2 overlapping squares
    def score(f):
        if len(f := f[::4, ::2, ::2]) < 2: return 0.0
        m = measure(f, 8); x = torch.stack([prep(Image.fromarray(np.ascontiguousarray(t))) for g in pick(f, 4) for t in squares(g)])
        with torch.inference_mode():
            e = model.encode_image(x); e = e / e.norm(dim=-1, keepdim=True)
            aes, interest = float((e @ head["weight"].T + head["bias"]).mean()), float((100.0 * e @ text.T).softmax(dim=-1)[:, :len(pos)].sum(dim=1).mean())
        total = (0.30 * np.clip((aes - 3.5) / 3.0, 0, 1) + 0.25 * interest + 0.25 * (m["sharpness"] + m["exposure"] + m["stability"]) / 3
                 + 0.12 * m["motion"] + 0.08 * m["colour"]) * (0.5 if m["scene_cut"] else 1) * (0.7 if m["frozen"] else 1) * (0.2 if m["brightness"] < 12 else 1)
        return round(float(total), 4)
    return score


def iqa(name, frames=4):  # a pyiqa model, averaged over a few frames of the clip
    def load():
        import pyiqa, torch
        m = pyiqa.create_metric(name, device="cpu")
        def score(f):
            x = torch.from_numpy(np.stack(pick(f, frames))).permute(0, 3, 1, 2).float() / 255
            with torch.inference_mode(): return float(m(x).mean())
        return score
    return load


def aes25():  # Aesthetic Predictor v2.5 (SigLIP), averaged over 4 frames
    import torch
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
    from PIL import Image
    model, proc = convert_v2_5_from_siglip(low_cpu_mem_usage=True, trust_remote_code=True); model.eval()
    def score(f):
        x = proc(images=[Image.fromarray(a) for a in pick(f, 4)], return_tensors="pt").pixel_values
        with torch.inference_mode(): return float(model(x).logits.float().mean())
    return score


def dover(yml, weights):  # DOVER: the whole clip, technical + aesthetic views fused as its repo does. Its code downloads on first use
    def load():
        import torch, yaml
        repo, w = MODELS_DIR / "dover" / "DOVER-master", MODELS_DIR / "dover" / weights.rsplit("/", 1)[1]
        if not repo.is_dir(): z = repo.with_suffix(".zip"); z.parent.mkdir(parents=True, exist_ok=True); torch.hub.download_url_to_file(DOVER_CODE, str(z)); shutil.unpack_archive(z, repo.parent); z.unlink()
        if not w.is_file(): torch.hub.download_url_to_file(DOVER_URL + weights, str(w))
        class VideoReader:  # DOVER imports decord + scikit-video (no Python 3.14 wheels): stand-ins hand it the clip's frames from memory
            def __init__(self, key, *a, **k): self.f = DOVER_FRAMES[key]
            def __len__(self): return len(self.f)
            def __getitem__(self, i): return torch.from_numpy(self.f[int(i) % len(self.f)].copy())
            def get_avg_fps(self): return 30.0
        sys.modules.setdefault("decord", types.SimpleNamespace(VideoReader=VideoReader, bridge=types.SimpleNamespace(set_bridge=lambda n: None), cpu=None, gpu=None))
        sys.modules.setdefault("skvideo", types.SimpleNamespace(io=None)); sys.modules.setdefault("skvideo.io", types.ModuleType("skvideo.io")); sys.path.insert(0, str(repo))
        from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
        from dover.models import DOVER
        opt = yaml.safe_load((repo / yml).read_text()); dopt = opt["data"]["val-l1080p"]["args"]
        m = DOVER(**opt["model"]["args"]).eval(); m.load_state_dict(torch.load(w, map_location="cpu"))
        mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
        samplers = {k: UnifiedFrameSampler(s["clip_len"], s["num_clips"], s["frame_interval"]) if "t_frag" not in s
                    else UnifiedFrameSampler(s["clip_len"] // s["t_frag"], s["t_frag"], s["frame_interval"], s["num_clips"]) for k, s in dopt["sample_types"].items()}
        def score(f):
            DOVER_FRAMES["clip"] = f; views, _ = spatial_temporal_view_decomposition("clip", dopt["sample_types"], samplers)
            for k, v in views.items():
                n = dopt["sample_types"][k].get("num_clips", 1)
                views[k] = ((v.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2).reshape(v.shape[0], n, -1, *v.shape[2:]).transpose(0, 1)
            with torch.inference_mode(): r = [float(x.mean()) for x in m(views)]
            return float(1 / (1 + np.exp(-((r[0] - 0.1107) / 0.07355 * 0.6104 + (r[1] + 0.08285) / 0.03774 * 0.3896))))
        return score
    return load


# key: name, what it judges, published accuracy (SRCC vs human ratings; different benchmarks, so only a rough guide),
#      accuracy dots 1-5, first-use download GB, seconds per try and load seconds (measured on a Windows laptop, 2026-11-01), loader
MODELS = {
    "current":      ("Current", "looks + technical", "not benchmarked", 2, 0.6, 0.9, 16, current),
    "dover-mobile": ("DOVER-Mobile", "video technical + aesthetic", "SRCC 0.85", 4, 0.05, 2.4, 8, dover("dover-mobile.yml", "v0.5.0/DOVER-Mobile.pth")),
    "musiq-spaq":   ("MUSIQ-SPAQ", "blur · noise · exposure", "SRCC 0.92 (photos)", 5, 0.1, 2.8, 18, iqa("musiq-spaq")),
    "musiq-ava":    ("MUSIQ-AVA", "composition", "SRCC 0.73 (photo aesthetics)", 3, 0.1, 2.9, 5, iqa("musiq-ava")),
    "laion_aes":    ("LAION ViT-L/14", "beauty", "not formally benchmarked", 2, 1.7, 4.3, 12, iqa("laion_aes")),
    "dover":        ("DOVER", "video technical + aesthetic", "SRCC 0.88", 5, 0.25, 14.5, 17, dover("dover.yml", "v0.1.0/DOVER.pth")),
    "aes-v2.5":     ("Aesthetic v2.5", "beauty · colour · composition", "not published", 2, 3.4, 18, 93, aes25)}
HUB = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub"
FILES = {"current": [MODELS_DIR / "hub/models--timm--vit_base_patch32_clip_224.openai", MODELS_DIR / "laion_aesthetic_vit_b_32_linear.pth"],
         "dover-mobile": [MODELS_DIR / "dover/DOVER-Mobile.pth", MODELS_DIR / "dover/DOVER-master"], "dover": [MODELS_DIR / "dover/DOVER.pth", MODELS_DIR / "dover/DOVER-master"],
         "musiq-spaq": [HUB / "pyiqa/musiq_spaq_ckpt-358bb6af.pth"], "musiq-ava": [HUB / "pyiqa/musiq_ava_ckpt-e8d3f067.pth"],
         "laion_aes": [Path.home() / ".cache/clip/ViT-L-14.pt", HUB / "pyiqa/sac%2Blogos%2Bava1-114-linearMSE.pth"],
         "aes-v2.5": [MODELS_DIR / "hub/models--google--siglip-so400m-patch14-384", HUB / "checkpoints/aesthetic_predictor_v2_5.pth"]}  # where each model's weights land


def value(key, raw, history=()):  # 0..1, equal weight for every model: how this clip compares with that model's typical clips
    if len(history) >= 30 or (key not in calibration and len(history) >= 5): mu, sd = statistics.fmean(history), statistics.stdev(history)
    elif key in calibration: mu, sd = calibration[key]["mean"], calibration[key]["sd"]
    else: return 0.5  # not calibrated and too few scores yet: neutral
    return statistics.NormalDist(mu, max(sd, 1e-6)).cdf(raw)


# ---------------------------------------------------------------- filling the shots
def choose():  # the system's folder dialog (Mac: Finder's, else Tk's, in a process of its own); "" if cancelled
    cmd = ["osascript", "-e", "activate", "-e", 'POSIX path of (choose folder with prompt "Footage folder")'] if sys.platform == "darwin" else [sys.executable, "-c",
           "import tkinter as t, tkinter.filedialog as f; r = t.Tk(); r.withdraw(); r.attributes('-topmost', 1); print(f.askdirectory(title='Footage folder', initialdir='~'))"]
    return subprocess.run(cmd, capture_output=True, encoding="utf-8", env=os.environ | {"PYTHONIOENCODING": "utf-8"}).stdout.strip()


def videos(folder): return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in VIDEO_EXT and not p.name.startswith(".") and OUTPUT not in p.relative_to(folder).parts)  # all subfolders; skips hidden files + saved reels


def order(pool, rnd): return rnd.sample(pool, len(pool))  # ALL the videos as one shuffled flat list: each equally likely first, however deep its folder


def prefetch(link):  # as soon as a link is pasted: the template, its shot count and length (for the estimate)
    state["tpl"] = {"link": link, "status": "loading"}
    try: d, T = template(link); state["tpl"] = {"link": link, "status": "ready", "shots": len(T["shots"]), "dur": round(T["frames"] / T["fps"], 2)}; text_status(d)  # its on-video text, found meanwhile
    except Exception as e: traceback.print_exc(); state["tpl"] = {"link": link, "status": "error", "msg": last(e)}


def model_info():  # the setup page's model list: what each judges, its accuracy, measured time per try
    return [{"key": k, "name": m[0], "judges": m[1], "accuracy": m[2], "dots": m[3], "download_gb": m[4], "try_s": round(med("try:" + k, m[5]), 1),
             "load_s": round(med("load:" + k, m[6])), "on_disk": all(p.exists() for p in FILES[k])} for k, m in MODELS.items()]


def estimate(shots, tries, thr, nv, keys):  # seconds: models' load + expected tries x time per try + new videos read + render
    sc = stats.get("combined", []); rough = len(sc) < 30
    p = max(1e-3, sum(x >= thr for x in sc) / len(sc) if not rough else 1 - thr)  # combined scores are ~uniform 0..1 by design
    pass_in_video = 1 - (1 - p) ** tries; vids = min(nv, 1 / pass_in_video); per_video = pass_in_video / p  # E[min(geometric, tries)]
    try_s = med("decode_s", 0.4) + sum(med("try:" + k, MODELS[k][5]) for k in keys)
    load_s = sum(med("load:" + k, MODELS[k][6]) for k in keys if k not in LOADED)
    return {"seconds": round(load_s + shots * vids * per_video * try_s + min(nv, shots * vids) * med("probe_s", 3) + med("render_s", 20)), "rough": rough}


def models(keys, quiet=False):  # load the ticked scoring models once per process
    for k in keys:
        if k not in LOADED: quiet or busy(f"Loading {MODELS[k][0]}"); t = time.time(); LOADED[k] = MODELS[k][7](); note("load:" + k, time.time() - t)
    return keys


def rate(v, start, length, keys):  # one try: decode the clip once, every ticked model scores it, equal-weight combined
    t = time.time(); f = S["readers"][v].frames(start, length); note("decode_s", time.time() - t); vals = {}
    if len(f) <= 1: print(f"rate: {Path(v).name} decoded {len(f)} frame(s) at {start:.2f}s/{length:.2f}s - scoring skipped, score 0", flush=True)
    for k in keys if len(f) > 1 else []:
        t = time.time(); raw = LOADED[k](f); note("try:" + k, time.time() - t)
        vals[k] = round(value(k, raw, stats.get("raw:" + k, [])), 4); note("raw:" + k, raw)
    score = round(statistics.fmean(vals.values()), 4) if vals else 0.0; note("combined", score)
    return {"start": round(start, 3), "score": score, "scores": vals}


def probe_video(v):  # a video's meta (None if unreadable), read once
    if v not in S["metas"]:
        t = time.time()
        try: S["metas"][v] = describe(v); S["readers"][v] = Reader(v, S["metas"][v])
        except Exception: S["metas"][v] = None
        note("probe_s", time.time() - t)
    return S["metas"][v]


def search(vids, length, tries, thr, rnd, label, keys):  # videos in the given order, up to `tries` random starts each
    best = None; score_len = max(length, 0.5)  # rate() needs >= 2 frames; pad micro-shots (template occasionally yields 1-frame cuts) so scoring sees a real clip. Render still uses the shot's real length.
    print(f"search: {label} length={length:.2f}s score_len={score_len:.2f}s pool={len(vids)} thr={thr}", flush=True)
    for v in vids:
        m = probe_video(v)
        if not m or m["duration_s"] < score_len + 0.2: continue
        for t in range(tries):
            r = rate(v, rnd.uniform(0, m["duration_s"] - score_len - 0.1), score_len, keys)
            busy(f"{label} · {v.name} · try {t + 1}/{tries} · score {r['score']:.2f}")
            if not best or r["score"] > best[1]["score"]: best = (v, r)
            if r["score"] >= thr: break
        if best and best[1]["score"] >= thr: break
    if not best: raise RuntimeError(f"no usable video longer than {score_len:.2f} s")
    return {"src": str(best[0]), **best[1], "low": best[1]["score"] < thr}


def render(plan, fps, out, audio, a0=0.0, crf=18):  # one ffmpeg call: each clip from its start, exactly its frames, under the audio from a0 (crf 0: lossless)
    n = lambda p: p["end_f"] - p["start_f"]; t0 = time.time()
    graph = "".join(f"[{i}:v]{','.join(normalise(probe_video(Path(p['src'])), fps))},tpad=stop_mode=clone:stop={n(p)},"
                    f"trim=end_frame={n(p)},setpts=PTS-STARTPTS[v{i}];" for i, p in enumerate(plan))
    (gf := out.with_suffix(".graph.txt")).write_text(graph + "".join(f"[v{i}]" for i in range(len(plan))) + f"concat=n={len(plan)}:v=1:a=0[v]")
    ff([*sum([["-ss", f"{p['start']:.3f}", "-i", p["src"]] for p in plan], []), "-ss", f"{a0:.4f}", "-t", f"{sum(map(n, plan)) / fps:.4f}",
        "-i", audio, "-/filter_complex", gf, "-map", "[v]", "-map", f"{len(plan)}:a", "-c:v", "libx264", "-crf", str(crf),
        "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out], "render")
    gf.unlink(); note("render_s", time.time() - t0)


def publish(d, data, out):  # the reel's .json + a viewer page that also works opened as a file
    out.with_suffix(".json").write_text(json.dumps(data, indent=1)); view = out.with_name(f"{out.stem}_view.html")
    view.write_text(VIEW.replace("/*DATA*/null", json.dumps(data)), encoding="utf-8")
    S.update(reel=f"{d.name}/{out.name}", data=data, dir=d); save_stats(); return f"/t/{d.name}/{view.name}"


def build(link, folder, tries, threshold, keys=("current",), seed=None):
    busy("Getting the template"); d, T = template(link); pool = videos(folder)
    if not pool: raise RuntimeError(f"no videos in {folder}")
    keys = models(list(keys)); rnd = random.Random(seed); plan = []
    for k, s in enumerate(T["shots"]):
        for r in S["readers"].values(): r.cap and r.cap.release()  # previous shot's cv2 caps hold ffmpeg decoder threads; free them before this shot's search probes fresh videos
        S["readers"].clear(); S["metas"].clear()
        plan.append({**s, **search(order(pool, rnd), (s["end_f"] - s["start_f"]) / T["fps"], tries, threshold, rnd, f"Shot {k + 1} of {len(T['shots'])}", keys)})
    busy("Rendering"); out = d / f"reel_{stamp()}.mp4"; render(plan, T["fps"], out, d / "audio.wav")
    url = publish(d, {**T, "video": out.name, "shots": plan, "template": d.name, "folder": str(folder), "tries": tries, "threshold": threshold, "models": keys}, out)
    state.update(stage="done", url=url, path=str(out)); return out


def editing(reel):  # the reel being edited (its .json is reloaded after a restart)
    if S.get("reel") != reel: S.update(reel=reel, data=json.loads((TEMPLATES / reel).with_suffix(".json").read_text()), dir=(TEMPLATES / reel).parent)
    return S["data"], S["dir"]


def replace(reel, i, mode, path="", seed=None, start=0.0):  # 5 delta-stratified candidates for shot i: same = 5 moments in one video; random = 5 videos across ordering; slide = the moment you picked
    for r in S["readers"].values(): r.cap and r.cap.release()  # the previous replace's cv2 caps hold ffmpeg decoder threads; free them before probing fresh videos
    S["readers"].clear(); S["metas"].clear()
    data, d = editing(reel); s = data["shots"][i]; rnd = random.Random(seed); fps = data["fps"]
    L = (s["end_f"] - s["start_f"]) / fps; tries, thr = data.get("tries", TRIES), data["threshold"]
    keys = models(data.get("models", ["current"])); cands = []
    if mode == "path":
        p = Path(path.strip().strip('"'))
        if not p.is_file(): raise RuntimeError(f"not a video file: {p}")
        cands.append({**s, **search([p], L, tries, thr, rnd, f"Shot {i + 1}", keys)})
    elif mode == "slide":  # scored for reference only: never marked low
        v = S["slide"]; probe_video(v); cands.append({**s, "src": str(v), **rate(v, start, max(L, 0.5), keys), "low": False})
    elif mode == "same":
        v = Path(path.strip().strip('"')) if path else Path(s["src"]); m = probe_video(v)
        if not m: raise RuntimeError(f"cannot probe {v}")
        D = m["duration_s"]; dt = rnd.uniform(0, 0.2 * D)
        for k in range(5):
            start = max(0.0, min(D - L - 0.1, 0.2 * k * D + dt)); r = rate(v, start, L, keys)
            busy(f"Shot {i + 1} · tab {k + 1}/5 · score {r['score']:.2f}")
            cands.append({**s, "src": str(v), **r, "low": r["score"] < thr})
    else:
        pool = videos(data["folder"]); n = len(pool); dt = rnd.uniform(0, 0.2)
        for k in range(min(5, n)):
            lo, hi = int(0.2 * k * n), min(n, int(0.2 * (k + 1) * n))
            if lo >= hi: continue
            anc = min(int((0.2 * k + dt) * n), hi - 1); vids = [pool[anc], *rnd.sample(pool[lo:anc] + pool[anc + 1:hi], hi - lo - 1)]
            try: cands.append({**s, **search(vids, L, tries, thr, rnd, f"Shot {i + 1} · tab {k + 1}/5", keys)})
            except RuntimeError: pass
    if not cands: raise RuntimeError(f"no usable video longer than {L:.2f} s")
    (d / "previews").mkdir(exist_ok=True)
    for k, c in enumerate(cands):
        busy(f"Rendering {k + 1}/{len(cands)}"); out = d / "previews" / f"shot{i + 1}_tab{k + 1}_{stamp()}.mp4"
        render([c], fps, out, d / "audio.wav", s["start_f"] / fps); c["preview"] = f"/t/{d.name}/previews/{out.name}"
    S["pending"] = (reel, i, cands); state.update(stage="preview", reel=reel, shot=i, cands=cands, msg=""); save_stats(); return cands


def tkey(p): st = p.stat(); return hashlib.sha1(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]  # a changed video gets a new icon
def tile(f): return {"img": f"/tc/{f.name}", "dur": float(f.stem.split("_")[1])} if f else {"img": None, "dur": None}  # None: unreadable


def grid(reel, i):  # Choose a video: every video in the footage folder by file name (a phone's shooting order, across subfolders), icons already made, the shot's own video
    data, d = editing(reel); s = data["shots"][i]; have = {f.name.split("_")[0]: f for f in THUMBS.glob("*_*.jpg")}
    S["grid"] = sorted((p.resolve() for p in videos(data["folder"])), key=lambda p: (p.name.lower(), str(p).lower()))
    return {"tiles": [{"name": p.name, "path": str(p), **(tile(f) if (f := have.get(tkey(p))) else {})} for p in S["grid"]],
            "cur": next((k for k, p in enumerate(S["grid"]) if p == Path(s["src"]).resolve()), -1), "need": max((s["end_f"] - s["start_f"]) / data["fps"], 0.5) + 0.2}


def thumb(p):  # the first frame at 480 px + the length, cached in THUMBS. OpenCV's built-in FFmpeg, in-process (~0.15 s; an ffmpeg launch costs ~5 s on Windows), else the ffmpeg command
    import cv2
    h = tkey(p); THUMBS.mkdir(parents=True, exist_ok=True); cap = cv2.VideoCapture(str(p)); ok, f = cap.read(); fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT); cap.release()
    if ok and fps > 0 and n > 0:  # imencode + write_bytes: cv2.imwrite can't take non-ASCII paths on Windows
        s = min(1, 480 / max(f.shape[:2])); (out := THUMBS / f"{h}_{n / fps:.2f}.jpg").write_bytes(cv2.imencode(".jpg", cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))[1].tobytes()); return out
    tmp = THUMBS / f"{h}.part.jpg"
    r = subprocess.run([binary("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-i", str(p), "-an", "-vf", "scale=480:480:force_original_aspect_ratio=decrease",
                        "-frames:v", "1", str(tmp)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode or not tmp.is_file() or not (m := re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)): return None
    return tmp.replace(THUMBS / f"{h}_{int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]):.2f}.jpg")


def strip(p, n=16):  # Slide's timeline: n frames evenly across the video, centre squares side by side, cached as THUMBS/strip-<hash>_<duration s>.jpg
    import cv2
    h = tkey(p); THUMBS.mkdir(parents=True, exist_ok=True)
    if f := next(THUMBS.glob(f"strip-{h}_*.jpg"), None): return f
    cap = cv2.VideoCapture(str(p)); fps, N = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT); fs = []
    for k in range(n if fps > 0 and N > 0 else 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * N / n)); ok, f = cap.read()
        if not ok: break
        c = min(f.shape[:2]); y, x = (f.shape[0] - c) // 2, (f.shape[1] - c) // 2; fs.append(cv2.resize(f[y:y + c, x:x + c], (90, 90), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(fs) == n: (out := THUMBS / f"strip-{h}_{N / fps:.2f}.jpg").write_bytes(cv2.imencode(".jpg", cv2.hconcat(fs))[1].tobytes()); return out
    if not (m := probe_video(p)): return None  # OpenCV couldn't (some HEVC on a Mac): the ffmpeg command
    ff(["-i", p, "-an", "-vf", f"fps={n / m['duration_s']:.6f},crop=min(iw\\,ih):min(iw\\,ih),scale=90:90,tile={n}x1", "-frames:v", "1", (out := THUMBS / f"strip-{h}_{m['duration_s']:.2f}.jpg")], "strip")
    return out


def slide(reel, i, k):  # Slide: the video (grid tile k, or the shot's own when k < 0), its strip and length, the shot's length, where the window starts
    data, d = editing(reel); s = data["shots"][i]; p = S["slide"] = Path(s["src"]).resolve() if k < 0 else S["grid"][k]
    if not (f := strip(p)): raise RuntimeError(f"can't read {p.name}")
    return {"name": p.name, "strip": f"/tc/{f.name}", "dur": float(f.stem.split("_")[1]), "len": (s["end_f"] - s["start_f"]) / data["fps"], "start": s["start"] if k < 0 else 0}


def slide_score(reel, i, start):  # the window's score, for reference only; the models load quietly the first time
    data, d = editing(reel); s = data["shots"][i]
    with SCORE: keys = models(data.get("models", ["current"]), True); probe_video(S["slide"]); return rate(S["slide"], start, max((s["end_f"] - s["start_f"]) / data["fps"], 0.5), keys)


def keep(tab=0):  # the selected tab's clip becomes the shot's clip, and the reel is re-rendered with it right away
    reel, i, cs = S.pop("pending"); data, d = editing(reel); data["shots"][i] = {k: x for k, x in cs[tab].items() if k != "preview"}; return save(reel)


def save(reel):  # a new version of the reel (earlier versions are kept)
    data, d = editing(reel); busy("Updating the reel"); out = d / f"reel_{stamp()}.mp4"; plan = data["shots"]; render(plan, data["fps"], out, d / "audio.wav")
    url = publish(d, {**data, "video": out.name, "shots": plan}, out); shutil.rmtree(d / "previews", ignore_errors=True)
    state.update(stage="saved", url=url, path=str(out)); return out


def export(reel, text=False):  # Save reel: the final reel + every shot as its own numbered clip (with its audio), in <footage>/template_reel_output/<reel id>/; text: with the on-video text
    data, d = editing(reel); src = d / data["video"]; out = Path(data["folder"]) / OUTPUT / src.stem; out.mkdir(parents=True, exist_ok=True)
    busy("Saving the reel and its clips" + (" with the on-video text" if text else "")); cuts = [s["start_f"] for s in data["shots"][1:]]
    if text: ins, g = burn(d, True, 1080, 1920, 1); (gf := d / "text.graph.txt").write_text(g); ff(["-i", src, *ins, "-/filter_complex", gf, "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-crf", "18",
               "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", out / src.name], "on-video text"); gf.unlink()
    else: shutil.copyfile(src, out / src.name)
    ff(["-i", out / src.name, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",  # keyframes half a frame early: land on frame f, never f+1
        "-force_key_frames", ",".join(f"{(f - 0.5) / data['fps']:.5f}" for f in cuts), "-f", "segment", "-segment_frames", ",".join(map(str, cuts)),
        "-segment_start_number", "1", "-reset_timestamps", "1", out / f"{src.stem}_%02d.mp4"], "clips")
    if c := caption(d): (out / "caption.txt").write_text(c, encoding="utf-8")
    state.update(stage="exported", folder=str(out), msg=""); return out


def esrgan():  # Real-ESRGAN's compact general model (SRVGGNetCompact, x4); its denoise strength blends its two weight sets, as Real-ESRGAN does
    import torch; nn = torch.nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.up = nn.PixelShuffle(4)
            self.body = nn.ModuleList([nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(64), *sum([[nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(64)] for _ in range(32)], []), nn.Conv2d(64, 48, 3, 1, 1)])
        def forward(self, x):
            y = x
            for m in self.body: y = m(y)
            return self.up(y) + nn.functional.interpolate(x, scale_factor=4, mode="nearest")
    w = []
    for v in ("x4v3", "wdn-x4v3"):
        if not (f := MODELS_DIR / f"realesr-general-{v}.pth").is_file(): f.parent.mkdir(parents=True, exist_ok=True); torch.hub.download_url_to_file(f"{ESRGAN}{v}.pth", str(f))
        w.append((c := torch.load(f, map_location="cpu")).get("params_ema", c.get("params")))
    net = Net(); net.load_state_dict({k: K4["denoise"] * w[0][k] + (1 - K4["denoise"]) * w[1][k] for k in w[0]}); return net.eval()


def rife():  # RIFE 4.26 (Practical-RIFE's IFNet_HDv3): net(a, b, t) = the frame at t (0-1) between a and b; 1x3xHxW in 0-1, H and W multiples of 64
    import torch; nn, F = torch.nn, torch.nn.functional; act = lambda: nn.LeakyReLU(0.2, True)
    def warp(x, fl):  # a clamped grid with zero padding = border padding, which MPS gets wrong
        _, _, H, W = fl.shape; g = torch.stack(torch.meshgrid(torch.linspace(-1, 1, H, device=x.device), torch.linspace(-1, 1, W, device=x.device), indexing="ij")[::-1])[None]
        g = (g + torch.cat([fl[:, :1] / ((W - 1) / 2), fl[:, 1:2] / ((H - 1) / 2)], 1)).permute(0, 2, 3, 1).clamp(-1, 1)
        return F.grid_sample(x, g, mode="bilinear", padding_mode="zeros", align_corners=True)
    class Res(nn.Module):
        def __init__(s, c): super().__init__(); s.conv = nn.Conv2d(c, c, 3, 1, 1); s.beta = nn.Parameter(torch.ones(1, c, 1, 1)); s.relu = act()
        def forward(s, x): return s.relu(s.conv(x) * s.beta + x)
    class Block(nn.Module):
        def __init__(s, i, c):
            super().__init__(); s.conv0 = nn.Sequential(nn.Sequential(nn.Conv2d(i, c // 2, 3, 2, 1), act()), nn.Sequential(nn.Conv2d(c // 2, c, 3, 2, 1), act()))
            s.convblock = nn.Sequential(*[Res(c) for _ in range(8)]); s.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 52, 4, 2, 1), nn.PixelShuffle(2))
        def forward(s, x, fl, k):
            x = F.interpolate(x, scale_factor=1 / k, mode="bilinear")
            if fl is not None: x = torch.cat((x, F.interpolate(fl, scale_factor=1 / k, mode="bilinear") / k), 1)
            t = F.interpolate(s.lastconv(s.convblock(s.conv0(x))), scale_factor=k, mode="bilinear"); return t[:, :4] * k, t[:, 4:5], t[:, 5:]
    class Head(nn.Module):
        def __init__(s): super().__init__(); s.cnn0, s.cnn1, s.cnn2, s.cnn3, s.relu = nn.Conv2d(3, 16, 3, 2, 1), nn.Conv2d(16, 16, 3, 1, 1), nn.Conv2d(16, 16, 3, 1, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1), act()
        def forward(s, x): return s.cnn3(s.relu(s.cnn2(s.relu(s.cnn1(s.relu(s.cnn0(x)))))))
    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.encode = Head(); s.block0 = Block(15, 192)
            for k, c in ((1, 128), (2, 96), (3, 64), (4, 32)): setattr(s, f"block{k}", Block(28, c))
        def forward(s, a, b, t):
            t = a[:, :1] * 0 + t; f0, f1 = s.encode(a), s.encode(b); fl = m = ft = None; wa, wb = a, b
            for i, k in enumerate((16, 8, 4, 2, 1)):
                if fl is None: fl, m, ft = s.block0(torch.cat((a, b, f0, f1, t), 1), None, k)
                else: d, m, ft = getattr(s, f"block{i}")(torch.cat((wa, wb, warp(f0, fl[:, :2]), warp(f1, fl[:, 2:4]), t, m, ft), 1), fl, k); fl = fl + d
                wa, wb = warp(a, fl[:, :2]), warp(b, fl[:, 2:4])
            m = torch.sigmoid(m); return wa * m + wb * (1 - m)
    if not (f := MODELS_DIR / "rife426.pth").is_file(): f.parent.mkdir(parents=True, exist_ok=True); torch.hub.download_url_to_file(RIFE, str(f))
    w = {k.removeprefix("module."): v for k, v in torch.load(f, map_location="cpu").items()}; net = Net(); net.load_state_dict({k: w[k] for k in net.state_dict()}); return net.eval()


def make4k(reel, text=False):  # Make 4K: a lossless 1080p render from your original videos (same cuts, 60 fps at least), then 2160x3840 three times (text: with the on-video text), next to Save reel's files:
    import cv2, torch  # _4k_plain (lanczos), _4k_ai (Real-ESRGAN 2x, blended with the plain one, light grain), _4k_ai_smooth (the same, with RIFE's in-between frames where a slower clip repeats one). The Mac's GPU when there is one
    data, d = editing(reel); src = d / data["video"]; out = Path(data["folder"]) / OUTPUT / src.stem; out.mkdir(parents=True, exist_ok=True); fps = max(data["fps"], 60)
    shots = [{**s, "start_f": round(s["start_f"] * fps / data["fps"]), "end_f": round(s["end_f"] * fps / data["fps"])} for s in data["shots"]]  # the cuts at that rate
    busy("4K: a lossless 1080p render from your videos"); tmp = d / f"{src.stem}_4k_master.mp4"; render(shots, fps, tmp, d / "audio.wav", crf=0)
    c = next(s for s in probe(tmp)["streams"] if s["codec_type"] == "video")  # its colour tags, kept on both 4K files
    col = [x for k, o in (("color_range", "-color_range"), ("color_space", "-colorspace"), ("color_primaries", "-color_primaries"), ("color_transfer", "-color_trc")) if c.get(k) not in (None, "unknown") for x in (o, c[k])]
    enc = lambda name: ["-c:v", "libx264", "-crf", K4["crf"], "-preset", "medium", "-pix_fmt", "yuv420p", *col, "-c:a", "copy", "-movflags", "+faststart", out / f"{src.stem}_{name}.mp4"]
    ins, g = burn(d, text, 2160, 3840, 1, pre="scale=2160:3840:flags=lanczos"); (gf := d / "text.graph.txt").write_text(g)
    busy("4K (plain): upscaling"); ff(["-i", tmp, *ins, "-/filter_complex", gf, "-map", "[v]", "-map", "0:a", *enc("4k_plain")], "4K plain")
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"; half = dev != "cpu"
    busy("4K (AI): loading Real-ESRGAN and RIFE"); net = esrgan().to(dev); net = net.half() if half else net; mid = rife().to(dev)
    n = sum(lens := [s["end_f"] - s["start_f"] for s in shots]); cuts = set(np.cumsum(lens).tolist())  # each shot's first frame
    rgb = f"scale=out_color_matrix={ {'bt470bg': 'bt470'}.get(c.get('color_space'), c.get('color_space') or 'bt709')}:out_range={c.get('color_range') or 'tv'},format=yuv420p,noise=alls={K4['grain']}:allf=t,setparams=" + ":".join(  # raw RGB carries no colour tags: set them
        f"{p}={c[k]}" for k, p in (("color_range", "range"), ("color_primaries", "color_primaries"), ("color_transfer", "color_trc"), ("color_space", "colorspace")) if c.get(k) not in (None, "unknown"))
    dec = subprocess.Popen([binary("ffmpeg"), "-v", "error", "-i", str(tmp), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    ins, g = burn(d, text, 2160, 3840, 2, post=rgb); gf.write_text(g)
    pipe = lambda name: subprocess.Popen([binary("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "2160x3840", "-r", str(fps), "-i", "-", "-i", str(tmp),
                               *map(str, ins), "-/filter_complex", str(gf), "-map", "[v]", "-map", "1:a", *map(str, enc(name))], stdin=subprocess.PIPE)
    outs, t0, gpu = [pipe("4k_ai"), pipe("4k_ai_smooth")], time.time(), lambda a: torch.from_numpy(a).to(dev).permute(2, 0, 1)[None].float() / 255
    byte = lambda y: (y.clamp(0, 1) * 255).round().byte()[0].permute(1, 2, 0).contiguous().cpu().numpy()
    def up(f):  # a 1080x1920 frame -> its AI 4K one: Real-ESRGAN x4, then exactly half, blended with lanczos
        x = gpu(f); ai = torch.nn.functional.avg_pool2d(net(x.half() if half else x).float(), 2)
        return byte(K4["ai"] * ai + (1 - K4["ai"]) * gpu(cv2.resize(f, (2160, 3840), interpolation=cv2.INTER_LANCZOS4))).tobytes()
    def flush(nxt):  # prev's repeats: _4k_ai keeps them, _4k_ai_smooth gets RIFE's frames between prev and nxt (none across a cut); 1080 wide padded to 1088 for RIFE
        p = lambda f: torch.nn.functional.pad(gpu(f), (0, 8, 0, 0), mode="replicate")
        for j in range(1, rep + 1): outs[0].stdin.write(prev[1]); outs[1].stdin.write(prev[1] if nxt is None else up(byte(mid(p(prev[0]), p(nxt), j / (rep + 1))[:, :, :, :1080])))
    prev, rep = None, 0
    try:
        with torch.no_grad():
            for k in range(n):
                if len(b := dec.stdout.read(1080 * 1920 * 3)) < 1080 * 1920 * 3: break
                f = np.frombuffer(b, np.uint8).reshape(1920, 1080, 3).copy()
                if prev and k not in cuts and np.array_equal(f, prev[0]): rep += 1; continue  # a repeated frame: a 30 fps clip in a 60 fps reel
                flush(None if k in cuts else f); y = up(f); [o.stdin.write(y) for o in outs]; prev, rep = (f, y), 0; left = (time.time() - t0) / (k + 1) * (n - k - 1)
                busy(f"4K (AI + smooth): frame {k + 1} / {n} · about {int(left // 60)} min {int(left % 60):02d} s left")
            flush(None)
    finally: [o.stdin.close() for o in outs]; dec.kill(); [o.wait() for o in outs]; tmp.unlink(missing_ok=True); gf.unlink(missing_ok=True)
    if any(o.returncode for o in outs): raise RuntimeError("4K (AI): the encode failed")
    state.update(stage="exported", folder=str(out), msg=""); return out


# ---------------------------------------------------------------- the browser app
class H(SimpleHTTPRequestHandler):
    def reply(self, body, kind="application/json", code=200):
        b = body if isinstance(body, bytes) else (json.dumps(body) if kind == "application/json" else body).encode()
        self.send_response(code); self.send_header("Content-Type", kind); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        q = self.path.split("?", 1); a = {k: v[0] for k, v in parse_qs(q[1] if len(q) > 1 else "").items()}
        if q[0] == "/":
            if state["stage"] in ("done", "saved", "exported"): state["stage"] = "setup"  # the last reel stays at its own URL
            return self.reply(PAGE, "text/html; charset=utf-8")
        if q[0] == "/status": return self.reply(state)
        if q[0] == "/count":
            if not str(f := Path(a.get("path", "").strip().strip('"'))).strip(". ") or not f.is_dir(): return self.reply({"error": "not a folder"})
            vs = videos(f); return self.reply({"videos": len(vs), "folders": len({v.parent for v in vs})})
        if q[0] == "/choose": return self.reply({"path": choose()})
        if q[0] == "/models": return self.reply(model_info())
        if q[0] == "/estimate": return self.reply(estimate(int(a["shots"]), int(a["tries"]), float(a["threshold"]), max(1, int(a["videos"])),
                                                           [k for k in a.get("models", "current").split(",") if k in MODELS] or ["current"]))
        if q[0].endswith("_view.html") and (j := TEMPLATES / unquote(q[0][3:]).replace("_view.html", ".json")).is_file():  # always today's viewer
            return self.reply(VIEW.replace("/*DATA*/null", j.read_text()), "text/html; charset=utf-8")
        if q[0].startswith("/t/"): return self.send_file(TEMPLATES / q[0][3:])
        try:  # Choose a video: the grid's list, one icon (made on first ask)
            if q[0] == "/caption": return self.reply({"text": caption((TEMPLATES / a["reel"]).parent)})
            if q[0] == "/text": return self.reply(text_status((TEMPLATES / a["reel"]).parent))  # On-video text: its pieces once found
            if q[0] == "/list": return self.reply(grid(a["reel"], int(a["shot"])))
            if q[0] == "/thumb": return self.reply(tile(thumb(S["grid"][int(a["i"])])))
            if q[0] == "/slide": return self.reply(slide(a["reel"], int(a["shot"]), int(a["i"])))  # Slide: open a video, score a window
            if q[0] == "/score": return self.reply(slide_score(a["reel"], int(a["shot"]), float(a["start"])))
        except Exception as e: traceback.print_exc(); return self.reply({"error": last(e)})
        if q[0] == "/sv": return self.send_file(S["slide"], True)  # the video open in Slide
        if q[0].startswith("/tc/"): return self.send_file(THUMBS / Path(unquote(q[0][4:])).name, True)
        if q[0] == "/v": return self.send_file(S["grid"][int(a["i"])], True)  # a grid video, whole, for the preview: only videos in the list are sent
        self.reply(b"not found", "text/plain", 404)

    def do_POST(self):
        a, route = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}"), self.path
        if route == "/template": threading.Thread(target=prefetch, args=(a["link"].strip(),), daemon=True).start(); return self.reply({"ok": True})
        if route == "/cancel": S.pop("pending", None); state.update(stage="idle", msg=""); return self.reply({"ok": True})
        if route == "/textdrop" and (t := (TEMPLATES / a["reel"]).parent / "text").is_dir(): (t / "drop.json").write_text(json.dumps([str(p) for p in a["drop"]])); return self.reply({"ok": True})  # a text piece ticked / unticked
        if state["stage"] == "working": return self.reply({"error": "already running"})
        if route == "/keep" and "pending" not in S: return self.reply({"error": "nothing to keep"})
        job = {"/start": lambda: build(a["link"].strip(), Path(a["folder"].strip().strip('"')), int(a["tries"]), float(a["threshold"]),
                                       [k for k in a.get("models", ["current"]) if k in MODELS] or ["current"]),
               "/replace": lambda: replace(a["reel"], int(a["shot"]), a["mode"], a.get("path", ""), start=float(a.get("start", 0))), "/keep": lambda: keep(int(a.get("tab", 0))), "/export": lambda: export(a["reel"], bool(a.get("text"))), "/4k": lambda: make4k(a["reel"], bool(a.get("text")))}[route]
        def run():
            try:
                with SCORE: job()  # waits for a Slide score in flight: they share the decoders and models
            except Exception as e: traceback.print_exc(); state.update(stage="error", msg=last(e))
        busy("Finishing the score first (the models load the first time)" if SCORE.locked() else "Starting"); threading.Thread(target=run, daemon=True).start(); self.reply({"ok": True})

    def send_file(self, p, anywhere=False):  # with HTTP Range, so the browser can seek in the video; templates/ only, unless the caller checked p
        p = Path(p).resolve() if anywhere else Path(unquote(str(p))).resolve()
        if not p.is_file() or not (anywhere or TEMPLATES.resolve() in p.parents): return self.reply(b"not found", "text/plain", 404)
        size = p.stat().st_size; m = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        a, b = (int(m[1]), min(size - 1, int(m[2] or size - 1))) if m else (0, size - 1)
        kind = "video/mp4" if p.suffix.lower() in (".mov", ".m4v") else self.guess_type(str(p))  # browsers play H.264/HEVC .mov sent as mp4
        self.send_response(206 if m else 200); self.send_header("Content-Type", kind); self.send_header("Accept-Ranges", "bytes")
        if m: self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.send_header("Content-Length", str(b - a + 1)); self.end_headers()
        with p.open("rb") as f: f.seek(a); self.wfile.write(f.read(b - a + 1))

    def log_message(self, *a): pass


PAGE = r"""<!doctype html><meta charset="utf-8"><title>Template reel</title>
<style>
body{margin:0;font:14px system-ui,sans-serif;background:#111;color:#eee;display:grid;place-items:center;min-height:100vh} main{width:min(520px,92vw);display:flex;flex-direction:column;gap:16px} h1{font-size:18px;font-weight:600;margin:0}
label,.f{display:flex;flex-direction:column;gap:5px;color:#aaa;font-size:12px} #choose{background:#1b1b1b;color:#eee;border:1px solid #333;border-radius:8px;font-weight:400;text-align:left;padding:9px 11px} summary{color:#777;font-size:11px;cursor:pointer} input[type=text]{background:#1b1b1b;color:#eee;border:1px solid #333;border-radius:8px;padding:9px 11px;font-size:14px} input:focus{outline:none;border-color:#e67e22}
small{color:#777;min-height:15px;font-size:11px} .bad{color:#e74c3c} .ok{color:#8c8} .sl{display:flex;flex-direction:column;gap:3px} .sl .top,.models .top{display:flex;justify-content:space-between;color:#aaa;font-size:12px} .sl b{color:#f5b041;font-weight:600}
input[type=range]{accent-color:#e67e22;width:100%;margin:2px 0} .ends{display:flex;justify-content:space-between;color:#555;font-size:10px} #eta{color:#ccc;font-size:13px;text-align:center;min-height:18px} #eta b{color:#f5b041}
.models{display:flex;flex-direction:column;gap:2px} .models a{color:#f5b041;cursor:pointer;font-size:11px} .m{display:grid;grid-template-columns:18px 118px 1fr 74px 52px;gap:6px;align-items:center;font-size:12px;padding:3px 4px;border-radius:6px;cursor:pointer} .m:hover{background:#1b1b1b}
.m input{accent-color:#e67e22;margin:0} .m .j{color:#888;font-size:11px} .m .t{color:#f5b041;text-align:right;font-size:11px} .m .d{color:#e67e22;letter-spacing:1px;font-size:10px} .m .dl{grid-column:3/6;color:#666;font-size:10px;margin-top:-3px}
button{background:linear-gradient(90deg,#f5b041,#e67e22);color:#111;border:0;border-radius:18px;padding:9px;font-weight:600;font-size:14px;cursor:pointer} button:disabled{background:#333;color:#777;cursor:default} #prog{display:none;flex-direction:column;gap:8px} #msg{color:#ccc} #err{color:#e74c3c}
.track{height:4px;border-radius:2px;background:#222;overflow:hidden} .track div{height:100%;width:30%;background:#e67e22;animation:s 1.2s infinite ease-in-out} @keyframes s{0%{margin-left:-30%}100%{margin-left:100%}}
</style>
<main>
<h1>Template reel</h1>
<label>Instagram Reel or YouTube Short link<input type="text" id="link" placeholder="https://www.instagram.com/reel/..." autofocus><small id="tpl"></small></label>
<div class="f">Footage folder (all subfolders are searched)<button id="choose">Choose folder…</button><small id="count"></small>
  <details><summary>or paste a path</summary><input type="text" id="folder" placeholder="paste the folder path" style="width:100%;box-sizing:border-box;margin-top:5px"></details></div>
<div class="models"><div class="top"><span>Scoring models (equal weight; tick one or more)</span><a id="vbench">VBench set</a></div><div id="mlist"></div>
  <small>Accuracy: each model's published agreement with human ratings, from different benchmarks, so a rough guide. Time: per try on this computer.</small></div>
<div class="sl"><div class="top"><span>Tries per video</span><b id="triesV"></b></div><input type="range" id="tries" min="1" step="1">
  <div class="ends"><span>1</span><span id="maxT"></span></div><small>How many random moments to try in one video before moving on to another video.</small></div>
<div class="sl"><div class="top"><span>Threshold</span><b id="thresholdV"></b></div><input type="range" id="threshold" min="0.2" max="0.8" step="0.05">
  <div class="ends"><span>0.20 · any decent clip</span><span>0.80 · only the best</span></div><small>Minimum quality a clip must score: 0.40 ≈ better than 40% of typical clips. Higher looks better but takes longer.</small></div>
<div id="eta"></div>
<button id="go" disabled>Make my reel</button>
<div id="prog"><div class="track"><div></div></div><div id="msg"></div></div><div id="err"></div>
</main>
<script>
const $ = s => document.querySelector(s); let S = {}, folder = null, timer, ftimer, started = false;
const mmss = s => s < 60 ? `${s} s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')} s`;
const ready = () => $('#go').disabled = !(folder && folder.videos > 0 && S.tpl?.status === 'ready' && S.tpl.link === $('#link').value.trim() && S.stage !== 'working');
async function eta() {
  $('#triesV').textContent = $('#tries').value; $('#thresholdV').textContent = (+$('#threshold').value).toFixed(2);
  if (!(S.tpl?.status === 'ready' && folder?.videos)) return $('#eta').textContent = 'Paste a link and a folder to see how long it will take.';
  const e = await (await fetch(`/estimate?shots=${S.tpl.shots}&tries=${$('#tries').value}&threshold=${$('#threshold').value}&videos=${folder.videos}&models=${ticked()}`)).json();
  $('#eta').innerHTML = `About <b>${mmss(e.seconds)}</b> for ${S.tpl.shots} shots${e.rough ? ' (rough until your first run)' : ''}`;
}
$('#tries').oninput = $('#threshold').oninput = eta;
const ticked = () => [...document.querySelectorAll('#mlist input:checked')].map(c => c.value);
fetch('/models').then(r => r.json()).then(ms => { $('#mlist').innerHTML = ms.map(m => `<label class="m" title="${m.accuracy}"><input type="checkbox" value="${m.key}"${m.key === 'current' ? ' checked' : ''}>
  <span>${m.name}</span><span class="j">${m.judges}</span><span class="t">~${m.try_s} s/try</span><span class="d">${'●'.repeat(m.dots)}${'○'.repeat(5 - m.dots)}</span>
  <span class="dl">${m.on_disk ? '' : ` · first use downloads ~${m.download_gb} GB`}</span></label>`).join('');
  document.querySelectorAll('#mlist input').forEach(c => c.onchange = () => { if (!ticked().length) c.checked = true; eta(); }); });
$('#vbench').onclick = () => { document.querySelectorAll('#mlist input').forEach(c => c.checked = ['laion_aes', 'musiq-spaq'].includes(c.value)); eta(); };
$('#link').oninput = () => { clearTimeout(timer); timer = setTimeout(() => { const l = $('#link').value.trim();
  if (/^https?:\/\//.test(l)) fetch('/template', {method: 'POST', body: JSON.stringify({link: l})}); }, 400); ready(); };
$('#folder').oninput = () => { clearTimeout(ftimer); folder = null; ready(); eta(); ftimer = setTimeout(async () => { const p = $('#folder').value.trim();
  const c = await (await fetch('/count?path=' + encodeURIComponent(p))).json(); folder = c.error ? null : c;
  $('#count').className = folder?.videos ? 'ok' : 'bad'; $('#count').textContent = p && `${p} · ` + (c.error || `${c.videos} videos in ${c.folders} folder(s)`); ready(); eta(); }, 300); };
$('#choose').onclick = async () => { $('#choose').disabled = true; const r = await (await fetch('/choose')).json(); $('#choose').disabled = false;  // the system's folder dialog
  if (r.path) { $('#folder').value = r.path; $('#folder').oninput(); } };
async function poll() {  // the server's state, twice a second
  const s = await (await fetch('/status')).json(), tplChanged = JSON.stringify(s.tpl) !== JSON.stringify(S.tpl); S = s;
  if (s.stage === 'done' && started) return location.href = s.url;
  const t = s.tpl; $('#tpl').className = t?.status === 'error' ? 'bad' : t?.status === 'ready' ? 'ok' : '';
  $('#tpl').textContent = !t || t.link !== $('#link').value.trim() ? '' : t.status === 'loading' ? 'Getting the template and its cuts…'
    : t.status === 'ready' ? `Template: ${t.shots} shots · ${t.dur} s` : 'Failed: ' + t.msg;
  $('#prog').style.display = s.stage === 'working' ? 'flex' : 'none'; $('#msg').textContent = s.msg || '';
  $('#err').textContent = s.stage === 'error' ? 'Failed: ' + s.msg : ''; if (tplChanged) eta(); ready(); setTimeout(poll, 500);
}
$('#go').onclick = async () => { started = true; $('#err').textContent = '';
  await fetch('/start', {method: 'POST', body: JSON.stringify({link: $('#link').value, folder: $('#folder').value, tries: $('#tries').value, threshold: $('#threshold').value, models: ticked()})}); };
fetch('/status').then(r => r.json()).then(s => { $('#tries').max = s.max_tries; $('#maxT').textContent = s.max_tries;
  $('#tries').value = s.tries; $('#threshold').value = s.threshold; eta(); poll(); });
</script>
"""

VIEW = r"""<!doctype html><meta charset="utf-8"><title>Template reel</title>
<style>
body{margin:0;font:14px system-ui,sans-serif;background:#111;color:#eee;display:flex;flex-direction:column;gap:8px;padding:12px;height:100vh;box-sizing:border-box} [hidden]{display:none!important}
#bar{position:relative;display:flex;flex:none;width:min(560px,100%);height:14px;margin:4px auto 0;border-radius:7px;overflow:hidden;cursor:pointer;background:#222} #bar div:hover{background:#3a4a5e}
#bar div{min-width:0;overflow:hidden;box-shadow:inset -2px 0 #111;font-size:9px;line-height:14px;text-align:center;color:#9ab;background:#2a3442;transition:background .15s} #bar div.on{background:linear-gradient(90deg,#f5b041,#e67e22);color:#111;font-weight:600}
#bar div.unsure{box-shadow:inset -2px 0 #e74c3c} #bar div.low{background:#5a2626;color:#f99} /* shadows, not borders: borders would add width and skew the time scale */ #head{position:absolute;top:0;bottom:0;width:2px;background:#fff;box-shadow:0 0 6px #fff;pointer-events:none}
#ctl,#edit{display:flex;flex:none;gap:8px;align-items:center;justify-content:center;flex-wrap:wrap;font-size:12px} button{background:#1e1e1e;color:#ddd;border:1px solid #333;border-radius:12px;padding:3px 12px;font-size:12px;cursor:pointer} button:hover{background:#2a2a2a}
button.hot{background:linear-gradient(90deg,#f5b041,#e67e22);color:#111;border-color:transparent;font-weight:600} #edit input{background:#1b1b1b;color:#eee;border:1px solid #333;border-radius:12px;padding:3px 10px;font-size:12px;width:280px}
#info{color:#888;width:100%;text-align:center} #info .q,.cap .q{color:#e74c3c} #msg{color:#aaa} #stage{flex:1;min-height:0;display:flex;gap:16px;justify-content:center} .cap{font-size:12px;color:#aaa;display:flex;gap:8px;align-items:center} .cap b{color:#eee} #R .cap b{color:#f5b041}
figure{margin:0;flex:1;min-width:0;display:flex;flex-direction:column;align-items:center;gap:6px} figure video{flex:1;min-height:0;width:100%;object-fit:contain}
#tabs{display:flex;gap:4px;flex:none} .tab{padding:2px 10px;background:#1b1b1b;border:1px solid #333;border-radius:10px;cursor:pointer;font-size:11px;color:#aaa} .tab:hover{background:#2a2a2a} .tab.on{background:#e67e22;color:#111;border-color:transparent} .tab .lo{color:#e74c3c}
#grid,#look,#slide{position:fixed;inset:0;z-index:5;display:flex;flex-direction:column;background:#111} #look,#slide{z-index:6;background:#000e;padding:14px;gap:10px;box-sizing:border-box} #slide{z-index:7;background:#000}
#strip{position:relative;flex:none;aspect-ratio:16/1;background:#1b1b1b center/100% 100% no-repeat;border-radius:6px;overflow:hidden;cursor:pointer;touch-action:none}
#win{position:absolute;top:0;bottom:0;min-width:6px;box-sizing:border-box;border:3px solid #f5b041;border-radius:6px;box-shadow:0 0 0 9999px #000a;cursor:grab} #sscore{color:#f5b041}
#gbar,#lbar,#sbar{flex-wrap:wrap;display:flex;flex:none;gap:12px;align-items:center;justify-content:center;font-size:12px;color:#aaa} #gbar{padding:10px 16px;background:#1a1a1a} #gbar input{accent-color:#e67e22;width:160px;vertical-align:middle}
#tiles{flex:1;overflow:auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--s),1fr));gap:10px;padding:16px;align-content:start}
.v{min-width:0;display:flex;flex-direction:column;align-items:center;gap:5px;padding:6px;border-radius:8px;cursor:pointer;font-size:11px;color:#aaa;user-select:none}
.v .im{position:relative;width:100%;aspect-ratio:1;display:flex;align-items:center;justify-content:center;background:#1b1b1b;border-radius:6px;color:#666} .v img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
.v span{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap} .v.on{background:#e67e2240;color:#fff} .v.cur .im{outline:2px solid #f5b041;outline-offset:2px} .v.short{opacity:.3}
#lv,#sv{flex:1;min-height:0;width:100%;object-fit:contain} #L{position:relative} #ov{position:absolute;pointer-events:none;overflow:hidden} #ov img{position:absolute} #ov img.hi{outline:2px solid #f5b041;outline-offset:3px}
#tl{flex:none;width:170px;overflow:auto;display:flex;flex-direction:column;gap:6px;font-size:11px;color:#aaa} #tl b{font-weight:400;color:#777} .tp{display:flex;flex-direction:column;gap:4px;padding:6px;border-radius:8px;background:#1b1b1b;border:1px solid transparent;cursor:pointer}
.tp:hover{background:#242424} .tp.now{border-color:#e67e22} .tp.off{opacity:.4} .tp img{height:24px;max-width:100%;object-fit:contain;object-position:left;background:#2b2b44;border-radius:4px;padding:4px} .tp div{display:flex;gap:6px;align-items:center} .tp input{accent-color:#e67e22;margin:0}
</style>
<div id="bar"><span id="head"></span></div>
<div id="ctl"><button id="prev">◀ Prev</button><button id="play">Play shot</button><button id="next">Next ▶</button>
<label><input type="checkbox" id="loop" checked> Loop shot</label><button id="all">Play whole reel</button><button id="rep" hidden>Replace clip</button>
<button id="export" class="hot" hidden>Save reel</button><button id="capt" hidden title="The template post's caption, to paste under your reel (Save reel also writes it as caption.txt)">Copy caption</button><label id="txl" hidden title="The template's own on-video text: same font, place and timing. Only shows or hides it; Save reel and Make 4K follow this box and the pieces ticked in the list on the right">
<input type="checkbox" id="tx" checked> <span id="txn">On-video text</span></label>
<button id="k4" hidden title="From your original videos: plain, AI and AI smooth (60 fps) 4K versions, next to Save reel's files (Mac: ~30-50 min)">Make 4K</button><span id="info"></span></div>
<div id="edit" hidden><span id="choose"><button data-m="same">Same video</button><button id="slideSame">Slide same video</button><button data-m="random">Random video</button><button class="browse">Choose a video</button>
  <button id="x">✕</button></span>
  <span id="msg"></span><span id="decide" hidden><button id="keep" class="hot">Keep new</button><button id="againSame">Try again: same video</button>
  <button id="againRand">Try again: random video</button><button class="browse">Try again: choose a video</button><button id="cancel">Cancel</button></span></div>
<div id="stage"><figure id="L"><div class="cap" hidden><b>Current</b><span id="lcap"></span><button id="lplay">▶ Play</button><label><input type="checkbox" id="lloop"> Loop</label></div>
  <video id="player" playsinline></video><div id="ov" hidden></div></figure>
  <figure id="R" hidden><div class="cap"><b>New</b><span id="rcap"></span><button id="rplay">▶ Play</button><label><input type="checkbox" id="rloop"> Loop</label></div>
  <div id="tabs"></div><video id="cand" playsinline></video></figure><div id="tl" hidden></div></div>
<div id="grid" hidden><div id="gbar"><span id="gn"></span><label>Icon size <input type="range" id="gsize" min="120" max="480" step="10"></label>
  <button id="gsel" class="hot" disabled>Select</button><button id="gslide" disabled>Slide</button><button id="gx">Cancel</button><span>Click: highlight · Enter / double-click: select · Space: preview · S: slide</span></div><div id="tiles"></div></div>
<div id="look" hidden><video id="lv" playsinline controls></video><div id="lbar"><span id="lname"></span><button id="lsel" class="hot">Select</button><button id="lslide">Slide</button><button id="lx">Close</button>
  <span>Space / Esc: close · ← →: previous / next · Enter: select · S: slide</span></div></div>
<div id="slide" hidden><video id="sv" playsinline muted></video><div id="strip"><div id="win"></div></div>
  <div id="sbar"><span id="sname"></span><span id="sscore"></span><button id="splay">▶ Play</button><button id="suse" class="hot">Use this clip</button><button id="sx">Back</button>
  <span>Drag the window · ← →: 0.1 s · Shift + ← →: 1 s · Space: play · Enter: use · Esc: back</span></div></div>
<script>
const D = /*DATA*/null, $ = s => document.querySelector(s), P = $('#player'), Q = $('#cand'), T = f => f / D.fps;
const fmt = s => `${Math.floor(s / 60)}:${(s % 60).toFixed(2).padStart(5, '0')}`, name = s => s.src.split(/[\\/]/).pop();
const seek = f => P.currentTime = T(f + 0.5);  // middle of frame f: never lands on its neighbour
let cur = 0, whole = false, pend = null; P.src = D.video;
D.shots.forEach((s, i) => $('#bar').insertAdjacentHTML('beforeend',  // cuts[i] ends shot i
  `<div style="flex:${s.len_s}" class="${D.cuts[i] && !D.cuts[i].sure ? 'unsure' : ''} ${s.low ? 'low' : ''}" title="Shot ${i + 1}: ${fmt(T(s.start_f))} - ${fmt(T(s.end_f))}">${i + 1}</div>`));
const segs = [...document.querySelectorAll('#bar div')], mark = k => segs.forEach((t, j) => t.classList.toggle('on', j === k));
const looping = () => $(pend ? '#lloop' : '#loop').checked;  // while comparing, each side has its own Loop
// Keep playback inside the shot: on its last frame, stop there (or loop back to its first frame).
// Judged per presented frame (requestVideoFrameCallback gives the frame's exact time); a 10 ms timer backs it up.
function guard(atLast) {
  const s = D.shots[cur]; if (whole || P.paused || !atLast(s)) return; P.pause(); seek(s.end_f - 1);
  setTimeout(() => { if (looping() && P.paused && !whole && D.shots[cur] === s) { seek(s.start_f); P.play().catch(() => {}); } }, 1000 / D.fps);
}
P.requestVideoFrameCallback?.(function f(_, m) { guard(s => m.mediaTime >= T(s.end_f - 1) - 1e-3); P.requestVideoFrameCallback(f); });
setInterval(() => {  // a timer, not rAF: rAF stops when the page isn't painted
  guard(s => P.currentTime >= T(s.end_f) - 0.3 / D.fps);
  if (whole) { const k = D.shots.findIndex(s => P.currentTime < T(s.end_f)); if (k >= 0 && k !== cur) { cur = k; mark(k); } }
  $('#head').style.left = `${100 * P.currentTime / T(D.frames)}%`;
  $('#lplay').textContent = P.paused ? '▶ Play' : '❚❚ Pause'; $('#rplay').textContent = Q.paused ? '▶ Play' : '❚❚ Pause'; txDraw();
}, 10);
// ---- On-video text: the template's text pieces (see-through PNGs) drawn over the player at their frames; the box only shows or hides them
let TX = null, drop = new Set(); const txKey = 'text:' + D.template, txOn = () => !!TX && $('#tx').checked;
function txDraw() {
  const ov = $('#ov'), on = txOn(); $('#tl').hidden = !on; if (!on) return ov.hidden = true; ov.hidden = false;
  const r = P.getBoundingClientRect(), f = $('#L').getBoundingClientRect(), vw = P.videoWidth || 1080, vh = P.videoHeight || 1920, k = Math.min(r.width / vw, r.height / vh);
  Object.assign(ov.style, {left: `${r.left - f.left + (r.width - vw * k) / 2}px`, top: `${r.top - f.top + (r.height - vh * k) / 2}px`, width: `${vw * k}px`, height: `${vh * k}px`});
  const [w, h] = TX.size, s = Math.max(vw / w, vh / h), ox = (w * s - vw) / 2, oy = (h * s - vh) / 2, fr = Math.floor(P.currentTime * TX.fps + 1e-3);  // the reel and the template share frames
  [...ov.children].forEach((im, i) => { const p = TX.sprites[i], now = p.frames.some(([a, b]) => a <= fr && fr <= b); im.style.display = now && !drop.has(p.png) ? '' : 'none'; TX.items[i].classList.toggle('now', now);
    Object.assign(im.style, {left: `${(p.x * s - ox) / vw * 100}%`, top: `${(p.y * s - oy) / vh * 100}%`, width: `${p.w * s / vw * 100}%`}); });
}
function txLoad() {  // found once per template, in the background (a minute or two the first time)
  fetch(`/text?reel=${encodeURIComponent(reel)}`).then(r => r.json()).then(t => {
    if (t.status === 'working') { $('#txl').hidden = false; $('#tx').disabled = true; $('#txn').textContent = 'Finding the on-video text…'; return setTimeout(txLoad, 3000); }
    if (!t.sprites?.length) return $('#txl').hidden = true;
    const src = p => `/t/${D.template}/text/${p.png}?v=${t.v}`, at = f => fmt(f / t.fps); drop = new Set(t.drop); $('#ov').innerHTML = t.sprites.map(p => `<img src="${src(p)}" alt="">`).join('');
    $('#tl').innerHTML = '<b id="tln"></b>' + t.sprites.map((p, i) => [p.frames[0][0], i]).sort((a, b) => a[0] - b[0]).map(([a, i]) => { const p = t.sprites[i];  // in time order: a picture of the piece, its time, keep it or not
      return `<div class="tp${drop.has(p.png) ? ' off' : ''}" data-i="${i}" title="Click: go to it · Tick: keep it (Save reel and Make 4K use only the ticked ones)"><img src="${src(p)}" alt="">`
        + `<div><input type="checkbox"${drop.has(p.png) ? '' : ' checked'}>${at(a)} - ${at(p.frames.at(-1)[1] + 1)}</div></div>`; }).join('');
    TX = t; t.items = t.sprites.map((p, i) => $(`.tp[data-i="${i}"]`)); tlCount();
    try { $('#tx').checked = localStorage[txKey] !== 'off'; } catch {} $('#tx').disabled = false; $('#txn').textContent = 'On-video text'; $('#txl').hidden = false; });
}
const tlCount = () => $('#tln').textContent = `Text pieces · ${TX.sprites.length - drop.size} of ${TX.sprites.length} kept`;
$('#tl').onclick = e => { const it = e.target.closest('.tp'); if (!it) return; const i = +it.dataset.i, p = TX.sprites[i];
  if (e.target.type === 'checkbox') { e.target.checked ? drop.delete(p.png) : drop.add(p.png); it.classList.toggle('off', !e.target.checked); tlCount(); return post('/textdrop', {reel, drop: [...drop]}); }
  const a = p.frames[0][0], im = $('#ov').children[i]; P.pause(); pick(D.shots.findIndex(s => a < s.end_f), false); seek(a);  // its first frame, paused, the piece outlined a moment
  im.classList.add('hi'); setTimeout(() => im.classList.remove('hi'), 1500); };
$('#tx').onchange = () => { try { localStorage[txKey] = $('#tx').checked ? 'on' : 'off'; } catch {} };
function pick(i, play = true) {
  cur = Math.max(0, Math.min(D.shots.length - 1, i)); whole = false; const s = D.shots[cur], c = D.cuts[cur - 1];
  seek(s.start_f); if (play) P.play().catch(() => {}); mark(cur);
  $('#info').innerHTML = `Shot ${cur + 1} of ${D.shots.length}: ${fmt(T(s.start_f))} - ${fmt(T(s.end_f))} · ${s.len_s.toFixed(2)} s · frames ${s.start_f}-${s.end_f - 1}`
    + (c ? ` · <span class="${c.sure ? '' : 'q'}">cut ${c.sure ? '' : '(unsure) '}by ${c.by.join('+')}</span>` : '')
    + (s.src ? `<br>from ${name(s)} @ ${fmt(s.start)} · score ${s.score.toFixed(2)}${s.low ? ' <span class="q">(low: best found)</span>' : ''}`
      + (s.scores && Object.keys(s.scores).length > 1 ? ` (${Object.entries(s.scores).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(' · ')})` : '') : '');
  $('#lcap').innerHTML = s.src ? ` ${name(s)} @ ${fmt(s.start)} · ${s.score.toFixed(2)}` : '';
}
$('#bar').onclick = e => { if (pend) return; const x = (e.clientX - $('#bar').getBoundingClientRect().left) / $('#bar').clientWidth * D.frames;
  pick(D.shots.findIndex(s => x < s.end_f)); };
$('#prev').onclick = () => !pend && pick(cur - 1); $('#next').onclick = () => !pend && pick(cur + 1); $('#play').onclick = () => pick(cur);
$('#all').onclick = () => { if (pend) return; whole = true; P.currentTime = 0; P.play().catch(() => {}); };
const toggle = v => v.paused ? v.play().catch(() => {}) : v.pause();
P.onclick = $('#lplay').onclick = () => P.paused ? pick(cur) : P.pause(); Q.onclick = $('#rplay').onclick = () => toggle(Q);
const keys = () => !$('#slide').hidden ? {' ': splay, Enter: suse, Escape: unslide, ArrowRight: e => place(W.st + (e.shiftKey ? 1 : 0.1)), ArrowLeft: e => place(W.st - (e.shiftKey ? 1 : 0.1))}
  : !$('#look').hidden ? {' ': unlook, Escape: unlook, Enter: () => use(hi), s: () => slide(hi), ArrowRight: () => look(hi + 1), ArrowLeft: () => look(hi - 1)}
  : !$('#grid').hidden ? {' ': () => look(hi), Enter: () => use(hi), s: () => slide(hi), Escape: () => $('#gx').onclick()}
  : {ArrowRight: () => $('#next').onclick(), ArrowLeft: () => $('#prev').onclick(), ' ': () => toggle(P)};  // Slide, the preview, the grid, or the viewer
onkeydown = e => { const f = keys()[e.key.length === 1 ? e.key.toLowerCase() : e.key]; if (!f || (e.target.tagName === 'INPUT' && e.target.type !== 'range')) return; e.preventDefault(); f(e); };
// ---- editing: only when the page comes from the server (opened as a file it is view-only)
const post = (u, b) => fetch(u, {method: 'POST', body: JSON.stringify(b)}).then(r => r.json()), reel = `${D.template}/${D.video}`;
const ui = (choose, decide, msg = '') => { $('#edit').hidden = !(choose || decide || msg); $('#choose').hidden = !choose; $('#decide').hidden = !decide; $('#msg').textContent = msg; };
function compare(p) {  // current clip on the left (never touched here), 5 proposals on the right as tabs; null closes the split
  pend = p; $('#R').hidden = !p; $('#L .cap').hidden = !p; if (!p) return Q.removeAttribute('src'), Q.load();
  $('#tabs').hidden = p.cs.length <= 1; $('#tabs').innerHTML = p.cs.map((c, j) => `<span class="tab" data-i="${j}">${j + 1}${c.low ? '<span class="lo">*</span>' : ''}</span>`).join("");
  document.querySelectorAll('#tabs .tab').forEach(t => t.onclick = () => { pend.tab = +t.dataset.i; show(); });
  show(); ui(false, true);
}
function show() {
  const c = pend.cs[pend.tab]; Q.src = c.preview; Q.muted = P.muted; Q.loop = $('#rloop').checked;
  $('#rcap').innerHTML = ` ${name(c)} @ ${fmt(c.start)} · ${c.score.toFixed(2)}${c.low ? ' <span class="q">(low)</span>' : ''}`;
  document.querySelectorAll('#tabs .tab').forEach((t, j) => t.classList.toggle('on', j === pend.tab));
}
$('#rloop').onchange = () => Q.loop = $('#rloop').checked;
async function wait(done) { const s = await (await fetch('/status')).json();  // follows a replace / keep job on the server
  if (s.stage === 'working') { ui(false, false, s.msg); return setTimeout(() => wait(done), 400); }
  s.stage === 'error' ? ui(!pend, !!pend, 'Failed: ' + s.msg) : done(s); }
function replace(m, path = '', start = 0) { const i = cur, first = !pend;
  if (first) { P.pause(); pick(i, false); }  // the left is set once, paused on the shot's first frame; try-agains never touch it
  post('/replace', {reel, shot: i, mode: m, path, start}).then(() => wait(s => compare({i, cs: s.cands, tab: 0}))); }
document.querySelectorAll('#choose [data-m]').forEach(b => b.onclick = () => replace(b.dataset.m));
// ---- Choose a video: every video in the footage folder as icons, like Finder's Icons view. Click highlights; Enter, double-click or Select uses it; Space previews it whole
const LV = $('#lv'), gs = $('#gsize'); let G = null, hi = -1;  // the grid (its tiles, the shot it was opened for), the highlighted tile
try { gs.value = localStorage.gsize || 200; } catch { gs.value = 200; }  // the icon size you left it at
gs.oninput = () => { $('#tiles').style.setProperty('--s', gs.value + 'px'); try { localStorage.gsize = gs.value; } catch {} G?.tiles[hi]?.el.scrollIntoView({block: 'nearest'}); };
gs.oninput(); new ResizeObserver(() => G?.tiles[hi]?.el.scrollIntoView({block: 'nearest'})).observe($('#tiles'));  // a resized window keeps the highlighted video in view
function light(k, scroll = true) {  // highlight tile k; Select names it (off for a video shorter than the shot)
  G.tiles[hi]?.el.classList.remove('on'); hi = k; const t = G.tiles[k]; t.el.classList.add('on'); scroll && t.el.scrollIntoView({block: 'nearest'});
  $('#gsel').disabled = $('#gslide').disabled = !!t.short; $('#gsel').textContent = 'Select ' + t.name;
}
function paint(k) {  // tile k's icon is ready (img null: the video can't be read)
  const t = G.tiles[k]; if (t.dur === undefined) return; t.short = !(t.dur >= G.need); t.el.classList.toggle('short', t.short);
  t.el.title = `${t.path}\n${t.dur ? t.dur.toFixed(1) + ' s' : "can't read"}${t.short ? ' · shorter than this shot' : ''}`;
  t.el.firstChild.innerHTML = t.img ? `<img src="${t.img}">` : "can't read"; k === hi && light(k, false);
}
const seen = new Set(), io = new IntersectionObserver(es => es.forEach(e => seen[e.isIntersecting ? 'add' : 'delete'](+e.target.dataset.k)), {root: $('#tiles')});
const todo = () => { const free = k => G.tiles[k].dur === undefined && !G.tiles[k].busy; for (const k of seen) if (free(k)) return k; return G.tiles.findIndex((t, k) => free(k)); };
async function fill(g) {  // icons not made yet, one at a time per worker: what's on screen first, then the rest in order
  for (let k; g === G && (k = todo()) >= 0;) { const t = g.tiles[k]; t.busy = true; const r = await fetch(`/thumb?i=${k}`).then(r => r.json()).catch(() => ({}));
    if (g !== G) return; Object.assign(t, 'dur' in r ? r : {img: null, dur: null}); paint(k); }
}
async function browse() {
  const fresh = G?.shot !== cur;  // a new grid for this shot; the same shot reopens as you left it
  if (fresh) {
    ui(false, false, 'Loading your videos…'); const r = await fetch(`/list?reel=${encodeURIComponent(reel)}&shot=${cur}`).then(r => r.json()).catch(e => ({error: String(e)}));
    if (r.error) return ui(!pend, !!pend, 'Failed: ' + r.error);
    G = {...r, shot: cur, top: 0}; hi = -1; seen.clear(); io.disconnect(); $('#gn').textContent = `${r.tiles.length} videos`;
    $('#tiles').replaceChildren(...r.tiles.map((t, k) => { t.el = Object.assign(document.createElement('div'), {className: 'v' + (k === r.cur ? ' cur' : ''),
      innerHTML: '<div class="im"></div><span></span>', onclick: () => light(k), ondblclick: () => use(k)});
      t.el.dataset.k = k; t.el.lastChild.textContent = t.name; io.observe(t.el); return t.el; }));
    r.tiles.forEach((t, k) => paint(k));
  }
  ui(false, false); $('#grid').hidden = false; document.activeElement?.blur(); $('#tiles').scrollTop = G.top;
  if (fresh && G.cur >= 0) { light(G.cur); G.tiles[G.cur].el.scrollIntoView({block: 'center'}); }  // the shot's own video, highlighted, in view
  if (fresh) setTimeout(() => { for (let n = 0; n < 4; n++) fill(G); }, 100);  // after the first layout, so what's on screen is known
}
const ungrid = () => { G.top = $('#tiles').scrollTop; $('#grid').hidden = true; };
function use(k) { const t = G.tiles[k]; if (!t || t.short) return; unlook(); ungrid(); replace('path', t.path); }
function look(k) {  // tile k's whole video, starting from its icon
  const t = G.tiles[k]; if (!t) return; light(k); LV.poster = t.img || ''; LV.src = `/v?i=${k}`; LV.play().catch(() => {});
  $('#lname').textContent = t.name + (t.short ? ' · shorter than this shot' : ''); $('#lsel').disabled = $('#lslide').disabled = !!t.short; $('#look').hidden = false; document.activeElement?.blur();
}
const unlook = () => { LV.pause(); LV.removeAttribute('src'); LV.load(); $('#look').hidden = true; };
LV.onerror = () => LV.getAttribute('src') && ($('#lname').textContent += " · this browser can't play it (Select still works)");
document.querySelectorAll('.browse').forEach(b => b.onclick = browse);
$('#gsel').onclick = $('#lsel').onclick = () => use(hi); $('#lx').onclick = unlook; $('#gx').onclick = () => { ungrid(); ui(!pend, !!pend); };
// ---- Slide: a window as long as the shot, slid along the video's frame strip; Play plays just the window under the reel's audio; the score is for reference only
const SV = $('#sv'), AU = new Audio(), clamp = st => Math.max(0, Math.min(W.dur - W.len - 0.05, st)); let W = null, sbusy = false;
async function slide(k = -1) {  // k: a grid tile; -1: the shot's own video, from its current moment
  if (k >= 0 && (!G.tiles[k] || G.tiles[k].short)) return; LV.pause(); document.body.style.cursor = 'progress';
  const r = await fetch(`/slide?reel=${encodeURIComponent(reel)}&shot=${cur}&i=${k}`).then(r => r.json()).catch(e => ({error: String(e)})); document.body.style.cursor = '';
  if (r.error) return alert('Slide: ' + r.error);
  W = {...r, k, st: r.start}; $('#strip').style.backgroundImage = `url("${r.strip}")`; $('#win').style.width = `${100 * r.len / r.dur}%`;
  SV.src = `/sv?k=${k}&t=${Date.now()}`; AU.src ||= `/t/${D.template}/audio.wav`; $('#slide').hidden = false; document.activeElement?.blur(); place(r.start);
}
function place(st) {  // the window starts at st: the big player shows that frame; a score follows once you stop
  stop(); W.st = clamp(st); $('#win').style.left = `${100 * W.st / W.dur}%`; SV.currentTime = W.st; clearTimeout(W.t); W.t = setTimeout(rescore, 500);
  $('#sname').textContent = `${W.name} · ${fmt(W.st)} - ${fmt(W.st + W.len)} of ${fmt(W.dur)}${W.bad ? " · this browser can't show it: slide by the strip" : ''}`; $('#sscore').textContent = 'scoring…';
}
async function rescore() {  // one score at a time; only the latest window's is shown
  if (sbusy) return W.t = setTimeout(rescore, 200); sbusy = true; const w = W, st = W.st;
  const r = await fetch(`/score?reel=${encodeURIComponent(reel)}&shot=${cur}&start=${st}`).then(r => r.json()).catch(e => ({error: String(e)})); sbusy = false;
  if (w === W && st === W.st) $('#sscore').textContent = r.error ? "score: can't tell" : `score ${r.score.toFixed(2)}`;
}
async function splay() {  // the window once, the video muted under the reel's audio for this shot; again stops it
  if (!SV.paused) return stop(); SV.currentTime = W.st; AU.currentTime = T(D.shots[cur].start_f);
  await Promise.all([SV, AU].map(m => m.seeking && new Promise(ok => m.addEventListener('seeked', ok, {once: true})))); SV.play().catch(() => {}); AU.play().catch(() => {});
}
const stop = () => { SV.pause(); AU.pause(); if (W) SV.currentTime = W.st; };
setInterval(() => { if (W && !SV.paused && SV.currentTime >= W.st + W.len) stop(); $('#splay').textContent = SV.paused ? '▶ Play' : '■ Stop'; }, 10);
$('#strip').onpointerdown = e => {  // drag the window, or click the strip to centre it there
  const box = $('#strip').getBoundingClientRect(), at = x => (x - box.left) / box.width * W.dur, off = e.target.id === 'win' ? at(e.clientX) - W.st : W.len / 2;
  place(at(e.clientX) - off); $('#strip').setPointerCapture(e.pointerId); $('#strip').onpointermove = m => place(at(m.clientX) - off);
  $('#strip').onpointerup = () => $('#strip').onpointermove = null;
};
function suse() { const k = W.k, st = W.st; unslide(); if (k >= 0) { unlook(); ungrid(); } replace('slide', '', st); }
const unslide = () => { stop(); clearTimeout(W?.t); SV.removeAttribute('src'); SV.load(); $('#slide').hidden = true; };
SV.onerror = () => { if (SV.getAttribute('src') && W) { W.bad = true; place(W.st); } };
$('#slideSame').onclick = () => slide(); $('#gslide').onclick = $('#lslide').onclick = () => slide(hi); $('#splay').onclick = splay; $('#suse').onclick = suse; $('#sx').onclick = unslide;
$('#rep').onclick = () => ui(true, false); $('#x').onclick = () => ui(false, false);
$('#againSame').onclick = () => replace('same', pend.cs[pend.tab].src);  // 5 more moments of the current tab's video
$('#againRand').onclick = () => replace('random');
$('#export').onclick = () => post('/export', {reel, text: txOn()}).then(() => wait(s => ui(false, false, `Saved: ${s.folder}`)));
$('#k4').onclick = () => post('/4k', {reel, text: txOn()}).then(() => wait(s => ui(false, false, `4K saved (plain, AI, AI smooth): ${s.folder}`)));
let caption = '';  // fetched now: the clipboard only takes a copy made right at the click
$('#capt').onclick = () => { const t = Object.assign(document.createElement('textarea'), {value: caption}); document.body.append(t); t.select();
  (navigator.clipboard?.writeText(caption) ?? Promise.reject()).catch(() => document.execCommand('copy')).finally(() => t.remove());
  $('#capt').textContent = 'Copied ✓'; setTimeout(() => $('#capt').textContent = 'Copy caption', 1500); };
$('#cancel').onclick = () => post('/cancel', {}).then(() => { compare(null); ui(false, false); pick(cur); });
$('#keep').onclick = () => { const i = pend.i; post('/keep', {tab: pend.tab}).then(() => wait(s => location.href = `${s.url}#shot=${i + 1}`)); };
pick((+location.hash.split('shot=')[1] || 1) - 1, false);
if (location.protocol.startsWith('http') && D.template) { $('#rep').hidden = $('#export').hidden = $('#k4').hidden = false;  // after a reload: a proposal still waiting, or a job still running
  fetch(`/caption?reel=${encodeURIComponent(reel)}`).then(r => r.json()).then(r => { caption = r.text || ''; $('#capt').hidden = !caption.trim(); }); txLoad();
  fetch('/status').then(r => r.json()).then(s => s.stage === 'preview' && s.reel === reel ? (pick(s.shot, false), compare({i: s.shot, cs: s.cands, tab: 0}))
    : s.stage === 'working' && wait(s => s.stage === 'preview' ? (pick(s.shot, false), compare({i: s.shot, cs: s.cands, tab: 0})) : location.href = s.url)); }
</script>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=0); ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--cookies-from-browser", metavar="BROWSER", help="for a Reel that needs a login: safari, chrome, edge, firefox")
    a = ap.parse_args()
    if a.cookies_from_browser: COOKIES["cookiesfrombrowser"] = (a.cookies_from_browser,)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H); url = f"http://127.0.0.1:{srv.server_port}/"
    print(f"template_reel: {url}  (Ctrl+C to stop)", flush=True)
    if not a.no_browser: webbrowser.open(url)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
