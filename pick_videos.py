"""Template reel: an Instagram Reel / YouTube Short's cuts + audio, filled with your own footage. One file, Windows or Mac, CPU only.
    python template_reel.py [--port N] [--no-browser] [--cookies-from-browser safari]        (Mac: python3.14 template_reel.py)
The first run installs the packages it needs into this Python; work is kept next to this file (templates/, .cache/). Needs ffmpeg 7+
(+ deno for YouTube Shorts). The page takes the link, your footage folder, the scoring models, tries and threshold; each shot gets
random clips of random videos until one clears the threshold (else the best, marked low), rendered under the reel's audio. The viewer
replaces clips; Save reel writes <footage folder>/template_reel_output/<reel id>/ (the reel + one clip per shot)."""
import argparse, datetime, importlib.util, json, os, random, re, shutil, statistics, subprocess, sys, threading, time, traceback, types, webbrowser
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
LOCK, LOADED, COOKIES, DOVER_FRAMES = threading.Lock(), {}, {}, {}
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


def template(link):  # the template folder + its cuts (detected again if the video or this script is newer, so CUT config changes bust the cache)
    with LOCK:
        d = fetch(link); cj = d / "cuts.json"
        if not cj.exists() or cj.stat().st_mtime < max((d / "video.mp4").stat().st_mtime, Path(__file__).stat().st_mtime): cj.write_text(json.dumps(cuts(d / "video.mp4"), indent=1))
        return d, json.loads(cj.read_text())


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
def videos(folder): return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in VIDEO_EXT and not p.name.startswith(".") and OUTPUT not in p.relative_to(folder).parts)  # all subfolders; skips hidden files + saved reels


def order(pool, rnd): return rnd.sample(pool, len(pool))  # ALL the videos as one shuffled flat list: each equally likely first, however deep its folder


def prefetch(link):  # as soon as a link is pasted: the template, its shot count and length (for the estimate)
    state["tpl"] = {"link": link, "status": "loading"}
    try: d, T = template(link); state["tpl"] = {"link": link, "status": "ready", "shots": len(T["shots"]), "dur": round(T["frames"] / T["fps"], 2)}
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


def models(keys):  # load the ticked scoring models once per process
    for k in keys:
        if k not in LOADED: busy(f"Loading {MODELS[k][0]}"); t = time.time(); LOADED[k] = MODELS[k][7](); note("load:" + k, time.time() - t)
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


def render(plan, fps, out, audio, a0=0.0):  # one ffmpeg call: each clip from its start, exactly its frames, under the audio from a0
    n = lambda p: p["end_f"] - p["start_f"]; t0 = time.time()
    graph = "".join(f"[{i}:v]{','.join(normalise(probe_video(Path(p['src'])), fps))},tpad=stop_mode=clone:stop={n(p)},"
                    f"trim=end_frame={n(p)},setpts=PTS-STARTPTS[v{i}];" for i, p in enumerate(plan))
    (gf := out.with_suffix(".graph.txt")).write_text(graph + "".join(f"[v{i}]" for i in range(len(plan))) + f"concat=n={len(plan)}:v=1:a=0[v]")
    ff([*sum([["-ss", f"{p['start']:.3f}", "-i", p["src"]] for p in plan], []), "-ss", f"{a0:.4f}", "-t", f"{sum(map(n, plan)) / fps:.4f}",
        "-i", audio, "-/filter_complex", gf, "-map", "[v]", "-map", f"{len(plan)}:a", "-c:v", "libx264", "-crf", "18",
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


def replace(reel, i, mode, path="", seed=None):  # 5 delta-stratified candidates for shot i: same = 5 moments in one video; random = 5 videos across ordering
    for r in S["readers"].values(): r.cap and r.cap.release()  # the previous replace's cv2 caps hold ffmpeg decoder threads; free them before probing fresh videos
    S["readers"].clear(); S["metas"].clear()
    data, d = editing(reel); s = data["shots"][i]; rnd = random.Random(seed); fps = data["fps"]
    L = (s["end_f"] - s["start_f"]) / fps; tries, thr = data.get("tries", TRIES), data["threshold"]
    keys = models(data.get("models", ["current"])); cands = []
    if mode == "path":
        p = Path(path.strip().strip('"'))
        if not p.is_file(): raise RuntimeError(f"not a video file: {p}")
        cands.append({**s, **search([p], L, tries, thr, rnd, f"Shot {i + 1}", keys)})
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


def keep(tab=0):  # the selected tab's clip becomes the shot's clip, and the reel is re-rendered with it right away
    reel, i, cs = S.pop("pending"); data, d = editing(reel); data["shots"][i] = {k: x for k, x in cs[tab].items() if k != "preview"}; return save(reel)


def save(reel):  # a new version of the reel (earlier versions are kept)
    data, d = editing(reel); busy("Updating the reel"); out = d / f"reel_{stamp()}.mp4"; plan = data["shots"]; render(plan, data["fps"], out, d / "audio.wav")
    url = publish(d, {**data, "video": out.name, "shots": plan}, out); shutil.rmtree(d / "previews", ignore_errors=True)
    state.update(stage="saved", url=url, path=str(out)); return out


def export(reel):  # Save reel: the final reel + every shot as its own numbered clip (with its audio), in <footage>/template_reel_output/<reel id>/
    data, d = editing(reel); src = d / data["video"]; out = Path(data["folder"]) / OUTPUT / src.stem; out.mkdir(parents=True, exist_ok=True)
    busy("Saving the reel and its clips"); shutil.copyfile(src, out / src.name); cuts = [s["start_f"] for s in data["shots"][1:]]
    ff(["-i", src, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",  # keyframes half a frame early: land on frame f, never f+1
        "-force_key_frames", ",".join(f"{(f - 0.5) / data['fps']:.5f}" for f in cuts), "-f", "segment", "-segment_frames", ",".join(map(str, cuts)),
        "-segment_start_number", "1", "-reset_timestamps", "1", out / f"{src.stem}_%02d.mp4"], "clips")
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
        if q[0] == "/models": return self.reply(model_info())
        if q[0] == "/estimate": return self.reply(estimate(int(a["shots"]), int(a["tries"]), float(a["threshold"]), max(1, int(a["videos"])),
                                                           [k for k in a.get("models", "current").split(",") if k in MODELS] or ["current"]))
        if q[0].endswith("_view.html") and (j := TEMPLATES / unquote(q[0][3:]).replace("_view.html", ".json")).is_file():  # always today's viewer
            return self.reply(VIEW.replace("/*DATA*/null", j.read_text()), "text/html; charset=utf-8")
        if q[0].startswith("/t/"): return self.send_file(TEMPLATES / q[0][3:])
        self.reply(b"not found", "text/plain", 404)

    def do_POST(self):
        a, route = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}"), self.path
        if route == "/template": threading.Thread(target=prefetch, args=(a["link"].strip(),), daemon=True).start(); return self.reply({"ok": True})
        if route == "/cancel": S.pop("pending", None); state.update(stage="idle", msg=""); return self.reply({"ok": True})
        if state["stage"] == "working": return self.reply({"error": "already running"})
        if route == "/keep" and "pending" not in S: return self.reply({"error": "nothing to keep"})
        job = {"/start": lambda: build(a["link"].strip(), Path(a["folder"].strip().strip('"')), int(a["tries"]), float(a["threshold"]),
                                       [k for k in a.get("models", ["current"]) if k in MODELS] or ["current"]),
               "/replace": lambda: replace(a["reel"], int(a["shot"]), a["mode"], a.get("path", "")), "/keep": lambda: keep(int(a.get("tab", 0))), "/export": lambda: export(a["reel"])}[route]
        def run():
            try: job()
            except Exception as e: traceback.print_exc(); state.update(stage="error", msg=last(e))
        busy("Starting"); threading.Thread(target=run, daemon=True).start(); self.reply({"ok": True})

    def send_file(self, p):  # with HTTP Range, so the browser can seek in the video
        p = Path(unquote(str(p))).resolve()
        if TEMPLATES.resolve() not in p.parents or not p.is_file(): return self.reply(b"not found", "text/plain", 404)
        size = p.stat().st_size; m = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        a, b = (int(m[1]), min(size - 1, int(m[2] or size - 1))) if m else (0, size - 1)
        self.send_response(206 if m else 200); self.send_header("Content-Type", self.guess_type(str(p))); self.send_header("Accept-Ranges", "bytes")
        if m: self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.send_header("Content-Length", str(b - a + 1)); self.end_headers()
        with p.open("rb") as f: f.seek(a); self.wfile.write(f.read(b - a + 1))

    def log_message(self, *a): pass


PAGE = r"""<!doctype html><meta charset="utf-8"><title>Template reel</title>
<style>
body{margin:0;font:14px system-ui,sans-serif;background:#111;color:#eee;display:grid;place-items:center;min-height:100vh} main{width:min(520px,92vw);display:flex;flex-direction:column;gap:16px} h1{font-size:18px;font-weight:600;margin:0}
label{display:flex;flex-direction:column;gap:5px;color:#aaa;font-size:12px} input[type=text]{background:#1b1b1b;color:#eee;border:1px solid #333;border-radius:8px;padding:9px 11px;font-size:14px} input:focus{outline:none;border-color:#e67e22}
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
<label>Footage folder (all subfolders are searched)<input type="text" id="folder" placeholder="paste the folder path"><small id="count"></small></label>
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
$('#folder').oninput = () => { clearTimeout(ftimer); folder = null; ready(); eta(); ftimer = setTimeout(async () => {
  const c = await (await fetch('/count?path=' + encodeURIComponent($('#folder').value))).json(); folder = c.error ? null : c;
  $('#count').className = folder?.videos ? 'ok' : 'bad'; $('#count').textContent = c.error || `${c.videos} videos in ${c.folders} folder(s)`; ready(); eta(); }, 300); };
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
</style>
<div id="bar"><span id="head"></span></div>
<div id="ctl"><button id="prev">◀ Prev</button><button id="play">Play shot</button><button id="next">Next ▶</button>
<label><input type="checkbox" id="loop" checked> Loop shot</label><button id="all">Play whole reel</button><button id="rep" hidden>Replace clip</button>
<button id="export" class="hot" hidden>Save reel</button><span id="info"></span></div>
<div id="edit" hidden><span id="choose"><button data-m="same">Same video</button><button data-m="random">Random video</button><button data-m="path">Choose a video</button>
  <input id="path" placeholder="paste a video file path, then Enter" hidden> <button id="x">✕</button></span>
  <span id="msg"></span><span id="decide" hidden><button id="keep" class="hot">Keep new</button><button id="againSame">Try again: same video</button>
  <button id="againRand">Try again: random video</button><button id="cancel">Cancel</button></span></div>
<div id="stage"><figure id="L"><div class="cap" hidden><b>Current</b><span id="lcap"></span><button id="lplay">▶ Play</button><label><input type="checkbox" id="lloop"> Loop</label></div>
  <video id="player" playsinline></video></figure>
  <figure id="R" hidden><div class="cap"><b>New</b><span id="rcap"></span><button id="rplay">▶ Play</button><label><input type="checkbox" id="rloop"> Loop</label></div>
  <div id="tabs"></div><video id="cand" playsinline></video></figure></div>
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
  $('#lplay').textContent = P.paused ? '▶ Play' : '❚❚ Pause'; $('#rplay').textContent = Q.paused ? '▶ Play' : '❚❚ Pause';
}, 10);
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
onkeydown = e => { if (e.target.tagName === 'INPUT') return; if (e.key === 'ArrowRight') $('#next').onclick(); else if (e.key === 'ArrowLeft') $('#prev').onclick();
  else if (e.key === ' ') { e.preventDefault(); toggle(P); } };
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
function replace(m, path = $('#path').value) { const i = cur, first = !pend;
  if (m === 'path' && !path.trim()) { $('#path').hidden = false; return $('#path').focus(); }
  if (first) { P.pause(); pick(i, false); }  // the left is set once, paused on the shot's first frame; try-agains never touch it
  post('/replace', {reel, shot: i, mode: m, path}).then(() => wait(s => compare({i, cs: s.cands, tab: 0}))); }
document.querySelectorAll('#choose [data-m]').forEach(b => b.onclick = () => replace(b.dataset.m));
$('#path').onkeydown = e => e.key === 'Enter' && replace('path');
$('#rep').onclick = () => ui(true, false); $('#x').onclick = () => ui(false, false);
$('#againSame').onclick = () => replace('same', pend.cs[pend.tab].src);  // 5 more moments of the current tab's video
$('#againRand').onclick = () => replace('random');
$('#export').onclick = () => post('/export', {reel}).then(() => wait(s => ui(false, false, `Saved: ${s.folder}`)));
$('#cancel').onclick = () => post('/cancel', {}).then(() => { compare(null); ui(false, false); pick(cur); });
$('#keep').onclick = () => { const i = pend.i; post('/keep', {tab: pend.tab}).then(() => wait(s => location.href = `${s.url}#shot=${i + 1}`)); };
pick((+location.hash.split('shot=')[1] || 1) - 1, false);
if (location.protocol.startsWith('http') && D.template) { $('#rep').hidden = $('#export').hidden = false;  // after a reload: a proposal still waiting, or a job still running
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
