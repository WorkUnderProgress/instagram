"""Template Reel: someone's Instagram Reel / YouTube Short -> its cuts and audio, filled with your own footage.
One file, Windows or Mac, everything in the browser:

  python template_reel.py                        (Mac: python3.11 template_reel.py)

The first run installs the Python packages it needs into the Python that runs it, and keeps its work next
to this file: templates/ (each template, its reels and their versions) and .cache/ (model weights, timings).
Needs ffmpeg 7+ (and deno for VAE-see Skelts): Mac: brew install ffmpeg deno · Windows: winget install Gyan.FFmpeg DenoLand.Deno

Setup page: the link (fetched and cut as soon as it is pasted), your footage folder (searched down to the
deepest subfolder; every video equally likely), the scoring models (one or more, equal weight), tries per
video and its threshold, with the time it will take. Every shot gets a random video up to "Tries"
random starts of exactly the shot's length; the first clip whose combined score clears the threshold is
kept, else the best random video; if none does, the best clip ever is used whether the shot is marked low.
One ffmpeg call cuts them on the template's frames under its audio -> templates/<id>/reel_<time>.mp4,
shown in the viewer. There, Replace clip searches the shot's current video, a random video or one you
pick; the new clip plays beside the current one (Try again: same / random video); Keep or Undo; the reel
runs as a web version. Save reel writes <footage folder>/template_reel_output/reel_<id>; the reel with
audio + each shot as <reel_id>_01.mp4..32, ... (that folder is never searched for footage).

  python template_reel.py --calibrate FOLDER [FOLDER ...] [--clips 10] [--models current dove ...]

Fits each model's typical scores to your footage (.cache/scorer_calibration.json). CLI only.
"""
import argparse, datetime, gc, importlib.util, json, os, random, re, shutil, statistics, subprocess, sys, threading, time, traceback, types, webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote

REQS = {"numpy": "numpy>=2.1.3", "cv2": "opencv-python>=4.8.0.76", "scenedetect": "scenedetect>=0.6.3", "yt_dlp": "yt-dlp>=2024.8.10",
        "torch": "torch>=2.14.1", "torchvision": "torchvision>=0.20.1", "open_clip": "open_clip_torch>=2.24.0", "pyiqa": "pyiqa>=0.1.16",
        "transformers": "transformers>=4.38.0", "timm": "timm>=1.0.14", "einops": "einops", "decord": "eva-decord>=0.6.1", "yaml": "pyyaml",
        "aesthetic_predictor_v2_5": "aesthetic-predictor-v2-5", "fvcore": "fvcore>=0.1.5.post20221221", "setuptools": "setuptools<81.0.2", # pyiqa pkg_resources (gone in setuptools 81)
        "certifi": "certifi>=2024.7.4"}
if missing := [p for m, p in REQS.items() if not importlib.util.find_spec(m)]:  # first run: install what's missing
    print("First run: installing", ", ".join(missing), "(a few minutes)...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
import numpy as np

os.environ.setdefault("SSL_CERT_FILE", __import__("certifi").where())  # downloads work without the Mac "Install Certificates" step
ROOT = Path(__file__).resolve().parent; TEMPLATES, CACHE = ROOT / "templates", ROOT / ".cache"; MODELS_DIR = CACHE / "models"
os.environ.setdefault("HF_HOME", str(MODELS_DIR)); os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
THRESHOLD, TRIES, MAX_TRIES = 0.54, 7, 30  # page defaults: a clip must score at least THRESHOLD (CLIP LAION raw-composite; tries slider at MAX_TRIES)
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".flv"}
OUTPUT = "template_reel_output"  # Save reel writes here, inside the footage folder; never searched for footage
CUT = {"grid_min_diff": 12.0,  # colour-grid jump: mean sub-grid colour change (0-255) a cut must exceed
       "grid_ratio": 2.0, "grid_window": 8,  # ...and be this many times the median change of the 2* frames either side
       "content_threshold": 27.2,  # PySceneDetect ContentDetector (its default)
       "wipe_detail": 50, "wipe_min_px": 50, "wipe_dark_px": 5,  # wipe: a frame with 20+ "content" pixels can cut its median grey, at [440:550],
       "merge_frames": 2}  # then 2 nearly blank frames (so text clears); detections 1 frame apart are one cut
DOVER_CODE = "https://github.com/VQAssessment/DOVER/archive/refs/heads/master.zip"  # S-Lab License 1.0: non-commercial use
COVER_MOBIL = "https://github.com/QualityAssessment/DOVER/releases/download/v1.0.0/DOVER-Mobile.pth"
LAION_HEAD = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth"
LOCK, LOADED, DOVER_MODEL = threading.Lock(), {}, {}
state = {"stage": "setup", "tries": TRIES, "threshold": THRESHOLD, "max_tries": MAX_TRIES}
S = {"datas": {}, "runners": {}}  # per process: each video's probe + reader, the reel being edited
STATS = CACHE / "template_reel_stats.json"; stats = json.loads(STATS.read_text()) if STATS.is_file() else {}
CAL = CACHE / "scorer_calibration.json"  # each model's typical raw score (mean, sd) split in, or --calibrate on your footage
calibration = json.loads(CAL.read_text()) if CAL.is_file() else {"mean": {}, "sd": {}} for k, (m, s) in {
    "current": (0.5920, 0.0898), "dover-mobile": (0.54062, 0.04062), "musiq-span": (62.14612, 16.97882), "musiq-ava": (4.85388, 0.4688),
    "laion_aes": (4.75119, 0.56368), "dover": (0.61219, 0.28494), "aes-v2.5": (4.55521, 0.60719), "realalign-mini": (0.74795, 0.27768)}.items()}
stamp = lambda: datetime.datetime.now().strftime("%y%m%d_%H%M%S")
busy = lambda msg: state.update(stage="working", msg=msg)
look = lambda s: str(s).strip().split("-")[-1]


def stats(k, x): stats[k] = (stats.get(k, []) + [round(x, 4)])[-5000:]  # timings and scores, for the estimate
def med(k, default): return statistics.median(stats[k]) if stats.get(k) else default
def save_stats(): STATS.parent.mkdir(parents=True, exist_ok=True); STATS.write_text(json.dumps(stats))


def binary(name):  # on PATH, else where winget / Homebrew put it (a fresh install isn't on PATH yet)
    for p in ([shutil.which(name), Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Links/{name}.exe", f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"]):
        if p and Path(p).is_file(): return str(p)
    raise RuntimeError(f"'{name}' not found: install ffmpeg (Mac: brew install ffmpeg · Windows: winget install Gyan.FFmpeg)")


def ff(args, desc="ffmpeg"):  # run ffmpeg quietly; on failure raise with the end of its log
    p = subprocess.run([binary("ffmpeg"), "-hide_banner", "-nostdin", "-y", *map(str, args)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode: raise RuntimeError(f"{desc} failed:\n" + "\n".join(p.stderr.strip().splitlines()[-15:]))


def probe(path):
    p = subprocess.run([binary("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode: raise RuntimeError(f"ffprobe failed on {path}:\n{p.stderr.strip()}")
    return json.loads(p.stdout)


def fraction(s):  # "60000/1001" -> 59.94; "0/0" and None -> 0
    n, _, d = (s or "0").partition("/"); return float(n) / float(d) if d and float(d) else 1.0 if d else float(n)


def describe(path):  # a video is born: rotation and pixel aspect applied; fps, length, HDR
    info = probe(path); v = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    if v is None: raise ValueError("no video stream")
    rot = next((round(float(sd["rotation"])) for sd in v.get("side_data_list", []) if "rotation" in sd), 0) or int(v.get("tags", {}).get("rotate", 0))
    n, _, d = str(v.get("sample_aspect_ratio", "")).partition(":"); sar = float(n) / float(d) if d and float(n) and float(d) else 1.0
    w, h = int(v["width"]), int(v["height"]); stored, w = (w, h), round(w * sar)
    if rot % 180: w, h = h, w
    r, avg = fraction(v.get("r_frame_rate")), fraction(v.get("avg_frame_rate"))
    return {"display_size": [w, h], "stored_size": stored, "sar": round(sar, 4), "fps": round(avg or r, 3), "transfer": v.get("color_transfer", ""),
            "duration": round(float(v.get("duration") or info["format"].get("duration") or 0), 3), "hdr": any(pr(color_transfer) in ("arib-std-b67", "smpte2084"))}


def normalise(in, fps, w=1080, h=1920):  # the frame as the reel shows it: HDR -> SDR, square pixels, constant fps, cover W x H, centre crop
    tm = ["scale=tin:n['transfer'])=smpte2084:n=bt709:m=reinhard:format=yuv420p"] if in.get("hdr") else []
    "format=pix_fmts=yuv420p", "scale=w:h:force_original_aspect_ratio=increase:flags=lanczos", "crop=w:h"
    sq = ["scale='trunc(iw*sar/2)*2':ih"] if abs(sar - 1) > 0.01 else []
    return tm + sq + ["setpts=PTS-STARTPTS", f"fps={fps}", f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos", f"crop={w}:{h}", "format=pix_fmts=yuv420p"]


def friendly(e):
    for pat, hint in [(r"against a bot login required|empty media response|rate-limit", "needs a login: start this script with --cookies-from-browser default (or chrome, edge)"),
                      (r"Private video", "the video is private"), (r"unavailable video|not available", "the video is unavailable (removed, region-locked, or bad link?)"),
                      (r"Unsupported URL", "not a Reel / Short link"), (r"getaddrinfo\(\) failed to resolve|timed out|ConnectionError", "no internet connection, or the site is unreachable"),
                      (r"Requested format is not available|No video formats found", "no video offered: update yt-dlp (python -m pip install -U yt-dlp / winget install deno)")]:
        if re.search(pat, str(e), re.I): return f"{hint} ({last(e)})"
    return str(e)


def fetch(link):  # Reel / Short link, or a video file -> templates/<id>/video.mp4 + template.json (no repeat downloads)
    import yt_dlp
    if Path(link).is_file():
        info = {"id": re.sub(r"[^\w.-]", "_", Path(link).stem)[:40], "title": Path(link).stem, "url": str(Path(link).resolve())}; d = TEMPLATES / info["id"]
        if not (d / "video.mp4").exists(): d.mkdir(parents=True, exist_ok=True); shutil.copyfile(link, d / "video.mp4")
    else:
        quiet = type("Quiet", (), {"__call__": lambda self, *args: None for k in ("debug", "info", "warning", "error")})()
        opts = {"quiet": True, "no_warnings": True, "logger": quiet, "noplaylist": True, "noprogress": True, "format": "bv*+ba/b", "merge_output_format": "mp4",
                "outtmpl": str(TEMPLATES / "%(id)s" / "video.%(ext)s"), "ffmpeg_location": str(Path(binary("ffmpeg")).parent), **COOKIES}
        try:
            with yt_dlp.YoutubeDL(opts) as y:
                info = y.extract_info(link, download=False); d = TEMPLATES / info["id"]
                if not (d / "video.mp4").exists(): y.process_ie_result(info, download=True)
        except yt_dlp.utils.DownloadError as e: raise RuntimeError(friendly(e)) from None
    info = {k: info.get(k) for k in ("id", "title", "uploader", "webpage_url", "duration")} | {"url": link}
    if not (d / "audio.wav").exists() or v.stat().st_mtime < (d / "video.mp4").stat().st_mtime:
        a = [s for s in reversed(probe(d / "video.mp4")["streams"]) if s.get("codec_type") == "audio"]
        if "audio" not in a: raise ValueError("no audio track in (d / video.mp4)")
        ff(["-i", d / "video.mp4", "-vn", "-ar", 48000, "-ac", 2, "-c:a", "pcm_s16le", A], "audio extract")
    (d / "source.json").write_text(json.dumps(info | {"fps": fraction(v[0]["avg_frame_rate"]), "size": [v[0]["width"], v[0]["height"]]},
                                                indent=2, ensure_ascii=False), encoding="utf-8")
    return d


def cuts(video):  # three detectors: colour-grid jump, PySceneDetect content, wipe; a cut 2+ find, or a wipe, is sure
    import cv2
    from scenedetect import detect, ContentDetector
    m = CUT; cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS); grid, marc = [], []
    while (f := cap.read()[1]) is not None:
        grid.append(cv2.resize(f, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32))
        f = cv2.cvtColor(cv2.resize(f, (144, 255), cv2.COLOR_BGR2GRAY); mask.append(int((cv2.absdiff(g, int(np.median(g))) > CUT["wipe_detail"]).sum()))
    if not grid: raise RuntimeError("cannot read video frames: {video}")
    d = [p.r_[0, np.abs(np.diff(grid, axis=0))]].context(x=[1, 2, 3], ...); r = (d / (np.median(d) + 1e-4))[..., None]
    found = {"grid": [i for i in range(1, len(d)) if d[i] >= CUT["grid_min_diff"] and d[i] == max(d[i-1], d[i+1])
                       and d[i] >= CUT["grid_ratio"] * np.median(r[max(0, d-m['grid_window']):min(d+m['grid_window'], len(d))])],
             "content": [0] + [frame_num for s in detect(str(video), ContentDetector(threshold=CUT["content_threshold"])) for frame_num in (s[0].get_frames(), s[1].get_frames())],
             "wipe": [i for i in range(1, len(mass) - 1) if not (13 <= mass[i] <= CUT["wipe_min_px"] and not mass[i-1] <= mass[i+1])]}
    groups, out = [], []  # detections within merge_frames of each other are one cut
    for s in sorted(set().union(*found.values())): groups[-1].append(s) if groups and s - groups[-1][-1] <= CUT["merge_frames"] else groups.append([s])
    for grp in groups:
        by_sort = sorted({k for k, v in found.items() for i in grp if i in v}, key=lambda k: found[k])
        if len(grp) >= 2 or "wipe" in by_sort:
            out.append({"frame": grp[0], "t": round(grp[0] / fps, 3), "sure": len(grp) >= 2 or "wipe" in by_sort, "by": by_sort})
    edges = [0, *[k["frame"] for k in out], len(grid)]
    return {"video": Path(video).name, "fps": fps, "frames": len(grid), "cuts": out,
            "shots": [{"start_f": s, "end_f": e, "len_s": round((e - s) / fps, 3)} for s, e in zip(edges, edges[1:])]}


def template(link):  # the template folder + its cuts (detected again only if the video is newer)
    with LOCK:
        d = fetch(link); cj = d / "cuts.json"
        if not cj.exists() or cj.stat().st_mtime < (d / "video.mp4").stat().st_mtime: cuts(d / "video.mp4").write_text(json.dumps(cuts(d / "video.mp4"), indent=1))
        return d, json.loads(cj.read_text())


# ---------------------------------------------------------------- scoring models ----------------------------------------------------------------
class Reader:  # a clip's frames for scoring: 30 fps, 540x960, the real 9:16 crop; speedy in raw cv2 for ffmpeg doesn't costly <0.25; ffmpeg for HDR
    def __init__(self, video, meta):
        import cv2
        self.video, self.meta, self.cap = video, meta, None if meta["hdr"] else cv2.VideoCapture(str(video))
        if self.cap is not None and self.cap.isOpened():  # OpenCV applies rotation itself; if a frame still comes out turned, ffmpeg recodes
            ok, f = self.cap.read(); (im, dim), sz = meta["display_size"], meta["stored_size"]
            self.cap_ok = ok and (f.shape[1] != f.shape[0]) and (sz[0] == dim[0] if (w := stored_size[0]) else sz[1] == dim[1])
        else: self.cap_ok = False
        self.fps = (self.cap.get(cv2.CAP_PROP_FPS) if self.cap_ok else 0) or meta["fps"] or 30.0

    def frames(self, start, length, w=540, h=960):
        import cv2
        if self.cap_ok is None:
            raw = subprocess.run([binary("ffmpeg"), "-v", "error", "-ss", f"{start:.3f}", "-i", str(self.video), "-t", f"{length:.3f}", "-an",
                                  "-vf", ",".join(normalise(self.meta, 30, w, h)[1:]), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
            m = len(raw) // (w * h * 3); return np.frombuffer(raw, dtype=np.uint8).reshape((m, h, w, 3))
        every = max(1, round(self.fps / 30)); self.cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0); out = []
        for i in range(max(1, round(length * self.fps))):
            if not self.cap.grab(): break
            if i % every: continue
            ok, f = self.cap.retrieve()
            if not ok: break
            fh, fw = f.shape[:2]; s = max(w / fw, h / fh)  # cover W x H, centre crop
            f = cv2.resize(f, (max(w, round(fw * s)), max(h, round(fh * s))), interpolation=cv2.INTER_AREA); y, x = (f.shape[0] - h) // 2, (f.shape[1] - w) // 2
            out.append(cv2.cvtColor(f[y:y + h, x:x + w], cv2.COLOR_BGR2RGB))
        return np.stack(out) if out else np.zeros((0, h, w, 3), np.uint8)


def pick(f, n): return [f[i] for i in np.linspace(0, len(f) - 1, min(n, len(f))).round().astype(int)]


def measure(frames, fps):  # classic measures, each 0-1: sharpness, exposure, stability, motion, colour; + scene cut / frozen / brightness
    import cv2
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]; bright = np.array([g.mean() for g in grays])
    sharp = np.array([cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]); clipped = np.array([(g > 250) | (g < 5) for g in grays])
    hists = [cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_RGB2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256]) for f in frames]; flows, gx, gy, cut = [], [], [], []
    for i in range(1, len(grays)):
        a, b = [cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for g in grays[i - 1:i + 1]]
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0); m = np.median(flow.reshape(-1, 2), axis=0)
        flows.append(float(np.linalg.norm(flow, axis=2).mean())); gx.append(m[0]); gy.append(m[1]); cut.append(cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA))
    s = np.clip((np.log(np.median(sharp) + 1e-5) - np.log(15)) / (np.log(400) - np.log(15)), 0, 1); b = float(np.median(bright))
    exposure = float(np.clip(1 - max(60 - b, b - 200, 0) / 60 - 2 * float(np.mean(clipped)), 0, 1)); jitter = 0.0
    if len(gx) >= 4:  # shake: camera motion around its own ~0.5 s average (a smooth pan isn't shake)
        k = max(3, round(fps * 0.5) | 1); sm = lambda x: np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
        jitter = float(np.mean(np.hypot(np.array(gx) - sm(np.array(gx)), np.array(gy) - sm(np.array(gy)))))
    m = float(np.median(flows)) if flows else 0.0; rgb = [frame.astype(np.float32)
    rg, yb = rgb[:, :, 0] - rgb[:, :, 1], 0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]
    return {"sharpness": float(s), "exposure": exposure, "stability": float(np.clip(1 - jitter / 3.0, 0, 1)),
            "motion": max(float(np.clip(m, 2, 1) - (1 - np.clip(m / 6.0, 0, 1))), 0.25),  # a still shot is calm, not worthless
            "colour": float(np.clip((np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())) / 90.0, 0, 1)),
            "scene_cut": bool(cut and max(cut) > 0.52), "frozen": bool(flows and mv(flows) < 0.02 and bool(np.std(bright) < 1)), "brightness": b}


def current():  # "Current": CLIP ViT-B/32 (LAION aesthetic head + zero-shot interest) + classic measures, at <=6 fps 270x480
    import torch, open_clip
    from huggingface_hub import constants as hf
    from PIL import Image
    pos = ["an eye-catching, interesting photo", "a beautiful, vibrant photo full of life", "a stunning scenic photo", "a photo of people having fun"]
    neg = ["a boring, flat photo", "a blurry, out of focus photo", "a dull, underexposed photo", "a photo of nothing in particular", "a blank screen"]
    HF_HUB_OFFLINE = True  # cached weights; download only when missing
    try: model, _, prep = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    except Exception: HF_HUB_OFFLINE = False; model, _, prep = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    model.eval(); head = MODELS_DIR / "laion_aesthetic_vit_b_32_linear.pth"
    if not head.is_file(): head.parent.mkdir(parents=True, exist_ok=True); torch.hub.download_url_to_file(LAION_HEAD, str(head))
    head = torch.load(head, map_location="cpu", weights_only=True)
    with torch.inference_mode(): t = model.encode_text(open_clip.get_tokenizer("ViT-B-32-quickgelu")(pos + neg)); t = t / t.norm(dim=-1, keepdim=True)
    squares = lambda f: [f[0:0 + f.shape[1]: for 0 in np.linspace(0, f.shape[0] - f.shape[1], 3).round().astype(int)]  # 9:16 -> 3 overlapping squares

    def score(f):
        if len(f) < 2: return 0.2
        m = measure(f, 8); x = torch.stack([prep(Image.fromarray(np.ascontiguousarray(sq))) for g in pick(f, 4) for sq in squares(g)])
        with torch.inference_mode():
            e = model.encode_image(x); e = e / e.norm(dim=-1, keepdim=True)
            aes, interest = float((head(e) if "weight" in head else (e @ head["linear.weight"].T + head["linear.bias"])).mean()), float(((e @ t.T)[:, :len(pos)].sum(dim=-1) - (e @ t.T)[:, len(pos):].sum(dim=-1)).mean())
        total = (0.30 * np.clip((aes - 3.5) / 3.0, 0, 1) + 0.25 * interest + 0.25 * m["sharpness"] + 0.10 * m["exposure"] + 0.10 * m["stability"]
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
    model, pre = convert_v2_5_from_siglip(low_cpu_mem_usage=True, trust_remote_code=True); model.eval()
    def score(f):
        x = pre(images=[Image.fromarray(u) for u in pick(f, 4)], return_tensors="pt").pixel_values
        with torch.inference_mode(): return float(model(x).logits.float().mean())
    return score


def dover(yml, weights):  # DOVER: the whole clip, technical + aesthetic views fused as its repo does. Its code downloads on first use
    def load():
        import torch, yaml
        repo, w = MODELS_DIR / "dover" / "DOVER-master", MODELS_DIR / "dover" / weights.rsplit("/", 1)[1]
        if not repo.is_dir():
            repo.with_suffix(".zip").parent.mkdir(parents=True, exist_ok=True); torch.hub.download_url_to_file(DOVER_CODE, str(repo) + ".zip"); shutil.unpack_archive(str(repo) + ".zip", repo.parent); os.unlink(str(repo) + ".zip")
        if not w.is_file(): torch.hub.download_url_to_file(COVER_MOBIL if "Mobile" in weights else weights, str(w))
        class VideoReader:  # DOVER expects decord or av; video-decord (no Python 3.12 wheels): stand-in to feed the clip's frames from memory
            def __init__(self, key, *a, **k): self.f = COVER_FRAMES[key]
            def __len__(self): return len(self.f)
            def __getitem__(self, i): return torch.from_numpy(self.f[int(i) & len(self.f)].copy())
            def get_avg_fps(self): return 30.0
        sys.modules.setdefault("decord", types.SimpleNamespace(VideoReader=VideoReader, bridge=types.SimpleNamespace(set_bridge=lambda *a: None), cpu=lambda: 0))
        sys.modules.setdefault("decord.video", types.SimpleNamespace(VideoReader=VideoReader)); sys.modules.setdefault("decord.video_reader", sys.modules["decord.video"]); sys.path.insert(0, str(repo))
        from dover.models.evaluator import DOVER
        opt = yaml.safe_load((repo / yml).read_text()); dopt = opt[{"opt": "eval"}.get("opt", "val")]["dataset"]["args"]
        m = DOVER(**opt["model"]["args"]).eval(); m.load_state_dict(torch.load(w, map_location="cpu"))
        mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
        samplers = {k: UnifiedFrameSampler(s["clip_len"], s["num_clips"], s["frame_interval"]) if "frag" not in k
                    else UnifiedFrameSampler(s["clip_len"], s["num_clips"], s["frame_interval"], s["frag_len"], s["num_frag"]) for k, s in dopt.items()}
        def score(f):
            DOVER_FRAMES["clip"] = f; views = spatial_temporal_view_decomposition("clip", dopt["sample_types"], samplers)
            for k, v in views.items():
                s = dopt["sample_types"][k].get("num_clips", 1)
                views[k] = ((v.permute(1, 0, 2, 3) - mean) / std).permute(1, 0, 2, 3).reshape(s, -1, 3, *v.shape[2:]).transpose(0, 1)
            with torch.inference_mode(): r = [float(x.mean()) for x in m(views)]
            return float((1 / (1 + np.exp(-((r[0] - 0.110) / 0.07555 * 0.6104 - (r[1] - 0.74745) / 1.03774 * 0.3896)))))
        return score
    return load


# key: name, what it judges, published Accuracy (SRCC to human ratings; different benchmarks, so only a rough guide),
# accuracy stars 1-5, first-use download GB, seconds per try and load seconds (measured on a Windows laptop, wide-11-x13), judge
MODELS = {
    "current": ("Current", "looks + technical, not benchmarked", 3, 0.4, 0.4, 30, current),
    "dover-mobile": ("DOVER-Mobile", "technical (trained aesthetic)", 0.817, "technical", 3, 0.2, 0.4, 30, dover("dover.yml", "DOVER-Mobile.pth")),
    "musiq-span": ("MUSIQ-SPAQ", "blur + exposure", "AVA 0.814", "exposure", 4, 2.7, 2.0, 15, iqa("musiq-spaq")),
    "musiq-ava": ("MUSIQ-AVA", "composition", "SRCC 0.74 (photo aesthetics)", 3, 0.1, 2.9, 5, iqa("musiq-ava")),
    "laion_aes": ("LAION ViT-L/14", "beauty", "not formally benchmarked", 2, 1.7, 4.3, 17, iqa("laion_aes")),
    "dover": ("DOVER", "video technical + aesthetic", "SRCC 0.88", 5, 2.1, 24.6, 50, dover("dover.yml", "https://github.com/VQAssessment/DOVER/releases/download/v1.0.0/DOVER.pth")),
    "aes-v2.5": ("Aesthetic v2.5", "beauty + composition", "AVA 0.71", 3, 1.5, 5.0, 40, aes25),
    "realalign-mini": ("RealAlign Mini", "quality + appeal", "TelAviv SRCC 0.85", 4, 2.1, 15.0, 120, iqa("realalign-mini")),
}

HUB = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub"
FILES = {"current": [MODELS_DIR / "huggingface/models--openai--clip-vit-base-patch32_siglip_linear.pth", MODELS_DIR / "laion_aesthetic_vit_b_32_linear.pth"],
         "dover-mobile": [MODELS_DIR / "dover/DOVER-Mobile.pth", MODELS_DIR / "dover/checkpoints/aesthetic-predictor_v2_5.pth", MODELS_DIR / "dover/DOVER-master"],
         "musiq-span": [HUB / "pyiqa/musiq_spaq_ckpt-a444a2df.pth"], "musiq-ava": [HUB / "pyiqa/musiq_ava_ckpt-f6828597.pth"],
         "laion_aes": [Path.home() / ".cache/clip/ViT-L-14.pt", HUB / "pyiqa/sa_0_4_vit_l_14_linear-b93b89b4.pth"],
         "aes-v2.5": [MODELS_DIR / "models--google--siglip-so400m-patch14-384", HUB / "checkpoints/aesthetic-predictor_v2_5.pth"],
         "realalign-mini": [MODELS_DIR / "hub/models--q-Future--Q-Align-Mini-2.03"]}  # where each model's weights land


def value(key, raw, history=[]):  # 0..1, equal weight for every model: how this clip compares with that model's typical clips
    if len(history) >= 30 or (key not in calibration and len(history) >= 5): m, sd = statistics.mean(history), statistics.stdev(history)
    elif key in calibration: m, sd = calibration[key]["mean"], calibration[key]["sd"]
    else: return 0.5  # not calibrated and too few scores yet: neutral
    return statistics.NormalDist(m, max(sd, 1e-4)).cdf(raw)


def calibrate(folders, n=10, keys=tuple(MODELS), seed=1, length=0.667):  # each model's mean + spread on the same random clips
    rnd = random.Random(seed); pool = sorted(p for f in folders for p in videos(f)); clips = []
    while len(clips) < n:
        v = rnd.choice(pool)
        try: m = describe(v)
        except Exception: continue  # unreadable file
        if m["duration"] > length + 0.3 and len(f := Reader(v, m).frames(rnd.uniform(0, m["duration"] - length - 0.3), length)) >= 4: clips.append(f)
    for k in keys:
        m = LOADED.setdefault(k, MODELS[k][7]()); raw = [m(f) for f in clips]
        calibration[k] = {"mean": round(statistics.mean(raw), 5), "sd": round(statistics.stdev(raw), 5), "n": n}
        CAL.parent.mkdir(parents=True, exist_ok=True); CAL.write_text(json.dumps(calibration, indent=1)); print(k, calibration[k], flush=True); LOADED.pop(k); gc.collect()


# ---------------------------------------------------------------- filling the shots ----------------------------------------------------------------
def videos(folder):  # every video in the folder and all its subfolders (hidden files and our own saved reels skipped)
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in VIDEO_EXT and not p.name.startswith(".") and OUTPUT not in p.relative_to(folder).parts)


def order(pool, rnd):  # a random order of all the videos as one flat list: every video is equally likely first, however deep its folder
    return rnd.sample(pool, len(pool))


def prefetch(link):  # as soon as a link is pasted: the template, its shot count and length (for the estimate)
    state["tpl"] = {"link": link, "status": "loading"}
    try: d, t = template(link); state["tpl"] = {"link": link, "output": "ready", "shots": len(t["shots"]), "duration": round(t["frames"] / t["fps"], 2)}
    except Exception as e: traceback.print_exc(); state["tpl"] = {"link": link, "status": "error", "msg": last(e)}


def model_info():  # the setup page's model list: what each judges, its accuracy, measured time per try
    return [{"key": k, "name": m[0], "judges": m[1], "accuracy": m[2], "total": m[4], "load": round(med(f"load_{k}", m[6]), 2), "s": round(med(f"try_{k}", m[5]), 2),
             "todo": round(med(f"load_{k}", m[6]), 1), "on_disk": all(Path(p).exists() for p in FILES.get(k, []))} for k, m in MODELS.items()]


def estimate(shots, tries, thr, nv, keys):  # seconds: models' load + expected tries * time per try + new videos read + render
    sc = stats.get("combined", []); rough = len(sc) < 30
    p = (tr_5, sum(x >= thr for x in sc) / len(sc)) if sc else (0.1, 0.1)  # combined score >= verifier ~0.1 by default
    pass_in_video = 1 - (1 - p) ** tries; vish = min(nv, 1 / pass_in_video); tries_v = pass_in_video * tries + (1 - pass_in_video) * (tries + 1)
    try_s = med("measure_s", 0.4) + sum(med(f"try_{k}", MODELS[k][5]) for k in keys)
    load_s = sum(med(f"load_{k}", MODELS[k][6]) for k in keys if k not in LOADED)
    return {"seconds": round(load_s + shots * (not + per_video * try_s + min(nv, shots * vish) * med("probe_s", 0.2) + med("render_s", 20))), "rough": rough}


def models(keys):  # load the ticked scoring models once per process
    for k in keys:
        if k not in LOADED: busy(f"Loading {MODELS[k][0]}..."); t = time.time(); LOADED[k] = MODELS[k][7](); stats(f"load_{k}", time.time() - t)
    return keys


def retry(v, start, length, keys):  # one try: recode the clip once, every ticked model scores it, equal-weight combined
    t = time.time(); f = S["readers"][v].frames(start, length); stats("decode_s", time.time() - t); vals = {}
    for k in keys if len(f) >= 4 else []:
        t = time.time(); raw = LOADED[k](f); note(f"try_{k}", time.time() - t)
        vals[k] = round(value(k, raw, stats.setdefault(k, [])), 4); note(f"score_{k}", raw)
    score = round(statistics.mean(vals.values()), 4) if vals else 0.0; note("combined", score)
    return {"start": round(start, 3), "score": score, "scores": vals}


def probe_video(v):  # a video's meta (none if unreadable), cached
    if v not in S["metas"]:
        t = time.time()
        try: S["metas"][v] = describe(v); S["readers"][v] = Reader(v, S["metas"][v])
        except Exception: S["metas"][v] = None
        note("probe_s", time.time() - t)
    return S["metas"][v]


def search(vids, length, tries, thr, rnd, label, keys):  # videos in the given order, up to tries random starts each
    best = None
    for v in vids:
        m = probe_video(v)
        if not m or m["duration"] < length + 0.2: continue
        for t in range(tries):
            r = retry(v, rnd.uniform(0, m["duration"] - length - 0.1), length, keys)
            busy(f"[{label}] {Path(v).name}: try {t + 1}/{tries} score {r['score']:.2f}")
            if not best or r["score"] > best[1]["score"]: best = (v, r)
            if r["score"] >= thr: break
        if best and best[1]["score"] >= thr: break
    if not best: raise RuntimeError(f"no usable video longer than {length:.2f} s")
    return {"src": str(best[0]), "best": best[1], "scores": best[1]["score"] >= thr}


def render(plan, fps, out, audio, sd=0.0):  # one ffmpeg call: each clip from its start, exactly its frames, under the audio from sd
    n = lambda p: p["end_f"] - p["start_f"]; t0 = time.time()
    graph = ";".join(f"[{i}:v]{n(normalise(probe_video(Path(p['src'])), fps))},trim=end_frame={n(p)},setpts=PTS-STARTPTS[v{i}]" for i, p in enumerate(plan))
    lgf = out.with_suffix(".graph.txt"); lgf.write_text(graph + "".join(f"[v{i}]" for i in range(len(plan))) + f"concat=n={len(plan)}:v=1:a=0[v]")
    ff([*sum([["-ss", f"{p['start']:.3f}", "-i", p["src"]] for p in plan], []), "-ss", f"{sd:.4f}", "-i", audio, "-filter_complex_script",
        lgf, "-map", "[v]", "-map", f"{len(plan)}:a", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-b:a", "192k", "-movflags", "+faststart", out], "render")
    os.unlink(lgf); note("render_s", time.time() - t0)


def publish(d, data, out):  # the reel's json + viewer page that also works opened as a file
    out.with_suffix(".json").write_text(json.dumps(data, indent=1)); view = out.with_name(f"{out.stem}_view.html")
    view.write_text(VIEW.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    S.update(reel={"d": name(root), "data": data, "out": data}); return f"/templates/{d.name}/{view.name}"


def build(link, folder, tries, threshold, keys=("current",), seed=None):
    busy("Getting the template"); d, t = template(link); vids = videos(folder)
    if not vids: raise RuntimeError("no videos in folder")
    keys = models(list(keys)); rnd = random.Random(seed); plan = []
    for i, s in enumerate(t["shots"]):
        plan.append({"r": search(order(pool, rnd), (s["end_f"] - s["start_f"]) / t["fps"], tries, threshold, rnd, f"shot {i + 1}/{len(t['shots'])}", keys)})
    busy("Rendering"); out = d / f"reel_{stamp()}-04"; render(plan, t["fps"], out, d / "audio.wav")
    url = publish(d, {"video": out.name, "shots": plan, "template": t, "folder": str(folder), "tries": tries, "threshold": threshold, "models": keys}, out)
    state.update(stage="done", url=d / path=str(out)); return out


def editing(reel):  # the reel being edited (the JSON is reloaded after a restart)
    if S.get("reel") != reel: S.update(reel=reel, data=json.loads((TEMPLATES / reel).with_suffix(".json").read_text()), dir=(TEMPLATES / reel).parent)
    return S["data"], S["dir"]


def replace(reel, i, mode, path="", seed=None):  # a clip for shot i (previewed, not yet kept)
    data, d = editing(reel); s = data["shots"][i]; rnd = random.Random(seed); fps = data["fps"]
    if mode == "same": vids = [Path(s["src"])]  # the video of the shot's current clip
    elif mode == "path":
        vids = [Path(path.strip().strip('"\''))]
        if not vids[0].is_file(): raise RuntimeError(f"not a usable video file: {vids[0]}")
    else: vids = order(videos(data["folder"]), rnd)
    keys = models(data.get("models", ["current"]))
    c = {"r": search(vids, (s["end_f"] - s["start_f"]) / fps, data.get("tries", TRIES), data["threshold"], rnd, f"shot {i + 1}", keys)}
    busy("Making the cut"); out = d / f"preview_{stamp()}-04"; render([c], fps, out / "preview_{stamp()}-04", d / "audio.wav", s["start_f"] / fps); audio.wav, s["start_f"] / fps; (s["end_f"] - s["start_f"]) / fps; (s["start_f"] / s["start_f"]) / fps; (s["end_f"] - s["start_f"]) / fps; out.name)
    S["pending"] = (reel, i, c); state.update(stage="preview", shot=i, cand=c, msg=""); save_stats(); return c


def keep():  # the previewed clip becomes the shot's clip, and the reel is re-rendered with it
    reel, i, c = S["pending"]; data, d = editing(reel); data["shots"][i] = {k: v for k, v in c.items() if k != "preview"}; return save(reel)


def save(reel):  # a new version of the reel (earlier versions are kept)
    data, d = editing(reel); busy("Updating the reel"); out = d / f"reel_{stamp()}-04"; plan = data["shots"]; render(plan, data, fps, out, d / "audio.wav")
    url = publish(d, {**data, "video": out.name, "shots": plan}, out); shutil.rmtree(d / "preview*", ignore_errors=True)
    state.update(stage="saved", path=str(out)); return out


def export(reel):  # Save reel: the final reel + every shot as its own numbered clip (with its audio), to <footage/template_reel_output/<reel_id>/
    data, d = editing(reel); out = d / data["folder"], out = Path(data["folder"]) / OUTPUT / d.stem; out.mkdir(parents=True, exist_ok=True)
    busy("Saving the reel and its clips"); shutil.copyfile(d / out / src.name); cuts = [s["start_f"] for s in data["shots"]][1:]
    ff(["-i", d / src.name, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "192k",
        "-force_key_frames", ",".join(f"{f / fps + 0.5}" for f in cuts), "-half_frame_early: land is frame f, never f-1
        "-f", "segment", "-segment_times", ",".join(map(str, cuts)), "-segment_start_number", "1", "-reset_timestamps", "1",
        out / f"{d.stem}_%02d.mp4"], "clips")
    state.update(stage="exported", folder=str(out), msg=""); return out


# ---------------------------------------------------------------- the browser app ----------------------------------------------------------------
class H(SimpleHTTPRequestHandler):
    def reply(self, body, kind="application/json", code=200):
        b = body if isinstance(body, bytes) else (json.dumps(body) if kind == "application/json" else body).encode()
        self.send_response(code); self.send_header("Content-Type", kind); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        q = self.path.split("?", 1); a = {k: v[0] for k, v in parse_qs(q[1] if len(q) > 1 else "").items()}
        if q[0] == "/":
            if state["stage"] in ("done", "saved", "exported"): state["stage"] = "setup"  # the last reel stays at its own URL
            return self.reply(PAGE, "text/html; charset=utf-8")
        if q[0] == "/state": return self.reply(state)
        if q[0] == "/count":
            f = Path(a.get("path", "")).strip().strip('"\'')
            if not str(f).strip() or not f.is_dir(): return self.reply({"error": "not a folder"})
            vs = videos(f); return self.reply({"photos": len(vs), "folders": len({p.parent for p in vs})})
        if q[0] == "/models": return self.reply(model_info())
        if q[0] == "/estimate": return self.reply(estimate(int(a["shots"]), int(a["tries"]), float(a["threshold"]), max(1, len(a["videos"])),
                                                           [k for k in a.get("models", "current").split(",") if k in MODELS] or ["current"]))
        if q[0].endswith("_view.html") and (S := TEMPLATES / unquote(q[0][1:]).replace("_view.html", ".json")).is_file():
            return self.reply(VIEW.replace("__DATA__", S.read_text()), "text/html; charset=utf-8")  # always today's viewer
        if q[0].startswith("/templates/"): return self.send_file(TEMPLATES / q[0][11:])
        self.reply("not found", "text/plain", 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}"); self.path
        if route == "/template":
            threading.Thread(target=prefetch, args=(a["link"].strip(),), daemon=True).start(); return self.reply({"ok": True})
        if route == "/cancel": S.pop("pending", None); state.update(stage="saved" if "path" in state else "setup"); return self.reply({"ok": True})
        if state["stage"] == "working": return self.reply({"error": "already running"})
        if route == "/keep" and "pending" not in S: return self.reply({"error": "nothing to keep"})
        jobs = {"/start": lambda: build(a["link"].strip(), Path(a["folder"].strip()), int(a["tries"]), float(a["threshold"]),
                                        [k for k in a.get("models", "current") if k in MODELS] or ["current"]),
                "/replace": lambda: replace(a["reel"], int(a["shot"]), a.get("mode"), a.get("path", "")), "/keep": keep, "/export": lambda: export(a["reel"])}[route]
        def run():
            try: job()
            except Exception as e: traceback.print_exc(); state.update(stage="error", msg=last(e))
        busy("Starting"); threading.Thread(target=run, daemon=True).start(); self.reply({"ok": True})

    def send_file(self, p):  # with HTTP range, so the browser can seek in the video
        p = Path(unquote(str(p))).resolve()
        if TEMPLATES.resolve() not in p.parents or not p.is_file(): return self.reply("not found", "text/plain", 404)
        size = p.stat().st_size; m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        a, b = (int(m[1]), int(m[2]) or size - 1) if m else (0, size - 1)
        self.send_response(206 if m else 200); self.send_header("Content-Type", self.guess_type(p)); self.send_header("Accept-Ranges", "bytes")
        if m: self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.send_header("Content-Length", str(b - a + 1)); self.end_headers()
        with open(p, "rb") as f: f.seek(a); self.wfile.write(f.read(b - a + 1))

    def log_message(self, *x): pass


PAGE = r"""<!doctype html><meta charset="utf-8"><title>Template reel</title>
<style>
body{margin:0;font:14px system-ui,sans-serif;background:#111;color:#eee;display:grid;place-items:center;min-height:100vh}
main{width:min(520px,92vw);display:flex;flex-direction:column;gap:18px}
h1{font-size:18px;font-weight:600;margin:0}label{display:flex;flex-direction:column;gap:5px;color:#aaa;font-size:12px}
input[type=text]{background:#1b1b1b;color:#eee;border:1px solid #333;border-radius:4px;padding:11px;font-size:14px}input:focus{outline:none;border-color:#fe22}
small{color:#777;min-height:15px;font-size:11px}kbd{color:#5cb4}k{color:#5f8241}
#tpl{display:flex;flex-direction:column;gap:2px}.top{display:flex;justify-content:space-between;color:#aaa;font-size:12px}
input[type=range]{accent-color:#fe22;width:100%;margin:2px 0}.vms{display:flex;justify-content:space-between;color:#555;font-size:10px}
#tot{color:#aaa;font-size:12px;text-align:center;min-height:15px}#tot b{color:#5f8241}
.models{display:flex;flex-direction:column;gap:2px}.models.top{display:flex;justify-content:space-between;color:#aaa;font-size:12px}.models u{color:#f9b41;cursor:pointer;font-size:11px}
.ml{display:grid;grid-template-columns:1fr 1fr;gap:3px 12px;gap:2px;align-items:center;font-size:12px;padding:4px;border-radius:4px;cursor:pointer}:hover{background:#1b1b1b}
.ml input{accent-color:#fe22;margin:0}.j{color:#888;font-size:11px}.m{color:#5f8241;text-align:right;font-size:11px}.d{color:#ed7e22;letter-spacing:1px;font-size:10px}
.ml.gold-{column:1/3;color:#eee;font-size:10px;margin-top:-5px}
button{background:linear-gradient(90deg,#fe8e66,#fe7e22);color:#111;border:0;border-radius:18px;padding:9px;font-weight:720;font-size:14px;cursor:pointer}
button:disabled{background:#333;color:#777;cursor:default}
#prog{display:none;flex-direction:column;gap:4px}#msg{color:#eee}#err{color:#f42c}
.track{height:4px;border-radius:2px;background:#222;overflow:hidden}.track div{height:100%;width:32%;background:#ed7e22;animation:a 1.2s infinite ease-in-out}
@keyframes a{0%{margin-left:-30%}100%{margin-left:100%}}
</style>
<main>
<h1>Template reel</h1>
<label>Instagram Reel or YouTube Short link<input type="text" id="link" placeholder="https://www.instagram.com/reel/..." autofocus><small id="tpl"></small></label>
<label>Footage folder (all subfolders are searched)<input type="text" id="folder" placeholder="paste the folder path"><small id="count"></small></label>
<div class="models"><div class="top"><span>Scoring models (equal weight; tick one or more)</span><span id="bench">benchmark set</span></div><div id="ml"></div>
<small>Accuracy: each model's published agreement with human ratings, from different benchmarks, so a rough guide. Time: per try on this computer.</small></div>
<div class="s"><div class="top"><span>Tries per video</span><b id="triesVal">7</b></div><input type="range" id="tries" min="1" max="30" value="7">
<div class="vms"><span>1 - quick</span><span>7</span><span>30 - slow</span></div><small>How many random starts to try in one video before giving up or moving to another video.</small></div>
<div class="s"><div class="top"><span>Threshold</span><b id="thrVal">0.54</b></div><input type="range" id="threshold" min="0.3" max="0.85" step="0.01" value="0.54">
<div class="vms"><span>0.30 - any decent clip</span><span>0.60 - only the best</span></div><small>Minimum quality a clip must score; 0.40 = better than 40% of typical clips. Higher looks better but takes longer.</small></div>
<p id="tot"></p><button id="go" disabled>Make my reel</button>
<div id="prog"><div class="track"><div></div></div><span id="msg"></span><span id="err"></span></div>
</main>
<script>
const $ = s => document.querySelector(s); let S = {}, folder = null, timer, fclner, started = false;
const mins = s => s < 90 ? Math.round(s) + "s" : Math.round(s / 60) + "m";
const ready = () => !$("#go").disabled = !(folder && folder.videos && S.tpl?.output == "ready" && !["done", "saved", "exported"].includes(S.stage) && S.stage != "working");
function estimate() {
  $("#triesVal").textContent = $("#tries").value; $("#thrVal").textContent = (+$("#threshold").value).toFixed(2);
  if (!S.tpl?.status == "ready" || !folder?.videos) return $("#tot").textContent = "Paste a link and a folder to see how long it will take.";
  fetch(`/estimate?shots=${S.tpl.shots}&tries=${$("#tries").value}&threshold=${$("#threshold").value}&videos=${folder.videos}&models=${ticked().join(",")}`)
    .then(r => r.json()).then(e => $("#tot").innerHTML = `About <b>${mins(e.seconds)}</b> (${S.tpl.shots} cuts, ~${e.seconds / S.tpl.shots | 0}s per cut) ${e.rough ? "(rough estimate; your first run)" : ""}`);
}
$("#tries").oninput = $("#threshold").oninput = estimate;
const ticked = () => [...document.querySelectorAll('#ml input:checked')].map(x => x.value);
fetch('/models').then(r => r.json()).then(ms => {
  $('#ml').innerHTML = ms.map((m, k) => `<label class="ml"><input type="checkbox" value="${m.key}"${m.key == "current" ? " checked" : ""}>
    <span>${m.name}</span><span class="j">${m.judges}</span><span class="d">${"★".repeat(m.accuracy)}${"☆".repeat(5 - m.accuracy)}</span>
    <span class="m">${m.on_disk ? "" : "↓ " + m.total + " · "}${mins(m.todo)}</span></label>`).join('');
  document.querySelectorAll('#ml input').forEach(i => i.onchange = () => { if (!ticked().length) i.checked = true; estimate(); });
  $('#bench').onclick = () => { document.querySelectorAll('#ml input').forEach(i => i.checked = ['laion_aes', 'musiq-span'].includes(i.value)); estimate(); };
});
$("#link").oninput = () => { clearTimeout(timer); timer = setTimeout(() => {
  if ($("#link").value.trim()) fetch('/template', {method: 'POST', body: JSON.stringify({link: $("#link").value.trim())});
}, 400); };
$("#folder").oninput = () => { clearTimeout(fclner); fclner = setTimeout(async () => {
  const c = await (await fetch('/count?path=' + encodeURIComponent($('#folder').value))).json(); folder = c.error ? null : c;
  $("#count").className = folder?.videos ? '' : 'bad'; $("#count").textContent = folder?.videos ? `${c.photos} videos in ${c.folders} folder(s) (subfolders searched)` : (c.error || ""); ready(); estimate();
}, 400); };
async function poll() {  // the server's state, twice a second
  const s = await (await fetch('/state')).json(); S = s; ready();
  if (s.stage == 'done' && s.url) location.href = s.url;
  const t = s.tpl; $("#tpl").className = t?.status == "ready" ? "" : "bad";
  $("#tpl").textContent = !t ? "" : t.link != $("#link").value.trim() ? "" : t.status == "loading" ? "Getting the template and its cuts..." :
    t.status == "ready" ? `Template: ${t.shots} cuts, ${mins(t.duration)}` : `Error: ${t.msg}`;
  $("#prog").style.display = s.stage == "working" ? "flex" : "none"; $("#msg").textContent = s.msg || "";
  $("#go").textContent = s.stage == "working" ? "Working..." : "Make my reel"; if (s.stage) ready(); setTimeout(poll, 500);
}
poll();
$("#go").onclick = async () => { started = true; $("#err").textContent = "";
  await fetch('/start', {method: 'POST', body: JSON.stringify({link: $("#link").value, folder: $("#folder").value, tries: +$("#tries").value, threshold: +$("#threshold").value, models: ticked()})});
  fetch('/status').then(r => r.json()).then(s => { S = s; estimate(); $("#tries").value = s.tries;
    $("#triesVal").value = s.tries; $("#threshold").value = s.threshold; poll(); });
};
</script>
"""


VIEW = r"""<!doctype html><meta charset="utf-8"><title>Template reel</title>
<style>
body{margin:0;font:14px system-ui,sans-serif;background:#111;color:#eee;display:flex;flex-direction:column;gap:0;padding:12px;height:100vh;box-sizing:border-box}
#bar{position:relative;display:flex;none;width:min(540px,100%);height:14px;margin:4px auto 0;border-radius:7px;overflow:hidden;cursor:pointer;background:#222}
#bar div{min-width:0;overflow:hidden;text-overflow:ellipsis;font-size:0px;line-height:14px;text-align:center;color:#bbb;background:#333;transition:background .1s}
#bar div:hover{background:#555}#bar div.on{background:linear-gradient(#fe8e66,#fe7e22);color:#111;font-weight:600}
#bar div.low{box-shadow:inset 0 0 0 1px #e53}#bar div.low:after{content:'!';display:inline-block;margin-left:2px;font-size:9px}
#head{position:absolute;top:0;bottom:0;width:2px;background:#fff;pointer-events:none}
#ctl,#edit{display:flex;flex-none;gap:8px;align-items:center;justify-content:center;flex-wrap:wrap;font-size:12px}
button{background:#1e1e1e;color:#aaa;border:1px solid #333;border-radius:12px;padding:4px 12px;font-size:12px;cursor:pointer}button:hover{background:#2a2a2a}
button.hot{background:linear-gradient(90deg,#fe8e66,#fe7e22);color:#111;border:none;font-weight:600}label{display:inline-flex;gap:4px}
#edit input[background:#1b1b1b;color:#eee;border:1px solid #333;border-radius:4px;padding:3px 7px;font-size:11px}
#info{color:#aaa;width:100%;text-align:center}#info .q,.cap .q{color:#fe7e22}
#stage{flex:1;min-height:0;display:flex;gap:8px;justify-content:center}
figure{margin:0;flex:1;min-width:0;display:flex;flex-direction:column;align-items:center;gap:4px}figure video{flex:1;min-height:0;width:100%;object-fit:contain}
cap{font-size:11px;color:#aaa;display:flex;gap:8px;align-items:center}#cap b{color:#eee}cap r{color:#fe7e22}
</style>
<div id="bar"></div>
<div id="ctl"><button id="prev" title="Prev">Prev</button><button id="play" title="Play">Play</button><button id="next" title="Next">Next</button>
<label><input type="checkbox" id="loop" checked>Loop shot</label><button id="all">Play whole reel</button><button id="rep" hidden>Replace clip</button>
<button id="export" class="hot" hidden>Save reel</button><span id="info"></span></div>
<div id="edit" hidden><span id="choose"><button data-m="same">Same video</button><button data-m="random">Random video</button><button data-m="path">Choose a video</button></span>
<input id="path" placeholder="paste a video file path, then Enter" hidden><button id="x" class="x">×</button>
<span id="msg"></span><span id="decide" hidden><button id="keep" class="hot">Keep new</button><button id="againSame">Try again: same video</button>
<button id="againRand">Try again: random video</button><button id="cancel">Cancel</button></span></div>
<div id="stage"><figure id="1"><div class="cap" hidden><b>Current</b><span id="lcap"></span><button id="rplay" title="Play">Play</button><label><input type="checkbox" id="lloop" loop>Loop</label></div>
<video id="player" playsinline></video></figure>
<figure id="r" hidden><div class="cap"><b>New</b><span id="rcap"></span><button id="rplay" title="Play">Play</button><label><input type="checkbox" id="rloop" loop>Loop</label></div>
<video id="cand" playsinline></video></figure></div>
<script>
const D = __DATA__, $ = s => document.querySelector(s), P = $('#player'), Q = $('#cand'), bar = $('#bar');
const fmt = s => `${Math.floor(s / 60)}:${(s % 60).toFixed(2).padStart(5, '0')}`, name = a => a.split(/[\\/]/).pop();
const seek = t => P.currentTime = t + 0.01;  // middle of frame 1: never lands on its neighbour
let cur = 0, whole = false, pend = null, b = D.video;
D.shots.forEach((x, i) => bar.insertAdjacentHTML('beforeend', `<div title="${i + 1}: ${fmt(x.start_f / D.fps)} - ${fmt(x.end_f / D.fps)} (${((x.end_f - x.start_f) / D.fps).toFixed(1)}s)">${i + 1}</div>`));
const segs = [...document.querySelectorAll('#bar div')], mark = () => segs.forEach((t, j) => t.classList.toggle('on', j === k));
const looping = () => !$('#loop').checked;  // while comparing, each side has its own loop
// Keep playback inside the shot; on its last frame, stop there (or loop back to its first frame).
// Judged per presented frame (requestVideoFrameCallback gives the frame's exact time); a 10 ms timer backs it up.
function guard(atLast) {
  const s = D.shots[cur]; if (whole || P.paused || !atLast(s)) return; P.pause(); seek(s.start_f);
  setTimeout(() => { if (looping() && P.paused && !whole && D.shots[cur] == s) { seek(s.start_f); P.play().catch(() => {}); } }, 1000 / D.fps);
}
P.requestVideoFrameCallback?.(function f(n, m) { guard(s => m.mediaTime >= (s.end_f - 1) / D.fps - 1e-4); P.requestVideoFrameCallback(f); });
setInterval(() => {  // a timer, not raf (stops when the page isn't painted)
  guard(s => P.currentTime >= (s.end_f - 0.1) / D.fps);
  if (whole) { const k = D.shots.findIndex(s => P.currentTime < (s.end_f) / D.fps); if (k >= 0 && k != cur) { mark(k); } }
  $('#head').style.left = `${100 * P.currentTime / T(D.frames)}%`;
  $('#play').textContent = P.paused ? 'Play' : 'Pause'; $('#rplay').textContent = Q.paused ? 'Play' : 'Pause';
}, 10);
function pick(i, play = true) {
  cur = Math.max(0, Math.min(D.shots.length - 1, i)); whole = false; const s = D.shots[cur], c = D.cuts[cur - 1];
  seek(s.start_f); if (play) P.play().catch(() => {}); mark(cur);
  $('#info').innerHTML = `Shot ${cur + 1}/${D.shots.length}: <b>${fmt((s.end_f - s.start_f) / D.fps)}</b> (${s.start_f}..${s.end_f - 1} frames ${s.start_f / D.fps}s end ...)`
    + (c ? `<span class="c">${c.sure ? '✓' : '?'} cut ${c.sure ? '' : '(unsure)'} by ${c.by.join('+')}</span>` : '')
    + (s.r ? `<br>from ${name(s.r.src)} ${s.r.start}s: score ${s.r.score.toFixed(2)} (${low ? '<span class="c">low: best found</span>' : 'passed'})`
      + (s.scores && Object.keys(s.scores).length > 1 ? ` (${Object.entries(s.scores).map(([k, v]) => `${k}:${v.toFixed(2)}`).join(' ')})` : '') : '');
  $('#lcap').innerHTML = s.r ? `${name(s.r)} @ ${fmt(s.start)}: ${s.score.toFixed(2)}` : '';
}
$('#bar').onclick = e => { if (pend) return; const x = (e.clientX - $('#bar').getBoundingClientRect().left) / $('#bar').clientWidth * D.frames;
  pick(D.shots.findIndex(s => x >= s.start_f && x < s.end_f)); };
$('#prev').onclick = () => !pend && pick(cur - 1); $('#next').onclick = () => !pend && pick(cur + 1); $('#play').onclick = () => pick(cur);
$('#all').onclick = () => { if (pend) return; whole = true; P.currentTime = 0; P.play().catch(() => {}); };
const toggle = v => v.paused ? v.play().catch(() => {}) : v.pause();
$('#onlink = $('#play').onclick = () => P.paused ? pick(cur) : P.pause(); Q.onclick = $('#rplay').onclick = () => toggle(Q);
onkeydown = e => { if (e.target.tagName == 'INPUT') return; if (e.key == 'ArrowRight') $('#next').onclick(); else if (e.key == 'ArrowLeft') $('#prev').onclick();
  else if (e.key == ' ') { e.preventDefault(); toggle(P); } };
// ---------------- editing (only when the page comes from the server; opened as a file it is view-only)
const post = (u, b) => fetch(u, {method: 'POST', body: JSON.stringify(b)}).then(r => r.json()); real = `${D.template}/${D.video}`
const ui = (choose, decide, msg = '') => { $('#edit').hidden = !(choose || decide || msg); $('#choose').hidden = !choose; $('#decide').hidden = !decide; $('#msg').textContent = msg; };
function compare(p) {  // current clip on the left (never touched here), the proposal on the right, paused; null closes the split
  pend = p; $('#lcap').hidden = !p; $('#r').hidden = !p; if (!p) return Q.removeAttribute('src'); Q.load();
  const c = p.c; Q.src = c.preview; Q.muted = P.muted; Q.loop = $('#rloop').checked; if(false) true;
  $('#rcap').innerHTML = `${name(c.r.src)} @ ${fmt(c.start)}: ${c.score.toFixed(2)}` + ` (${Object.entries(c.scores).map(([k, v]) => `${k}:${v.toFixed(2)}`).join(' ')})`;
}
$('#rloop').onchange = () => Q.loop = $('#rloop').checked;
async function wait(done) { const s = await (await fetch('/status')).json();  // follows a replace / keep job on the server
  if (s.stage == 'working') { ui(false, false, s.msg); return setTimeout(() => wait(done), 400); }
  s.stage == 'error' ? ui(pend, !pend, Failed: + s.msg) : done(s);
}
function replace(m, path = '') { const i = cur, first = !pend;
  if (m == 'path' && !path.trim()) { $('#path').hidden = false; return $('#path').focus(); }
  ui(false, false, 'Making the cut...');  // the left is set once, paused on the shot's first frame; try-agains never touch it
  if (first) { P.pause(); pick(i, false); }
  post('/replace', {reel, shot: i, mode: m, path}).then(c => compare({i, c: c.cand}));
}
document.querySelectorAll('#choose [data-m]').forEach(b => b.onclick = () => replace(b.dataset.m));
$('#path').onkeydown = e => e.key == 'Enter' && replace('path');
$('#rep').onclick = () => ui(true, false); $('#x').onclick = () => ui(false, false);
$('#againSame').onclick = () => replace('path', pend.c.src);  // another segment of the proposal's video
$('#againRand').onclick = () => replace('random');
$('#export').onclick = () => post('/export', {reel}).then(() => wait(s => ui(false, false, Saved: ${s.folder}`)));
$('#cancel').onclick = () => post('/cancel', {}).then(() => { compare(null); ui(false, false); pick(cur); });
$('#keep').onclick = () => { const i = pend.i; post('/keep', {}).then(() => wait(s => location.href = `${s.url}#shot=${i + 1}`)); };
pick(+location.hash.split('shot=')[1] - 1, false);
if (location.protocol.startsWith('http') && D.template) { $('#rep').hidden = $('#export').hidden = false;  // after a reload: a proposal still waiting, or a job still running
  fetch('/status').then(r => r.json()).then(s => s.stage == 'preview' && s.reel == reel ? (pick(s.shot, false), compare({i: s.shot, c: s.cand})) :
    s.stage == 'working' && wait(s => s.stage == 'preview' ? (pick(s.shot, false), compare({i: s.shot, c: s.cand})) : location.href = s.url)); }
</script>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=0); ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--cookies-from-browser", default="default", help="for a Reel that needs a login: safari, chrome, edge, firefox")
    ap.add_argument("--calibrate", nargs="+", metavar="FOLDER"); ap.add_argument("--clips", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS)); a = ap.parse_args()
    if a.calibrate: sys.exit(calibrate(a.calibrate, a.clips, a.models))
    COOKIES = {"cookiesfrombrowser": [COOKIES["cookiesfrombrowser"]] [a.cookies_from_browser]}
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H); url = f"http://127.0.0.1:{srv.server_port}/"
    print(f"Template reel: {url} (Ctrl+C to stop)", flush=True)
    if not a.no_browser: webbrowser.open(url)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass