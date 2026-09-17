#!/usr/bin/env python3
"""
Smart Anime/Game/Real Upscaler v11.0 — SERVER STABLE PIPELINE
  - ✅ Bounded queues & strictly ordered futures (RAM safe)
  - ✅ FFmpeg deadlock prevention (stderr routed to temp files)
  - ✅ True cancellation safety (threads & subprocesses cleanly die)
  - ✅ Smoothed ETA (Exponential Moving Average)
  - ✅ Milestone-based status updates (anti-flood)
  - ✅ GitHub Actions optimized (No global torch thread shifting during jobs)
  - ✅ MANAGER MODE: File system alag (manager_work/) — workers se separate
"""
import asyncio, concurrent.futures, gc, json, logging, math, os, queue, random, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import cv2
import numpy as np
import requests
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))
import torch
from pyrogram import Client, filters, idle
from pyrogram.errors import FloodWait
from pyrogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
try:
    from realesrgan.archs.rrdbnet_arch import RRDBNet
except ModuleNotFoundError:
    from basicsr.archs.rrdbnet_arch import RRDBNet

try:
    from archive import ChannelArchive
except Exception:
    ChannelArchive = None

# ================= CONFIG =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("anime-upscaler")

API_ID = int(os.getenv("API_ID", "0") or 0)
API_HASH = (os.getenv("API_HASH", "") or "").strip()
BOT_TOKEN = (os.getenv("BOT_TOKEN", "") or "").strip()
_raw_owner = (os.getenv("OWNER_CHAT_ID", "") or "").strip().strip("'").strip('"')
OWNER_CHAT_ID = _raw_owner
try:
    OWNER_CHAT_ID_INT = int(_raw_owner)
except ValueError:
    OWNER_CHAT_ID_INT = 0

def is_owner(chat_id) -> bool:
    return chat_id == OWNER_CHAT_ID_INT or str(chat_id) == OWNER_CHAT_ID

def _owner_cid() -> int:
    if OWNER_CHAT_ID_INT: return OWNER_CHAT_ID_INT
    if OWNER_CHAT_ID.lstrip("-").isdigit(): return int(OWNER_CHAT_ID)
    return 0

if not API_ID or not API_HASH or not BOT_TOKEN or not OWNER_CHAT_ID:
    raise RuntimeError("Missing GitHub Secrets")

MODEL_DIR = Path("weights")
# ===== MANAGER FILE SYSTEM (workers se alag) =====
WORK_DIR = Path("manager_work")
OUTPUT_DIR = Path("manager_output")
WORK_DIR.mkdir(exist_ok=True); OUTPUT_DIR.mkdir(exist_ok=True)

MAX_FRAMES = 3600
MAX_GIF_FRAMES = 240
MAX_OUT_PIXELS = 3840 * 2160
MAX_SEND_MB = 1900
MAX_JOB_SEC = int(os.getenv("MAX_JOB_MIN", "330")) * 60
DISTRIBUTED_START_WINDOW_SEC = 420
GH_PAT = (os.getenv("GH_PAT", "") or "").strip()
GH_REPO = (os.getenv("GH_REPO", os.getenv("GITHUB_REPOSITORY", "")) or "").strip()
DISTRIBUTED_WORKFLOW = os.getenv("DISTRIBUTED_WORKFLOW", "upscale.yml")
DISTRIBUTED_MAX_WORKERS = max(1, min(20, int(os.getenv("MAX_WORKERS", "20"))))
GIF_MIN_SEC = 2.0

# ===== Status message: sirf milestone par edit =====
STATUS_POLL_EVERY       = 3.0
STATUS_MILESTONES       = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100]
STATUS_STUCK_SEC        = 60.0

# ===== Panel: bahut slow =====
PANEL_EVERY             = 15.0
PANEL_MIN_EDIT_INTERVAL = 12.0

CPU_THREADS = os.cpu_count() or 4
POOL = ThreadPoolExecutor(max_workers=max(4, CPU_THREADS))
try:
    torch.set_num_threads(max(1, CPU_THREADS))
except Exception:
    pass

# ================= MODELS =================
MODELS = {
    "anime_video": {"file": "realesr-animevideov3.pth",       "arch": "srvgg", "label": "🎌 Anime Video", "best_for": "anime video (fast)",  "video_ok": True},
    "anime_image": {"file": "RealESRGAN_x4plus_anime_6B.pth", "arch": "rrdb",  "label": "🎌 Anime Image", "best_for": "anime still (crisp)", "video_ok": False},
    "game":        {"file": "realesr-general-x4v3.pth",       "arch": "srvgg", "label": "🎮 Game (Fast)", "best_for": "gameplay (FF/PUBG)",  "video_ok": True},
    "real":        {"file": "RealESRGAN_x4plus.pth",          "arch": "rrdb",  "label": "📷 Real Photo",  "best_for": "real-world photos",   "video_ok": False},
}
LEGACY_MODEL_MAP = {"anime": "anime_video", "gamehq": "real"}
VIDEO_FALLBACK = {"anime_image": "anime_video", "real": "game"}

def normalize_model_key(key) -> str:
    if isinstance(key, str) and key in MODELS: return key
    if isinstance(key, str) and key in LEGACY_MODEL_MAP: return LEGACY_MODEL_MAP[key]
    return "anime_video"

def normalize_colorize(val) -> str:
    if val is True:  return "fast"
    if val is False: return "off"
    if isinstance(val, str) and val in ("off", "fast", "high"): return val
    return "off"

PRESETS = {"fast": {"crf": "23", "preset": "veryfast"},
           "balanced": {"crf": "19", "preset": "veryfast"},
           "best": {"crf": "16", "preset": "slow"}}

DDCOLOR_MODELS = {
    "fast": {"file": "ddcolor_tiny.pth", "label": "🎨 Fast"},
    "high": {"file": "ddcolor_high.pth", "label": "💎 High"},
}

def _build_core_profiles():
    p = [("auto", "🤖 Auto", 0, 0)]
    for n in (1, 2, 3, 4, 6, 8, 12, 16):
        if n <= CPU_THREADS: p.append((f"solo{n}", f"🐢 Solo {n}T", 1, n))
    for w, t, lbl in [(2, 2, "⚡ Duo 2×2"), (2, 3, "⚡ Duo 2×3"),
                      (3, 1, "🔥 Trio 3×1"), (4, 1, "🔥 Quad 4×1"),
                      (6, 1, "🔥 Hexa 6×1")]:
        if w * t <= CPU_THREADS: p.append((f"m{w}x{t}", lbl, w, t))
    return p

CORE_PROFILES = _build_core_profiles()
CORE_MAP = {p[0]: p for p in CORE_PROFILES}

settings = {"scale": 2.0, "preset": "balanced", "audio": "keep",
            "model": "anime_video", "core": "auto", "colorize_mode": "off"}
job_state = {"active": False}
current_job: Optional[Dict[str, Any]] = None
cancel_event: Optional[threading.Event] = None
JOB_QUEUE: List[Dict[str, Any]] = []
_queue_task: Optional[asyncio.Task] = None

# ================= SYSTEM =================
def mem_avail_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception: pass
    return 8.0

def load1() -> float:
    try: return os.getloadavg()[0]
    except Exception: return 0.0

def fmt_time(s: float) -> str:
    if s < 0 or math.isnan(s): return "0s"
    s = max(0, int(s)); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h}h {m}m {sec}s" if h else (f"{m}m {sec}s" if m else f"{sec}s")

def fmt_scale(s: float) -> str: return f"{s:g}"

def bar(pct: float, n: int = 12) -> str:
    if math.isnan(pct): pct = 0
    f = int(n * min(100, max(0, pct)) / 100)
    return "▰" * f + "▱" * (n - f)

PW = 34
def _pad(s: str) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) >= PW else s + " " * (PW - len(s))

# ================= EMOTE =================
FACE_TICK = 0.85
FACES = {
    "idle":     ["(˘˘)… zZ", "(¬ᴗ¬) zZ", "(˘▽˘) ♪"],
    "work":     ["(っ⚙_)っ", "(っ⚙_⚙)っ✦", "(っ⚙_⚙)っ✧"],
    "think":    ["(◔_)…", "(◔‿◔)?", "(◕_◕)…"],
    "happy":    ["(ﾉ◕)ﾉ*:･ﾟ✧", "(◕‿)✧", "(＾▽＾)ﾉ★"],
    "error":    ["(×_×;)", "(╥_╥)…", "(⊙_)!"],
    "love":     ["(♥‿♥)", "(♡ω♡)", "(⁄⁄•⁄ω•⁄⁄)"],
    "start":    ["(ò_ó)⚡", "(◉)✧", "(ᐛ)و"],
    "upload":   ["(⇀↼)", "(_↼)️", "(⇀‿↼)"],
    "download": ["(⇂_⇂)📥", "(⇂_)", "(⇂_⇂)✦"],
    "wow":      ["(✧ω✧)", "(✧▽✧)", "(◍◍)✨"],
}
MOOD_ORDER = ["idle", "happy", "wow", "love", "think", "work", "start"]

class EmoteEngine:
    def __init__(self): self.state = "idle"; self.t0 = time.time()
    def set(self, s: str):
        if s != self.state: self.state = s; self.t0 = time.time()
    def face(self) -> str:
        fr = FACES.get(self.state, FACES["idle"])
        return fr[int((time.time() - self.t0) / FACE_TICK) % len(fr)]
    def one(self, s: str) -> str:
        return random.choice(FACES.get(s, FACES["idle"]))

EMO = EmoteEngine()

# ================= GOVERNOR =================
CONFIGS = [(1, 4), (2, 2), (2, 3), (3, 1), (4, 1)]

class Governor:
    def __init__(self, out_px: int, fixed: Optional[tuple] = None, max_workers: int = 4):
        self.fp = out_px * 512 / 1e9
        self.fixed = fixed
        self.max_workers = max_workers
        self.ema = {c: 0.0 for c in CONFIGS}
        self.cnt = {c: 0 for c in CONFIGS}
        if fixed is not None:
            w, t = fixed
            max_w = max(1, int(mem_avail_gb() * 0.6 / max(self.fp, 0.1)))
            if w > max_w: w = max_w
            start = (w, t)
        else:
            start = (2, 2) if self._ram_ok(2) else ((1, 2) if self._ram_ok(1) else (1, 1))
            for w in (min(4, self.max_workers), 3, 2):
                if w >= 1 and w <= CPU_THREADS and self._ram_ok(w):
                    start = (w, max(1, CPU_THREADS // w)); break
        self.current = start
        self.load_ema = load1(); self.ram_ema = mem_avail_gb()
        self.safe = False
        self.lock = threading.Lock()
        self.apply()
        log.info("🧠 Governor %s%s | fp %.2fGB/fr | RAM %.1fGB",
                 self.current, " [FIXED]" if fixed else "", self.fp, self.ram_ema)

    def _ram_ok(self, w): return w * self.fp <= max(1.0, mem_avail_gb() * 0.85)
    def apply(self): torch.set_num_threads(self.current[1])
    def thr(self, c): return c[0] / self.ema[c] if self.ema[c] else 0.0
    def gate(self, instances: int) -> int: return max(1, min(self.current[0], instances))

    def on_frame(self, cfg, dt: float, done: int):
        with self.lock:
            c = tuple(cfg)
            if c in self.ema:
                self.ema[c] = dt if self.cnt[c] == 0 else self.ema[c] * 0.7 + dt * 0.3
                self.cnt[c] += 1
            self.load_ema = self.load_ema * 0.8 + load1() * 0.2
            self.ram_ema = self.ram_ema * 0.8 + mem_avail_gb() * 0.2
            if done % 20 == 0:
                gc.collect()
            if self.ram_ema < 0.5 or self.load_ema > CPU_THREADS * 2.0:
                self.safe = True
            elif self.safe and self.ram_ema > 3.0 and self.load_ema < CPU_THREADS * 0.9:
                self.safe = False

    def status(self, spf: float) -> str:
        fx = " [FIXED]" if self.fixed else ""
        return (f"🤖 {self.current[0]}W×{self.current[1]}T{fx} | {spf:.2f}s/fr | "
                f"{self.thr(self.current):.2f} f/s | 🛡 {self.ram_ema:.1f}G | "
                f"load {self.load_ema:.1f}" + (" | SAFE" if self.safe else ""))

# ================= REAL-ESRGAN INSTANCE POOL =================
_ups_pool_queues: Dict[Any, queue.Queue] = {}
_ups_pool_meta: Dict[Any, Tuple[int, int]] = {}
_ups_pool_lock = threading.Lock()

def _detect_srvgg_num_conv(path: Path) -> int:
    try:
        sd = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(sd, dict):
            for k in ("params_ema", "params", "state_dict"):
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]; break
        max_idx = -1
        for k in sd.keys():
            if k.startswith("body.") and k.endswith(".weight"):
                parts = k.split(".")
                if len(parts) >= 3 and parts[1].isdigit():
                    max_idx = max(max_idx, int(parts[1]))
        if max_idx >= 2: return max(16, (max_idx - 2) // 2)
    except Exception as e:
        log.warning("num_conv detect fail: %s", e)
    return 16

def choose_tile(key: str, out_px: int) -> int:
    arch = MODELS[key]["arch"]
    ram = mem_avail_gb()
    if arch == "rrdb":
        if ram > 12: return 512
        if ram > 6: return 384
        return 256
    if ram > 6: return 0
    if ram > 3: return 640
    return 320

def _instance_count(key: str) -> int:
    arch = MODELS[key]["arch"]
    ram = mem_avail_gb()
    if arch == "rrdb":
        by_ram = max(1, int(ram / 4.5))
        return max(1, min(2, by_ram, CPU_THREADS))
    by_ram = max(1, int(ram / 2.5))
    return max(1, min(4, by_ram, CPU_THREADS))

def _create_ups_instance(key: str, tile: int) -> RealESRGANer:
    m = MODELS[key]; path = MODEL_DIR / m["file"]
    if not path.exists(): raise FileNotFoundError(f"Model missing: {path}")
    if m["arch"] == "rrdb":
        nb = 6 if "anime_6B" in m["file"] else 23
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=nb, num_grow_ch=32, scale=4)
        return RealESRGANer(scale=4, model_path=str(path), model=model,
                             tile=tile, tile_pad=16, pre_pad=0,
                             half=False, device=torch.device("cpu"))
    nconv = _detect_srvgg_num_conv(path)
    last_err = None
    for nc in [nconv, 32, 16]:
        try:
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                    num_conv=nc, upscale=4, act_type="prelu")
            return RealESRGANer(scale=4, model_path=str(path), model=model,
                                tile=tile, tile_pad=16, pre_pad=0,
                                half=False, device=torch.device("cpu"))
        except Exception as e:
            last_err = e
            log.warning("num_conv=%s load fail: %s", nc, str(e)[:140])
    raise last_err or RuntimeError("SRVGG load failed")

def init_ups_pool(key: str, tile: int, fixed: Optional[Tuple[int, int]] = None) -> queue.Queue:
    key = normalize_model_key(key)
    k = (key, tile)
    with _ups_pool_lock:
        if k in _ups_pool_queues: return _ups_pool_queues[k]
        if fixed is not None:
            workers = max(1, min(fixed[0], _instance_count(key)))
            threads = max(1, fixed[1])
        else:
            workers = _instance_count(key)
            threads = max(1, CPU_THREADS // workers)
        _ups_pool_meta[k] = (workers, threads)
        torch.set_num_threads(threads)
        q: queue.Queue = queue.Queue()
        for i in range(workers):
            log.info("🏗 Upsampler instance %d/%d (%s tile=%s)...", i + 1, workers, key, tile)
            q.put(_create_ups_instance(key, tile))
        _ups_pool_queues[k] = q
        log.info("✅ Pool ready: %d instances × %d threads (RAM %.1fGB free)",
                 workers, threads, mem_avail_gb())
        return q

def pool_instances(key: str, tile: int) -> int:
    return _ups_pool_meta.get((normalize_model_key(key), tile), (1, 1))[0]

# ================= DDCOLOR =================
_ddcolor_cache: Dict[str, Any] = {}
_ddcolor_lock = threading.Lock()

def _load_ddcolor(mode: str):
    if mode in _ddcolor_cache: return _ddcolor_cache[mode]
    with _ddcolor_lock:
        if mode in _ddcolor_cache: return _ddcolor_cache[mode]
        info = DDCOLOR_MODELS.get(mode)
        if info is None: return None
        path = MODEL_DIR / info["file"]
        if not path.exists():
            log.warning("🎨 DDColor %s missing: %s", mode, path); return None
        try:
            if "ddcolor-src" not in sys.path: sys.path.insert(0, "ddcolor-src")
            from ddcolor import DDColor
            try:
                from huggingface_hub import PyTorchModelHubMixin
                class DDColorHF(DDColor, PyTorchModelHubMixin):
                    def __init__(self, config=None, **kwargs):
                        if isinstance(config, dict): kwargs = {**config, **kwargs}
                        super().__init__(**kwargs)
            except Exception: DDColorHF = DDColor
            log.info("🎨 Loading DDColor %s ...", mode)
            model = DDColorHF.from_pretrained(str(path))
            model.eval()
            _ddcolor_cache[mode] = model
            log.info("✅ DDColor %s ready", mode)
            return model
        except Exception as e:
            log.error("DDColor %s load fail: %s", mode, e)
            _ddcolor_cache[mode] = None; return None

def colorize_frame(img: np.ndarray, mode: str, ref: int = 512) -> np.ndarray:
    if mode == "off" or mode not in DDCOLOR_MODELS: return img
    model = _load_ddcolor(mode)
    if model is None: return img
    try:
        import torchvision.transforms as T
        from PIL import Image as PILImage
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        ow, oh = pil.size
        pil_r = pil.resize((ref, ref), PILImage.LANCZOS)
        x = T.ToTensor()(pil_r).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
            if isinstance(out, (list, tuple)): out = out[0]
            out = out.squeeze(0).clamp(0, 1)
            arr = (out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        arr = cv2.resize(arr, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        log.warning("DDColor frame fail: %s", e)
        return img

# ================= CLIENT & PANEL UI =================
app = Client("anime_upscaler_bot", api_id=API_ID, api_hash=API_HASH,
             bot_token=BOT_TOKEN, in_memory=True)
archive = ChannelArchive(app, (os.getenv("ARCHIVE_CHANNEL_ID", "") or "").strip()) if ChannelArchive else None

_panel: Optional[Message] = None
_panel_lock = asyncio.Lock()
_panel_mode = "main"
_panel_last_edit_time = 0.0
_panel_last_text      = ""

def _core_label() -> str:
    if settings["core"] == "auto": return "🤖 Auto"
    cm = CORE_MAP.get(settings["core"])
    return cm[1] if cm else "🤖 Auto"

def _model_label() -> str:
    return MODELS.get(settings["model"], MODELS["anime_video"])["label"]

def _color_label() -> str:
    return {"off": "🎨 OFF", "fast": "🎨 Fast", "high": "💎 High"}.get(settings["colorize_mode"], "🎨 OFF")

def panel_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("😊", callback_data="b:mood"),
         InlineKeyboardButton(_model_label(), callback_data="b:mmenu"),
         InlineKeyboardButton(f"🎯 {fmt_scale(settings['scale'])}×", callback_data="b:qmenu")],
        [InlineKeyboardButton(f"⚡ {settings['preset'].title()}", callback_data="b:pmenu"),
         InlineKeyboardButton(f"🔊 {settings['audio'].title()}", callback_data="b:amenu"),
         InlineKeyboardButton(_color_label(), callback_data="b:colormenu")],
        [InlineKeyboardButton(_core_label(), callback_data="b:cmenu"),
         InlineKeyboardButton("📊 Stats", callback_data="b:stats"),
         InlineKeyboardButton("🧭 Help", callback_data="b:help")],
        [InlineKeyboardButton("🎥 Video", callback_data="b:sendv"),
         InlineKeyboardButton("🖼 Photo", callback_data="b:sendp"),
         InlineKeyboardButton("🎞 GIF", callback_data="b:sendg")],
        [InlineKeyboardButton("⛔ Stop", callback_data="b:stop"),
         InlineKeyboardButton("🧹 Clean", callback_data="b:clean")],
    ])

def models_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎌 Anime Video (fast)", callback_data="b:m:anime_video")],
        [InlineKeyboardButton("🎌 Anime Image (photo)", callback_data="b:m:anime_image")],
        [InlineKeyboardButton("🎮 Game Fast (FF/PUBG)", callback_data="b:m:game")],
        [InlineKeyboardButton("📷 Real Photo (HQ)", callback_data="b:m:real")],
        [InlineKeyboardButton("🔙 Panel", callback_data="b:back")],
    ])

def q_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("1.5×", callback_data="b:q:1.5"), InlineKeyboardButton("2×", callback_data="b:q:2"),
         InlineKeyboardButton("3×", callback_data="b:q:3"), InlineKeyboardButton("4×", callback_data="b:q:4")],
        [InlineKeyboardButton("🔙 Panel", callback_data="b:back")]])

def p_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚡ Fast", callback_data="b:p:fast"),
         InlineKeyboardButton("⚖️ Balanced", callback_data="b:p:balanced"),
         InlineKeyboardButton("💎 Best", callback_data="b:p:best")],
        [InlineKeyboardButton("🔙 Panel", callback_data="b:back")]])

def a_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔊 Keep", callback_data="b:a:keep"),
         InlineKeyboardButton("🗜 Compress", callback_data="b:a:compress"),
         InlineKeyboardButton("🔇 Remove", callback_data="b:a:remove")],
        [InlineKeyboardButton("🔙 Panel", callback_data="b:back")]])

def color_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚫 OFF", callback_data="b:col:off")],
        [InlineKeyboardButton("🎨 Fast (video+photo)", callback_data="b:col:fast")],
        [InlineKeyboardButton("💎 High (photo only)", callback_data="b:col:high")],
        [InlineKeyboardButton("🔙 Panel", callback_data="b:back")]])

def core_kb() -> InlineKeyboardMarkup:
    rows = []; row = []
    for key, label, w, t in CORE_PROFILES:
        row.append(InlineKeyboardButton(label, callback_data=f"b:c:{key}"))
        if len(row) == 2: rows.append(row); row = []
    if row: rows.append(row)
    rows.append([InlineKeyboardButton("🔙 Panel", callback_data="b:back")])
    return InlineKeyboardMarkup(rows)

SUBMENU_BUTTONS = {"mmenu", "qmenu", "pmenu", "amenu", "colormenu", "cmenu"}

HELP_TEXT = (
    "🧭 **Help (v11.0 Manager)**\n\n"
    "🎥 Video/GIF/Photo bhejo → turant ✅ tick message + live progress\n"
    "📚 Batch: multiple videos queue me, har ek ka apna status\n"
    "🎛 Models → Anime Video / Game = videos • Anime Image / Real = photos\n"
    "🎨 Color → OFF / Fast / High(photo) • ⚙️ Cores → Auto / Fixed\n"
    "✍️ /start /panel /reset /stats /cancel /queue"
)

def panel_text() -> str:
    m = MODELS.get(settings["model"], MODELS["anime_video"])
    lines = [_pad(f"{EMO.face()}  UPSCALER v11.0 (Manager)"), "─" * PW,
             _pad(f"🧠 {CPU_THREADS}c • 🛡 {mem_avail_gb():.1f}GB free • load {load1():.1f}"),
             _pad(f"{m['label']} {fmt_scale(settings['scale'])}× "
                  f"{settings['preset'][:4]} 🔊{settings['audio'][:4]}"),
             _pad(f"{_color_label()} • ⚙️ {_core_label()}"),
             _pad("")]
    if job_state.get("active") and current_job:
        j = current_job
        if j.get("total"):
            done = j.get("encoded", j.get("done", 0))
            pct = done * 100 / j["total"] if j["total"] > 0 else 0
            lines += [_pad(f"{j.get('stage', '🎨')} {j['filename'][:18]}"),
                      _pad(f"{bar(pct, 10)} {pct:.0f}%"),
                      _pad(f"🎞 {done}/{j['total']} • ETA {fmt_time(j.get('eta', 0))}"),
                      _pad(j.get("ai", "")[:PW])]
        else:
            lines += [_pad(f"{j.get('stage', '📥')} {j.get('filename', '')[:18]}")] + [_pad("")] * 2
    else:
        lines += [_pad("😴 Idle — video/photo bhejo"),
                  _pad("✅ tick + live progress milega"),
                  _pad("📚 batch = queue system")]
    if JOB_QUEUE:
        lines += [_pad(f"📋 Queue me: {len(JOB_QUEUE)} video")]
    lines += ["─" * PW, _pad("🎨 High color = photos only"),
              _pad("🎬 videos = SRVGG fast models")]
    return "\n".join(lines)

async def send_panel(cid: int) -> Optional[Message]:
    global _panel, _panel_mode, _panel_last_edit_time, _panel_last_text
    async with _panel_lock:
        if _panel is not None:
            try: await _panel.delete()
            except Exception: pass
            _panel = None
        for attempt in range(1, 4):
            try:
                txt = panel_text()
                msg = await app.send_message(cid, txt, reply_markup=panel_kb())
                _panel = msg; _panel_mode = "main"
                _panel_last_text = txt
                _panel_last_edit_time = time.time()
                return msg
            except Exception as e:
                log.warning("Panel send attempt %d fail: %s", attempt, e)
                await asyncio.sleep(2)
        return None

async def ensure_panel(cid: int) -> Optional[Message]:
    global _panel, _panel_last_edit_time, _panel_last_text
    if _panel is not None: return _panel
    async with _panel_lock:
        if _panel is not None: return _panel
        try:
            txt = panel_text()
            _panel = await app.send_message(cid, txt, reply_markup=panel_kb())
            _panel_last_text = txt
            _panel_last_edit_time = time.time()
            return _panel
        except Exception:
            return None

async def refresh_panel():
    global _panel, _panel_last_edit_time, _panel_last_text
    if _panel is None: return
    now = time.time()
    if (now - _panel_last_edit_time) < PANEL_MIN_EDIT_INTERVAL:
        return
    txt = panel_text()
    if txt == _panel_last_text:
        _panel_last_edit_time = now
        return
    try:
        await _panel.edit_text(txt, reply_markup=panel_kb())
        _panel_last_text = txt
        _panel_last_edit_time = now
    except FloodWait as e:
        log.warning("⏳ panel FloodWait %ss", e.value)
        _panel_last_edit_time = now + e.value
    except Exception as e:
        err = str(e).lower()
        if "not modified" in err:
            _panel_last_text = txt
            _panel_last_edit_time = now
            return
        if ("message_id_invalid" in err or "message to edit not found" in err
                or "deleted" in err):
            _panel = None

@app.on_callback_query(filters.regex(r"^b:"))
async def btn(client, cq):
    global _panel, _panel_mode, _panel_last_edit_time, _panel_last_text
    if not is_owner(cq.message.chat.id):
        await cq.answer("Private bot!", show_alert=True); return
    parts = cq.data[2:].split(":"); a = parts[0]; v = parts[1] if len(parts) > 1 else ""
    kb = None

    if a in SUBMENU_BUTTONS:
        _panel_mode = "submenu"
        if a == "mmenu": kb = models_kb(); await cq.answer("🎽 Model chuno")
        elif a == "qmenu": kb = q_kb(); await cq.answer("🎯 Scale chuno")
        elif a == "pmenu": kb = p_kb(); await cq.answer("⚡ Preset chuno")
        elif a == "amenu": kb = a_kb(); await cq.answer("🔊 Audio chuno")
        elif a == "colormenu": kb = color_kb(); await cq.answer("🎨 Colorize mode")
        elif a == "cmenu": kb = core_kb(); await cq.answer("⚙️ Cores chuno")
    elif a == "back":
        _panel_mode = "main"; kb = panel_kb(); await cq.answer("🔙")
    elif a == "m":
        if v in MODELS:
            settings["model"] = v
            if archive: archive.state["model"] = v
            _panel_mode = "main"; kb = panel_kb()
            note = " (photo-model: videos par auto SRVGG)" if not MODELS[v]["video_ok"] else ""
            await cq.answer(f"{MODELS[v]['label']}{note}")
        else:
            kb = models_kb(); await cq.answer()
    elif a == "q":
        try: settings["scale"] = float(v)
        except Exception: settings["scale"] = 2.0
        if archive: archive.state["scale"] = settings["scale"]
        _panel_mode = "main"; kb = panel_kb(); await cq.answer(f"{v}×")
    elif a == "p":
        if v in PRESETS:
            settings["preset"] = v
            if archive: archive.state["preset"] = v
        _panel_mode = "main"; kb = panel_kb(); await cq.answer(f"{v}")
    elif a == "a":
        if v in ("keep", "compress", "remove"):
            settings["audio"] = v
            if archive: archive.state["audio"] = v
        _panel_mode = "main"; kb = panel_kb(); await cq.answer(f"{v}")
    elif a == "col":
        if v in ("off", "fast", "high"):
            settings["colorize_mode"] = v
            if archive: archive.state["colorize_mode"] = v
            _panel_mode = "submenu"; kb = color_kb()
            await cq.answer(f"🎨 {v}" + (" (photo only)" if v == "high" else ""))
        else:
            kb = color_kb(); await cq.answer()
    elif a == "c":
        if v in CORE_MAP:
            settings["core"] = v
            _panel_mode = "submenu"; kb = core_kb()
            lbl = "Auto (Governor)" if v == "auto" else CORE_MAP[v][1]
            await cq.answer(f"⚙️ {lbl}")
        else:
            kb = panel_kb(); await cq.answer()
    elif a == "help":
        await cq.answer(); await cq.message.reply_text(HELP_TEXT); return
    elif a == "stats":
        await cq.answer(); await cq.message.reply_text(_stats_text()); return
    elif a == "sendv":
        await cq.answer(); await cq.message.reply_text("🎥 Ab **video** bhejo — ✅ tick + live progress milega!"); return
    elif a == "sendp":
        await cq.answer(); await cq.message.reply_text("🖼 Ab **photo** bhejo!"); return
    elif a == "sendg":
        await cq.answer(); await cq.message.reply_text("🎞 Ab **GIF** bhejo!"); return
    elif a == "mood":
        cur = EMO.state if EMO.state in MOOD_ORDER else "idle"
        EMO.set(MOOD_ORDER[(MOOD_ORDER.index(cur) + 1) % len(MOOD_ORDER)])
        kb = panel_kb(); await cq.answer(EMO.face())
    elif a == "stop":
        if cancel_event: cancel_event.set()
        await cq.answer("⛔ Cancel!"); return
    elif a == "clean":
        try:
            if _panel: await _panel.delete()
        except Exception: pass
        _panel = None
        _panel_last_text = ""
        _panel_last_edit_time = 0.0
        await send_panel(cq.message.chat.id)
        await cq.answer("🧹 Clean"); return
    else:
        await cq.answer(); return

    if archive:
        try: asyncio.create_task(archive.save_state())
        except Exception: pass

    if kb is not None:
        try:
            txt = panel_text()
            if _panel and cq.message.id == _panel.id:
                await _panel.edit_text(txt, reply_markup=kb)
                _panel_last_text = txt
                _panel_last_edit_time = time.time()
            else:
                await cq.message.edit_text(txt, reply_markup=kb)
                _panel = cq.message
                _panel_last_text = txt
                _panel_last_edit_time = time.time()
        except Exception as e:
            err = str(e).lower()
            if "not modified" in err:
                pass
            else:
                log.warning("Callback edit fail: %s", str(e)[:120])
                try: await send_panel(cq.message.chat.id)
                except Exception: pass

def _stats_text() -> str:
    m = MODELS.get(settings["model"], MODELS["anime_video"])
    lines = [f"📊 **Stats**",
             f"⚙️ Cores: {_core_label()}",
             f"🎽 Model: {m['label']} — {m['best_for']}",
             f"🎨 Colorize: {_color_label()}",
             f"📋 Queue: {len(JOB_QUEUE)} waiting",
             f"🖥 Cores: {CPU_THREADS} • 🛡 RAM: {mem_avail_gb():.1f}GB free",
             f"🎯 Scale: {fmt_scale(settings['scale'])}× • Preset: {settings['preset'].title()}"]
    if archive:
        st = archive.state
        hist = st.get("history", [])
        tot_t = sum(x.get("t", 0) for x in hist)
        lines += [f"✅ Jobs done: {st.get('jobs_done', 0)}", f"🕒 Total: {fmt_time(tot_t)}"]
    return "\n".join(lines)

# ================= PROBE =================
def probe_video(path: Path) -> Dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                        "-show_streams", "-show_format", str(path)],
                       capture_output=True, text=True, check=True)
    data = json.loads(r.stdout)
    vid = next(s for s in data["streams"] if s.get("codec_type") == "video")
    fs = vid.get("avg_frame_rate") or vid.get("r_frame_rate") or "30/1"
    n, d = fs.split("/"); fps = float(n) / float(d) if float(d) else 30.0
    dur = float(vid.get("duration") or data.get("format", {}).get("duration") or 0)

    raw_frames = vid.get("nb_frames")
    frames = 0
    if raw_frames not in (None, "", "N/A"):
        try: frames = int(float(raw_frames))
        except (TypeError, ValueError): pass

    if frames <= 0 and dur > 0 and fps > 0:
        frames = max(1, int(round(dur * fps)))
    frames = max(0, frames)

    raw_w = int(vid["width"]); raw_h = int(vid["height"])
    rot = 0.0
    for sd in (vid.get("side_data_list") or []):
        if isinstance(sd, dict) and "rotation" in sd:
            try: rot = float(sd["rotation"])
            except Exception: pass
    if rot == 0.0:
        try: rot = float((vid.get("tags") or {}).get("rotate", 0) or 0)
        except Exception: rot = 0.0

    if abs(rot) in (90.0, 270.0):
        w, h = raw_h, raw_w
        log.info("🔄 Rotation %.0f° — display %sx%s (raw %sx%s)", rot, w, h, raw_w, raw_h)
    else:
        w, h = raw_w, raw_h

    return {"width": w, "height": h, "raw_width": raw_w, "raw_height": raw_h,
            "rotation": rot, "fps": fps, "duration": dur, "frames": frames,
            "has_audio": any(s.get("codec_type") == "audio" for s in data["streams"])}

# ================= PIPELINE =================
def _build_vf_chain(w: int, h: int, rot: float) -> str:
    parts = []
    if abs(rot) in (90.0, 270.0):
        t = "1" if rot > 0 else "2"
        parts.append(f"transpose={t}")
    parts.append(f"scale={w}:{h}:flags=fast_bilinear")
    parts.append("format=bgr24")
    return ",".join(parts)

def read_exact(pipe, size):
    buf = bytearray()
    while len(buf) < size:
        chunk = pipe.read(size - len(buf))
        if not chunk: break
        buf.extend(chunk)
    return bytes(buf)

def run_pipeline(job, in_path: Path, out_path: Path, info: Dict, ups_queue: queue.Queue,
                 inst_count: int, gov: Governor, cancel: threading.Event, is_gif: bool,
                 colorize_mode: str, colorize_ref: int, cfg_snapshot: Dict[str, Any], job_dir: Path):
    w, h, fps = info["width"], info["height"], info["fps"]
    rot = info.get("rotation", 0.0)
    ow, oh = job["ow"], job["oh"]
    scale = float(cfg_snapshot.get("scale", 2.0))
    ff = PRESETS.get(cfg_snapshot.get("preset", "balanced"), PRESETS["balanced"])
    audio_mode = cfg_snapshot.get("audio", "keep")
    enc_threads = 1 if inst_count >= 2 else max(1, min(2, CPU_THREADS))
    fps_g = fps if fps > 0 else 10.0
    source_total = int(info.get("frames") or 0)
    total = min(source_total if source_total > 0 else max(1, int(round(info.get("duration", 0) * fps_g))),
                MAX_GIF_FRAMES if is_gif else MAX_FRAMES)
    total = max(1, total)
    job["total"] = total

    stats_lock = threading.Lock()
    encoded = [0]
    encoded_lock = threading.Lock()

    eta_tracker = {"last_time": time.time(), "last_done": 0, "ema_fps": 0.0}

    dec = enc = None
    in_q = queue.Queue(maxsize=max(2, inst_count))

    pending_futs = {}
    futs_cond = threading.Condition()

    t_start = time.time()
    err_flags = {"reader": None, "encoder": None}

    vf_chain = _build_vf_chain(w, h, rot)
    log.info("📐 Reader: vf='%s' frame=%sx%s | instances=%d", vf_chain, w, h, inst_count)

    def safe_put(item):
        while True:
            if cancel.is_set(): return False
            try:
                in_q.put(item, timeout=0.2)
                return True
            except queue.Full:
                pass

    def reader():
        nonlocal dec
        fb = w * h * 3
        log_path = job_dir / "dec_err.log"
        try:
            with open(log_path, "w+") as err_out:
                dec = subprocess.Popen(
                    ["ffmpeg", "-v", "error", "-noautorotate", "-i", str(in_path),
                     "-an", "-sn", "-dn", "-vsync", "0", "-vf", vf_chain,
                     "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                    stdout=subprocess.PIPE, stderr=err_out, bufsize=0)
                n = 0
                while not cancel.is_set():
                    raw = read_exact(dec.stdout, fb)
                    if not raw or len(raw) != fb: break
                    if is_gif and n >= MAX_GIF_FRAMES: break

                    frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
                    if not safe_put(frame):
                        del frame; break
                    n += 1

                if dec.poll() is None:
                    try: dec.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        dec.kill(); dec.wait(timeout=1)

                if dec.returncode not in (0, None) and not cancel.is_set():
                    err_out.seek(0)
                    err_txt = err_out.read()
                    err_flags["reader"] = f"Decoder failed ({dec.returncode}): {err_txt[-300:]}"

                if not cancel.is_set():
                    job["total"] = n
        except Exception as e:
            err_flags["reader"] = str(e)
        finally:
            safe_put(None)
            cancel.set() if err_flags["reader"] else None

    def upscale_one(img, cfg, idx):
        if cancel.is_set():
            del img; return None
        t0 = time.time()
        ups = ups_queue.get()
        out = None
        fail_err = None
        try:
            try:
                out, _ = ups.enhance(img, outscale=scale)
            except RuntimeError as e:
                estr = str(e).lower()
                if "out of memory" in estr or "alloc" in estr:
                    log.warning("OOM on frame %d — retrying tile=256...", idx)
                    gc.collect()
                    if not ups.tile or ups.tile > 256: ups.tile = 256
                    try:
                        out, _ = ups.enhance(img, outscale=scale)
                    except Exception as e2:
                        fail_err = e2
                else:
                    fail_err = e
        except Exception as e:
            fail_err = e
        finally:
            ups_queue.put(ups)
            del img

        if cancel.is_set():
            return None
        if out is None:
            log.error("Upscale frame %d fail (after retry): %s", idx, fail_err)
            raise RuntimeError(f"Frame {idx} upscale failed: {fail_err}")

        if colorize_mode != "off":
            out = colorize_frame(out, colorize_mode, colorize_ref)
        if len(out.shape) == 3 and out.shape[2] == 4:
            out = out[:, :, :3]
        elif len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        if out.shape[1] != ow or out.shape[0] != oh:
            out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_LINEAR)
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)

        dt = time.time() - t0
        with stats_lock:
            job["processed"] = job.get("processed", 0) + 1
            processed = job["processed"]
            job["spf"] = dt if processed == 1 else (job.get("spf", dt) * 0.9 + dt * 0.1)
            job["ai"] = gov.status(job["spf"])
        gov.on_frame(cfg, dt, processed)
        return np.ascontiguousarray(out)

    def encoder():
        nonlocal enc
        log_path = job_dir / "enc_err.log"
        try:
            with open(log_path, "w+") as err_out:
                cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                       "-s", f"{ow}x{oh}", "-r", f"{fps_g:.6f}", "-i", "pipe:0"]
                if not is_gif:
                    cmd += ["-i", str(in_path), "-map", "0:v:0"]
                    if info["has_audio"]:
                        if audio_mode == "keep": cmd += ["-map", "1:a?", "-c:a", "copy"]
                        elif audio_mode == "compress": cmd += ["-map", "1:a?", "-c:a", "aac", "-b:a", "128k"]
                    cmd += ["-map_metadata", "-1"]

                cmd += ["-c:v", "libx264", "-preset", ff["preset"], "-crf", ff["crf"],
                        "-threads", str(enc_threads), "-pix_fmt", "yuv420p"]

                if not is_gif and info["has_audio"] and audio_mode != "remove":
                    cmd += ["-shortest"]

                cmd += ["-movflags", "+faststart", str(out_path)]
                enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err_out)

                i = 0
                while not cancel.is_set():
                    f = None
                    with futs_cond:
                        while i not in pending_futs and not cancel.is_set():
                            futs_cond.wait(0.2)
                        if cancel.is_set(): break
                        f = pending_futs.pop(i)

                    if f is None:
                        break

                    arr = None
                    while not cancel.is_set():
                        try:
                            arr = f.result(timeout=0.2)
                            break
                        except concurrent.futures.TimeoutError:
                            continue

                    if cancel.is_set() or arr is None: break

                    enc.stdin.write(arr.tobytes())
                    del arr

                    with encoded_lock:
                        encoded[0] += 1
                        enc_count = encoded[0]

                    job["encoded"] = enc_count

                    now = time.time()
                    dt = now - eta_tracker["last_time"]
                    if dt >= 1.0:
                        df = enc_count - eta_tracker["last_done"]
                        inst_fps = df / dt
                        ema = eta_tracker["ema_fps"]
                        eta_tracker["ema_fps"] = inst_fps if ema == 0 else (ema * 0.8 + inst_fps * 0.2)
                        eta_tracker["last_time"] = now
                        eta_tracker["last_done"] = enc_count

                    tp = eta_tracker["ema_fps"]
                    job["throughput"] = tp
                    tot = job.get("total", total)
                    job["eta"] = max(0, tot - enc_count) / tp if tp > 0 else 0
                    i += 1

                if enc.stdin: enc.stdin.close()
                rc = enc.wait(timeout=7200)
                if rc != 0 and not cancel.is_set():
                    err_out.seek(0)
                    err_txt = err_out.read()
                    err_flags["encoder"] = f"FFmpeg encode failed ({rc}): {err_txt[-300:]}"

        except Exception as e:
            err_flags["encoder"] = str(e)
            cancel.set()
        finally:
            with futs_cond: futs_cond.notify_all()

    th_read = threading.Thread(target=reader, name="decoder", daemon=True)
    th_read.start()
    th_enc = threading.Thread(target=encoder, name="encoder", daemon=True)
    th_enc.start()

    try:
        job["stage"] = "🎨"
        i = 0
        while not cancel.is_set():
            if time.time() - t_start > MAX_JOB_SEC:
                raise RuntimeError("Job time-limit exceeded")

            try: img = in_q.get(timeout=0.2)
            except queue.Empty:
                if not th_read.is_alive() and in_q.empty(): break
                continue

            if img is None: break

            while not cancel.is_set():
                with encoded_lock: enc_now = encoded[0]
                gate = max(1, min(inst_count, gov.current[0]))
                if (i - enc_now) < (gate + 1):
                    break
                time.sleep(0.01)

            if cancel.is_set():
                del img; break

            cfg = tuple(gov.current)
            f = POOL.submit(upscale_one, img, cfg, i)
            with futs_cond:
                pending_futs[i] = f
                futs_cond.notify_all()
            i += 1

        with futs_cond:
            pending_futs[i] = None
            futs_cond.notify_all()

        th_enc.join(timeout=7200)
        if th_enc.is_alive(): raise RuntimeError("Encoder timeout/hung")

        th_read.join(timeout=30)
        if th_read.is_alive(): raise RuntimeError("Decoder hung")

        if err_flags["reader"]: raise RuntimeError(err_flags["reader"])
        if err_flags["encoder"]: raise RuntimeError(err_flags["encoder"])
        if cancel.is_set(): raise RuntimeError("Job Cancelled")

        job["frames_done"] = encoded[0]
        job["done"] = encoded[0]
        job["seconds"] = time.time() - t_start
        job["eta"] = 0

    finally:
        cancel.set()
        with futs_cond: futs_cond.notify_all()
        for p in (dec, enc):
            try:
                if p and p.poll() is None: p.kill()
            except Exception: pass
        for th in (th_read, th_enc):
            try:
                if th and th.is_alive(): th.join(timeout=2)
            except Exception: pass
        gc.collect()

def loop_short_clip_if_needed(job: Dict[str, Any], out_path: Path, is_gif: bool,
                              fps_hint: float, job_dir: Optional[Path]) -> Path:
    if not is_gif:
        return out_path
    frames_done = job.get("frames_done", job.get("done", 0))
    fps_g = fps_hint if fps_hint and fps_hint > 0 else 10.0
    if frames_done <= 0:
        return out_path
    duration = frames_done / fps_g
    if duration >= GIF_MIN_SEC:
        return out_path
    loops = min(max(1, math.ceil(GIF_MIN_SEC / duration)), max(1, 600 // max(1, frames_done)))
    if loops <= 1:
        return out_path
    looped_path = (job_dir / f"looped_{out_path.name}") if job_dir else out_path.with_name(f"looped_{out_path.name}")
    cmd = ["ffmpeg", "-y", "-v", "error", "-stream_loop", str(loops - 1),
           "-i", str(out_path), "-c", "copy", "-movflags", "+faststart", str(looped_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and looped_path.exists() and looped_path.stat().st_size > 0:
            out_path.unlink(missing_ok=True)
            looped_path.rename(out_path)
            job["loops"] = loops
            log.info("🔁 GIF looped ×%d to reach %.1fs", loops, GIF_MIN_SEC)
        else:
            log.warning("GIF loop pass failed (rc=%s): %s", r.returncode, (r.stderr or "")[-300:])
    except Exception as e:
        log.warning("GIF loop pass error: %s", e)
    return out_path

# ================= UPLOAD RETRY =================
async def send_with_retry(fn, desc: str):
    for attempt in range(1, 4):
        try: return await fn()
        except FloodWait as fw:
            log.warning("⏳ FloodWait %ss (%s)", fw.value, desc)
            await asyncio.sleep(fw.value + 2)
        except Exception as e:
            log.warning("📤 %s attempt %s fail: %s", desc, attempt, e)
            if attempt == 3: raise
            await asyncio.sleep(5 * attempt)

# ================= LIVE STATUS LOOP (milestone-based) =================
async def _job_status_loop(status_msg: Message, stop_evt: asyncio.Event):
    last_milestone = -1
    last_edit_time = 0.0
    last_text      = ""

    while not stop_evt.is_set():
        try:
            j = current_job
            if j and j.get("total"):
                done  = j.get("encoded", j.get("done", 0))
                total = j["total"]
                pct   = done * 100 / total if total > 0 else 0.0

                crossed = None
                for m in STATUS_MILESTONES:
                    if last_milestone < m <= pct:
                        crossed = m

                should_edit = False
                if crossed is not None:
                    last_milestone = crossed
                    should_edit = True
                elif (time.time() - last_edit_time) > STATUS_STUCK_SEC and pct < 100:
                    should_edit = True

                if should_edit:
                    txt = (f"{j.get('stage', '🎨')} **{j['filename'][:20]}**\n"
                           f"{bar(pct)} {pct:.0f}%\n"
                           f"🎞 {done}/{total} fr • ⏱ ETA {fmt_time(j.get('eta', 0))}\n"
                           f"⚙️ {j.get('throughput', 0):.2f} fr/s")
                    if txt != last_text:
                        await status_msg.edit_text(txt)
                        last_text = txt
                        last_edit_time = time.time()

            elif j:
                txt = f"{j.get('stage', '📥')} **{j.get('filename', '')[:20]}** …"
                if txt != last_text and (time.time() - last_edit_time) > 8.0:
                    await status_msg.edit_text(txt)
                    last_text = txt
                    last_edit_time = time.time()

        except FloodWait as e:
            log.warning("⏳ status FloodWait %ss", e.value)
            try: await asyncio.wait_for(stop_evt.wait(), timeout=e.value + 2)
            except asyncio.TimeoutError: pass
            continue
        except Exception as e:
            err = str(e).lower()
            if ("not modified" in err or "message_id_invalid" in err
                    or "not found" in err or "message to edit" in err):
                pass
        try: await asyncio.wait_for(stop_evt.wait(), timeout=STATUS_POLL_EVERY)
        except asyncio.TimeoutError: pass

# ================= PANEL REFRESH LOOP =================
async def _refresh_loop():
    global _panel_mode
    hb = 0
    owner = _owner_cid()
    while True:
        try:
            if job_state.get("active"): _panel_mode = "job"
            elif _panel_mode == "job": _panel_mode = "main"
            if job_state.get("active") and current_job:
                st = current_job.get("stage", "")
                if st.startswith("📥"): EMO.set("download")
                elif st.startswith("🔍"): EMO.set("think")
                elif st.startswith("⬆️"): EMO.set("upload")
                elif st.startswith("✅"): EMO.set("happy")
                elif st.startswith("❌"): EMO.set("error")
                else: EMO.set("work")
            else:
                EMO.set("idle")
            if _panel is None and owner:
                await ensure_panel(owner)
            elif _panel_mode in ("main", "job"):
                await refresh_panel()
            hb += 1
            if hb % 20 == 0 and job_state.get("active") and current_job:
                log.info("💓 %s/%s fr | RAM %.1fGB | load %.1f",
                         current_job.get("encoded", 0),
                         current_job.get("total", 0),
                         mem_avail_gb(), load1())
        except Exception as e:
            log.warning("refresh_loop err: %s", e)
        await asyncio.sleep(PANEL_EVERY)

# ================= INTAKE =================
@app.on_message((filters.video | filters.document | filters.animation) & filters.private)
async def media_handler(client, message: Message):
    if not is_owner(message.chat.id):
        await message.reply_text("❌ Private bot."); return
    media = message.video or message.document or message.animation
    filename = getattr(media, "file_name", None) or f"video_{message.id}.mp4"
    mime = getattr(media, "mime_type", "") or ""
    is_gif = filename.lower().endswith(".gif") or mime == "image/gif"
    if not filename.lower().endswith((".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".gif")):
        await message.reply_text("❌ Sirf video/GIF bhejo."); return
    pos = len(JOB_QUEUE) + (1 if job_state.get("active") else 0) + 1
    status_msg = await message.reply_text(
        f"✅ **Video mil gayi!** `{filename}`\n"
        f"📋 Queue position: {pos}\n"
        f"⏳ Process start hote hi LIVE progress yahan dikhega...")
    JOB_QUEUE.append({"message": message, "filename": filename,
                      "is_gif": is_gif, "status_msg": status_msg})
    await _ensure_queue_loop()
    await refresh_panel()

async def _ensure_queue_loop():
    global _queue_task
    if _queue_task is None or _queue_task.done():
        _queue_task = asyncio.create_task(_queue_loop())

# ================= QUEUE PROCESSOR (DISTRIBUTED) =================
async def _dispatch_distributed(job_id: str, message: Message, filename: str, cfg: Dict[str,Any], status_msg: Message):
    if not GH_PAT or not GH_REPO: raise RuntimeError("GH_PAT/GH_REPO missing")
    
    # 1. Download Telegram file to manager's own filesystem
    media = message.video or message.document or message.animation
    file_id = getattr(media, "file_id", None)
    if not file_id: raise RuntimeError("Telegram file_id missing")
    
    temp_path = WORK_DIR / f"input_{job_id}.mp4"
    log.info("📥 Downloading video from Telegram to %s", temp_path)
    current_job["stage"] = "📥 Download"
    await app.download_media(message, file_name=str(temp_path))
    log.info("✅ Downloaded: %s bytes", temp_path.stat().st_size)
    
    # 2. Dispatch workflow
    payload = {
        "ref": "main",
        "inputs": {
            "job_id": job_id,
            "filename": filename,
            "model": normalize_model_key(cfg["model"]),
            "scale": str(int(float(cfg["scale"]))),
            "preset": cfg["preset"],
            "audio": cfg["audio"],
            "colorize": normalize_colorize(cfg["colorize_mode"]),
            "workers": str(DISTRIBUTED_MAX_WORKERS)
        }
    }
    hdr = {"Authorization": f"Bearer {GH_PAT}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    api = f"https://api.github.com/repos/{GH_REPO}"
    
    r = await asyncio.to_thread(requests.post, f"{api}/actions/workflows/{DISTRIBUTED_WORKFLOW}/dispatches", headers=hdr, json=payload, timeout=30)
    r.raise_for_status()
    log.info("Workflow dispatched. Waiting for run ID...")
    
    # 3. Find run ID
    t0 = time.time()
    run = None
    while time.time() - t0 < 60:
        await asyncio.sleep(3)
        q = await asyncio.to_thread(requests.get, f"{api}/actions/workflows/{DISTRIBUTED_WORKFLOW}/runs", headers=hdr, params={"event": "workflow_dispatch", "per_page": 20}, timeout=30)
        q.raise_for_status()
        for x in q.json().get("workflow_runs", []):
            if x.get("created_at") >= time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0 - 15)) and x.get("status") in ("queued", "in_progress"):
                run = x
                break
        if run: break
    if not run: raise RuntimeError("GitHub worker run ID nahi mila")
    rid = run["id"]
    log.info("Run ID: %s", rid)
    
    # 4. Create zip and upload as artifact
    import zipfile
    zip_path = WORK_DIR / f"input_{job_id}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(temp_path, "source_video.mp4")
    
    upload_url = f"{api}/actions/runs/{rid}/artifacts"
    upload_hdr = {
        "Authorization": f"Bearer {GH_PAT}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/zip"
    }
    log.info("⬆️ Uploading artifact...")
    current_job["stage"] = "⬆️ Upload"
    with open(zip_path, "rb") as f:
        upload_res = await asyncio.to_thread(requests.post, upload_url, headers=upload_hdr, data=f, timeout=300)
    upload_res.raise_for_status()
    log.info("✅ Artifact uploaded: %s", upload_res.status_code)
    
    # 5. Monitor run
    current_job["stage"] = "⚡ 20W START"
    current_job["total"] = 100
    current_job["done"] = 0
    start_monitor = time.time()
    
    while True:
        if time.time() - start_monitor > MAX_JOB_SEC:
            raise RuntimeError("Distributed job time-limit exceeded")
        q = await asyncio.to_thread(requests.get, f"{api}/actions/runs/{rid}/jobs", headers=hdr, params={"per_page": 100}, timeout=30)
        q.raise_for_status()
        jobs = q.json().get("jobs", [])
        active = sum(1 for j in jobs if j.get("status") in ("queued", "in_progress"))
        done = sum(1 for j in jobs if j.get("name", "").startswith("worker") and j.get("conclusion") == "success")
        current_job["done"] = done
        current_job["processed"] = done
        current_job["throughput"] = active
        current_job["stage"] = f"⚡ {done}/{DISTRIBUTED_MAX_WORKERS} workers • {active} active"
        current_job["spf"] = 0.0
        
        rr = await asyncio.to_thread(requests.get, f"{api}/actions/runs/{rid}", headers=hdr, timeout=30)
        rr.raise_for_status()
        run = rr.json()
        status = run.get("status")
        conclusion = run.get("conclusion")
        
        try:
            await status_msg.edit_text(
                f"⚡ **20-Worker Upscale** `{filename}`\n"
                f"{current_job['stage']}\n"
                f"⏱ {fmt_time(time.time() - start_monitor)}\n"
                f"🚀 Startup window: 7 min"
            )
        except Exception:
            pass
        
        if status == "completed":
            if conclusion != "success":
                raise RuntimeError(f"Distributed workflow {conclusion or 'failed'}")
            arts = await asyncio.to_thread(requests.get, f"{api}/actions/runs/{rid}/artifacts", headers=hdr, params={"per_page": 100}, timeout=30)
            arts.raise_for_status()
            final = next((a for a in arts.json().get("artifacts", []) if a.get("name") == f"final-{job_id}"), None)
            if not final:
                raise RuntimeError("Final artifact missing")
            
            z = WORK_DIR / f"final_{job_id}.zip"
            outdir = WORK_DIR / f"final_{job_id}"
            outdir.mkdir(exist_ok=True)
            log.info("⬇️ Downloading final artifact...")
            current_job["stage"] = "⬇️ Download final"
            with requests.get(final["archive_download_url"], headers=hdr, stream=True, timeout=120) as dl:
                dl.raise_for_status()
                with open(z, "wb") as f:
                    for c in dl.iter_content(1024 * 1024):
                        if c: f.write(c)
            with zipfile.ZipFile(z) as zz:
                zz.extractall(outdir)
            out = next(outdir.glob("*.mp4"), None)
            if not out:
                raise RuntimeError("Final video missing")
            
            # Cleanup manager temp files
            temp_path.unlink(missing_ok=True)
            zip_path.unlink(missing_ok=True)
            z.unlink(missing_ok=True)
            
            return out, time.time() - t0
        await asyncio.sleep(8)

async def _queue_loop():
    global current_job, cancel_event, _panel_mode
    while JOB_QUEUE:
        job = JOB_QUEUE.pop(0)
        cfg = dict(settings)
        message = job["message"]
        filename = job["filename"]
        status_msg = job["status_msg"]
        job_state["active"] = True
        cancel_event = threading.Event()
        _panel_mode = "job"
        jid = f"tg-{message.id}-{int(time.time())}"
        current_job = {
            "filename": filename, "ow": 0, "oh": 0, "done": 0, "total": 100,
            "spf": 0.0, "eta": 0.0, "throughput": 0.0, "processed": 0,
            "encoded": 0, "ai": "", "stage": "📤 dispatch", "loops": 1
        }
        try:
            await status_msg.edit_text(
                f"📤 **Manager 20 workers ko call kar raha hai...** `{filename}`\n"
                f"⏳ Video download + upload hoga, phir workers shuru karenge"
            )
            out, elapsed = await _dispatch_distributed(jid, message, filename, cfg, status_msg)
            size = out.stat().st_size / 1048576
            if size > MAX_SEND_MB:
                raise RuntimeError(f"Output {size:.0f}MB exceeds limit")
            current_job["stage"] = "⬆️ upload to TG"
            await send_with_retry(
                lambda: app.send_video(
                    message.chat.id, str(out),
                    caption=f"✅ **{filename}**\n"
                            f"🎯 {fmt_scale(float(cfg['scale']))}× • {cfg['preset']}\n"
                            f"⚡ 20-worker distributed • ⏱ {fmt_time(elapsed)} • 📦 {size:.1f}MB",
                    supports_streaming=True
                ),
                "video"
            )
            try:
                await status_msg.edit_text(
                    f"✅ **DONE!** `{filename}`\n"
                    f"⚡ 20-worker distributed\n"
                    f"⏱ {fmt_time(elapsed)} • 📦 {size:.1f}MB"
                )
            except Exception:
                pass
            if archive:
                try:
                    archive.record_job(filename, float(cfg["scale"]), elapsed, True, extra=cfg)
                    await archive.save_state()
                except Exception as e:
                    log.warning("Archive fail: %s", e)
        except Exception as e:
            log.exception("Distributed job failed")
            try:
                await status_msg.edit_text(f"❌ **FAIL:** `{filename}`\n{str(e)[:220]}")
            except Exception:
                pass
        finally:
            job_state["active"] = False
            current_job = None
            cancel_event = None
            _panel_mode = "main"
            gc.collect()
            EMO.set("idle")
            try:
                await refresh_panel()
            except Exception:
                pass
            for p in WORK_DIR.glob(f"final_{jid}*"):
                try:
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
                except Exception:
                    pass

# ================= PHOTO =================
busy_lock = asyncio.Lock()
@app.on_message(filters.photo & filters.private)
async def photo_handler(client, message: Message):
    if not is_owner(message.chat.id):
        await message.reply_text("❌ Private bot."); return
    async with busy_lock:
        if job_state.get("active") or JOB_QUEUE:
            await message.reply_text("⏳ Videos queue me hain — photo baad me bhejo."); return
        job_state["active"] = True
    job_cfg = dict(settings)
    try:
        EMO.set("work")
        tmp = WORK_DIR / f"photo_{message.id}.jpg"
        await app.download_media(message, file_name=str(tmp))
        img = cv2.imread(str(tmp))
        if img is None: raise RuntimeError("Image read fail")
        model_key = normalize_model_key(job_cfg["model"])
        if not MODELS[model_key]["video_ok"]:
            photo_model = model_key
        else:
            photo_model = {"anime_video": "anime_image", "game": "real"}.get(model_key, model_key)
            if (MODEL_DIR / MODELS[photo_model]["file"]).exists():
                model_key = photo_model
        h, w = img.shape[:2]
        photo_scale = float(job_cfg["scale"])
        ow = int(w * photo_scale); oh = int(h * photo_scale)
        while photo_scale > 1.0 and ow * oh > MAX_OUT_PIXELS:
            photo_scale = max(1.0, photo_scale - 0.5)
            ow = int(w * photo_scale); oh = int(h * photo_scale)
        tile = choose_tile(model_key, ow * oh)
        ups_q = await asyncio.to_thread(init_ups_pool, model_key, tile)
        colorize_mode = normalize_colorize(job_cfg["colorize_mode"])

        def _do():
            ups = ups_q.get()
            try:
                out = ups.enhance(img, outscale=photo_scale)[0]
            finally:
                ups_q.put(ups)
            if colorize_mode != "off":
                out = colorize_frame(out, colorize_mode, 512)
            if len(out.shape) == 3 and out.shape[2] == 4: out = out[:, :, :3]
            if out.dtype != np.uint8: out = np.clip(out, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(out)

        t0 = time.time()
        out = await asyncio.to_thread(_do)
        dt = time.time() - t0
        outp = WORK_DIR / f"photo_{message.id}_up.png"
        cv2.imwrite(str(outp), out)
        EMO.set("happy")
        await send_with_retry(lambda: app.send_photo(
            message.chat.id, str(outp),
            caption=f"✅ Photo {MODELS[model_key]['label']} • "
                    f"{fmt_scale(settings['scale'])}× • 🎨 {_color_label()}\n"
                    f"{w}×{h} → {out.shape[1]}×{out.shape[0]} • {dt:.1f}s"), "photo")
    except Exception as e:
        log.exception("Photo fail")
        EMO.set("error")
        try: await message.reply_text(f"❌ Photo fail: {str(e)[:200]}")
        except Exception: pass
    finally:
        job_state["active"] = False
        for f in WORK_DIR.glob(f"photo_{message.id}*"):
            try: f.unlink()
            except Exception: pass

# ================= COMMANDS =================
@app.on_message(filters.command("panel") & filters.private)
async def panel_cmd(client, message: Message):
    if not is_owner(message.chat.id): return
    await message.reply_text("🎛 Panel bhej raha hoon...")
    await send_panel(message.chat.id)

@app.on_message(filters.command("reset") & filters.private)
async def reset_cmd(client, message: Message):
    if not is_owner(message.chat.id): return
    settings.update({"scale": 2.0, "preset": "balanced", "audio": "keep",
                     "model": "anime_video", "core": "auto", "colorize_mode": "off"})
    if archive:
        try:
            archive.state.update({"scale": 2.0, "preset": "balanced", "audio": "keep",
                                  "model": "anime_video", "core": "auto", "colorize_mode": "off"})
            await archive.save_state()
        except Exception: pass
    await message.reply_text("🔄 Reset to defaults!")
    await send_panel(message.chat.id)

@app.on_message(filters.command("stats") & filters.private)
async def stats_cmd(client, message: Message):
    if not is_owner(message.chat.id): return
    await message.reply_text(_stats_text())

@app.on_message(filters.command("queue") & filters.private)
async def queue_cmd(client, message: Message):
    if not is_owner(message.chat.id): return
    if not JOB_QUEUE:
        await message.reply_text("📋 Queue khali hai."); return
    lines = [f"📋 **Queue ({len(JOB_QUEUE)}):**"]
    for i, j in enumerate(JOB_QUEUE, 1):
        lines.append(f"{i}. `{j['filename']}`")
    await message.reply_text("\n".join(lines))

@app.on_message(filters.command("cancel") & filters.private)
async def cancel_cmd(client, message: Message):
    if not is_owner(message.chat.id): return
    if cancel_event: cancel_event.set()
    await message.reply_text("🛑 Cancel bhej di (current job).")

@app.on_message(filters.forwarded & filters.private)
async def forward_id_handler(client, message: Message):
    if not is_owner(message.chat.id): return
    src = getattr(message, "forward_from_chat", None)
    if src is not None and getattr(src, "id", None):
        await message.reply_text(f"📌 Channel ID: `{src.id}`")
    else:
        await message.reply_text("❌ Forward se ID nahi mili.")

@app.on_message(filters.text & filters.private & ~filters.command(["start", "panel", "reset", "stats", "cancel", "queue"]))
async def text_handler(client, message: Message):
    if not is_owner(message.chat.id): return
    t = (message.text or "").lower()
    if any(k in t for k in ["hi", "hello", "hey", "namaste"]):
        EMO.set("happy")
        await message.reply_text(f"{EMO.one('happy')} Namaste boss! Video bhejo → ✅ tick + live progress.")
        if _panel is None: await ensure_panel(message.chat.id)
    elif any(k in t for k in ["game", "free fire", "pubg", "bgmi"]):
        await message.reply_text("🎮 Videos ke liye Game (Fast) model best hai.")
    elif "anime" in t:
        await message.reply_text("🎌 Video → Anime Video • Photo → Anime Image.")
    elif any(k in t for k in ["color", "rang"]):
        await message.reply_text("🎨 Color: Fast = video+photo • High = photo only.")
    elif any(k in t for k in ["ram", "cpu", "load"]):
        await message.reply_text(f"🛡 RAM {mem_avail_gb():.1f}GB • {CPU_THREADS}c")
    elif any(k in t for k in ["thank", "shukriya", "thx"]):
        await message.reply_text("Apna kaam hai boss!")
    else:
        await message.reply_text("🤖 v11.0: video bhejo → ✅ tick + LIVE progress; batch = queue.")

@app.on_message(filters.command("start") & filters.private)
async def start_handler(client, message: Message):
    if not is_owner(message.chat.id):
        await message.reply_text("❌ Private bot."); return
    EMO.set("start")
    await send_panel(message.chat.id)
    await message.reply_text(f"✅ **v11.0 online!** Chat ID: `{message.chat.id}`")

# ================= BOOT + PRE-WARM =================
def notify_owner_startup():
    try:
        r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          json={"chat_id": OWNER_CHAT_ID_INT or OWNER_CHAT_ID,
                                "text": "✅ Upscaler v11.0 online!\n✅ tick message + LIVE progress + batch queue."},
                          timeout=15)
        log.info("Startup ping: %s", r.status_code)
    except Exception as e:
        log.warning("Ping fail: %s", e)

def _prewarm():
    try:
        mk = normalize_model_key(settings["model"])
        if not MODELS[mk]["video_ok"]:
            mk = VIDEO_FALLBACK.get(mk, "anime_video")
        init_ups_pool(mk, choose_tile(mk, 2_000_000))
        log.info("🔥 Pre-warm: %s pool ready", mk)
    except Exception as e:
        log.warning("Pre-warm model fail: %s", e)
    try:
        cm = normalize_colorize(settings["colorize_mode"])
        if cm != "off":
            _load_ddcolor("fast")
            if cm == "high": _load_ddcolor("high")
            log.info("🔥 Pre-warm: DDColor ready")
    except Exception as e:
        log.warning("Pre-warm ddcolor fail: %s", e)

async def _boot():
    try: await asyncio.to_thread(notify_owner_startup)
    except Exception as e: log.warning("notify fail: %s", e)
    try:
        if archive:
            await archive.load()
            st = archive.state
            settings["scale"]  = float(st.get("scale", settings["scale"]))
            settings["preset"] = st.get("preset", settings["preset"])
            settings["audio"]  = st.get("audio", settings["audio"])
            settings["model"]  = normalize_model_key(st.get("model", settings["model"]))
            settings["core"]   = st.get("core", settings["core"])
            settings["colorize_mode"] = normalize_colorize(st.get("colorize_mode", settings["colorize_mode"]))
            log.info("📚 Archive settings (normalized): %s", settings)
    except Exception as e:
        log.warning("Archive boot fail: %s", e)
        settings["model"] = normalize_model_key(settings["model"])
        settings["colorize_mode"] = normalize_colorize(settings["colorize_mode"])

    threading.Thread(target=_prewarm, daemon=True).start()

    cid = _owner_cid()
    if cid:
        try: await send_panel(cid)
        except Exception as e: log.error("Panel send fail: %s", e)
    asyncio.create_task(_refresh_loop())
    log.info("🚀 v11.0 ready (cores=%s)", CPU_THREADS)

async def _main():
    try:
        await app.start()
        log.info("🔌 Client started — boot...")
        await _boot()
        await idle()
    finally:
        try: await app.stop()
        except Exception: pass

if __name__ == "__main__":
    try:
        r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook", timeout=15)
        log.info("🧹 Webhook: %s", r.text[:120])
    except Exception as e:
        log.warning("Webhook fail: %s", e)
    app.run(_main())
