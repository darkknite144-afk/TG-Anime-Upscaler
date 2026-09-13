#!/usr/bin/env python3
"""
Smart Anime/Game/Real Upscaler v10.4
FIXES:
  - ✅ Button submenu overwrite bug (panel_mode tracker)
  - ✅ Black lines / rotation distortion (explicit transpose)
  - ✅ Legacy archive migration (anime → anime_video)
  - ✅ 1-second strict refresh rate (no panel spamming)
  - ✅ Removed extra messages during processing
  - ✅ Strip Metadata (-map_metadata -1) to fix mobile rotation squishing
  - ✅ PyTorch C-Contiguous memory fix (Permanent garbled frame fix)
  - ✅ Strict OS pipe chunk reading (Fixes shifted frame artifacts)
"""
import asyncio, gc, json, logging, math, os, queue, random, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import cv2
import numpy as np
import requests
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
WORK_DIR = Path("work"); OUTPUT_DIR = Path("output")
WORK_DIR.mkdir(exist_ok=True); OUTPUT_DIR.mkdir(exist_ok=True)

MAX_FRAMES = 3600
MAX_GIF_FRAMES = 240
MAX_OUT_PIXELS = 3840 * 2160
MAX_SEND_MB = 1900
MAX_JOB_SEC = int(os.getenv("MAX_JOB_MIN", "300")) * 60
GIF_MIN_SEC = 2.0
IN_QUEUE = 2
OUT_BACKLOG = 2
CPU_THREADS = os.cpu_count() or 4
POOL = ThreadPoolExecutor(max_workers=4)
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)

# ================= MODELS =================
MODELS = {
    "anime_video": {"file": "realesr-animevideov3.pth",       "arch": "srvgg", "label": "🎌 Anime Video", "best_for": "anime video (fast)"},
    "anime_image": {"file": "RealESRGAN_x4plus_anime_6B.pth", "arch": "rrdb",  "label": "🎌 Anime Image", "best_for": "anime still (crisp)"},
    "game":        {"file": "realesr-general-x4v3.pth",       "arch": "srvgg", "label": "🎮 Game (Fast)", "best_for": "gameplay (FF/PUBG)"},
    "real":        {"file": "RealESRGAN_x4plus.pth",          "arch": "rrdb",  "label": "📷 Real Photo",  "best_for": "real-world photos"},
}
LEGACY_MODEL_MAP = {"anime": "anime_video", "game": "game", "gamehq": "real"}

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
                      (6, 1, "🔥🔥 Hexa 6×1")]:
        if w * t <= CPU_THREADS: p.append((f"m{w}x{t}", lbl, w, t))
    return p

CORE_PROFILES = _build_core_profiles()
CORE_MAP = {p[0]: p for p in CORE_PROFILES}

settings = {"scale": 2.0, "preset": "balanced", "audio": "keep",
            "model": "anime_video", "core": "auto", "colorize_mode": "off"}
job_state = {"active": False}
current_job: Optional[Dict[str, Any]] = None
cancel_event: Optional[threading.Event] = None

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
    s = max(0, int(s)); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h}h {m}m {sec}s" if h else (f"{m}m {sec}s" if m else f"{sec}s")

def fmt_scale(s: float) -> str: return f"{s:g}"

def bar(pct: float, n: int = 8) -> str:
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
    "work":     ["(っ⚙_⚙)っ⚡", "(っ⚙_⚙)っ✦", "(っ⚙_⚙)っ✧"],
    "think":    ["(◔_)…", "(◔‿◔)?", "(◕_◕)…"],
    "happy":    ["(ﾉ◕)ﾉ*:･ﾟ✧", "(◕‿◕)✧", "(＾▽＾)ﾉ★"],
    "error":    ["(×_×;)", "(╥_╥)…", "(⊙_)!"],
    "love":     ["(♥‿♥)", "(♡ω♡)", "(⁄⁄•⁄ω•⁄⁄)"],
    "start":    ["(ò_ó)⚡", "(◉◉)✧", "(ᐛ)و✦"],
    "upload":   ["(⇀↼)", "(_↼)️", "(⇀‿↼)🚀"],
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
DOWNGRADE = {(4, 1): (3, 1), (3, 1): (2, 2), (2, 3): (2, 2), (2, 2): (1, 2),
             (1, 4): (1, 2), (1, 2): (1, 1), (1, 1): (1, 1)}
PROBE, EXPLOIT = 6, 12

class Governor:
    def __init__(self, out_px: int, fixed: Optional[tuple] = None):
        self.fp = out_px * 512 / 1e9
        self.fixed = fixed
        self.ema = {c: 0.0 for c in CONFIGS}
        self.cnt = {c: 0 for c in CONFIGS}
        if fixed is not None:
            w, t = fixed
            max_w = max(1, int(mem_avail_gb() * 0.6 / max(self.fp, 0.1)))
            if w > max_w:
                log.warning("⚙️ Fixed core clamped %d→%d (RAM)", w, max_w); w = max_w
            start = (w, t)
        else:
            start = (2, 2) if self._ram_ok(2) else ((1, 2) if self._ram_ok(1) else (1, 1))
        self.current = start
        self.probe_left, self.exploit = PROBE, 0
        self.load_ema = load1(); self.ram_ema = mem_avail_gb()
        self.press = 0; self.idle = 0; self.safe = False
        self.lock = threading.Lock()
        self.apply()
        log.info("🧠 Governor %s%s | fp %.2fGB/fr | RAM %.1fGB",
                 self.current, " [FIXED]" if fixed else "", self.fp, self.ram_ema)

    def _ram_ok(self, w): return w * self.fp <= max(1.0, mem_avail_gb() * 0.7)
    def apply(self): torch.set_num_threads(self.current[1])
    def thr(self, c): return c[0] / self.ema[c] if self.ema[c] else 0.0

    def on_frame(self, cfg, dt: float, done: int):
        with self.lock:
            c = tuple(cfg)
            if c in self.ema:
                self.ema[c] = dt if self.cnt[c] == 0 else self.ema[c] * 0.7 + dt * 0.3
                self.cnt[c] += 1
            self.load_ema = self.load_ema * 0.8 + load1() * 0.2
            self.ram_ema = self.ram_ema * 0.8 + mem_avail_gb() * 0.2
            if done % 20 == 0: gc.collect()
            if self.fixed is not None:
                if self.ram_ema < 1.2: self.safe = True
                return
            if self.ram_ema < 1.2 or self.load_ema > CPU_THREADS * 1.5:
                self.press += 1; self.idle = 0
                if self.press >= 2:
                    self.press = 0; self.safe = True
                    nxt = DOWNGRADE.get(self.current, self.current)
                    if nxt != self.current:
                        self.current = nxt; self.apply()
                        log.warning("🛡 DOWNGRADE -> %s", nxt)
                    return
            else:
                self.press = 0
                if self.safe and self.ram_ema > 3.0 and self.load_ema < CPU_THREADS * 0.9:
                    self.safe = False
            if not self.safe and self.load_ema < CPU_THREADS * 0.55 and self.ram_ema > 4.0:
                self.idle += 1
                if self.idle >= 4:
                    self.idle = 0; self._probe(up=True); return
            else: self.idle = 0
            if self.safe: return
            if self.probe_left > 0:
                self.probe_left -= 1
                if self.probe_left == 0:
                    self._pick_best(); self.exploit = EXPLOIT
                return
            if self.exploit > 0:
                self.exploit -= 1
                if self.exploit == 0: self._probe()
                return
            self._probe()

    def _probe(self, up: bool = False):
        if up:
            target = next((c for c in CONFIGS if c[0] == self.current[0] + 1 and self._ram_ok(c[0])), None)
            if target is None: self.probe_left = PROBE; return
        else:
            cand = [c for c in CONFIGS if self.cnt[c] < 3 and self._ram_ok(c[0])]
            if not cand: cand = [c for c in CONFIGS if self._ram_ok(c[0])]
            if not cand: return
            target = cand[0]
        if target != self.current:
            self.current = target; self.apply()
        self.probe_left = PROBE

    def _pick_best(self):
        tested = [c for c in CONFIGS if self.cnt[c] >= 3 and self._ram_ok(c[0])]
        if not tested: return
        b = max(tested, key=self.thr)
        if b != self.current:
            self.current = b; self.apply()

    def status(self, spf: float) -> str:
        fx = " [FIXED]" if self.fixed else ""
        return (f"🤖 {self.current[0]}W×{self.current[1]}T{fx} | {spf:.2f}s/fr | "
                f"{self.thr(self.current):.2f} f/s | 🛡 {self.ram_ema:.1f}G | "
                f"load {self.load_ema:.1f}" + (" | SAFE" if self.safe else ""))

# ================= REAL-ESRGAN =================
_ups_cache: Dict[Any, RealESRGANer] = {}

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
    if MODELS[key]["arch"] == "rrdb": return 256
    return 0 if out_px <= 2_600_000 else 320

def get_ups(key: str, tile: int) -> RealESRGANer:
    key = normalize_model_key(key)
    k = (key, tile)
    if k in _ups_cache: return _ups_cache[k]
    m = MODELS[key]; path = MODEL_DIR / m["file"]
    if not path.exists(): raise FileNotFoundError(f"Model missing: {path}")

    if m["arch"] == "rrdb":
        log.info("Loading %s (RRDBNet, tile=%s)...", m["file"], tile)
        nb = 6 if "anime_6B" in m["file"] else 23
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=nb, num_grow_ch=32, scale=4)
        ups = RealESRGANer(scale=4, model_path=str(path), model=model, tile=tile,
                           tile_pad=16, pre_pad=0, half=False, device=torch.device("cpu"))
        _ups_cache[k] = ups; return ups

    nconv = _detect_srvgg_num_conv(path)
    log.info("Loading %s (SRVGG num_conv=%s, tile=%s)...", m["file"], nconv, tile)
    last_err = None
    for nc in [nconv, 32, 16]:
        try:
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                    num_conv=nc, upscale=4, act_type="prelu")
            ups = RealESRGANer(scale=4, model_path=str(path), model=model, tile=tile,
                               tile_pad=16, pre_pad=0, half=False, device=torch.device("cpu"))
            if nc != nconv: log.info("✅ Fallback num_conv=%s worked!", nc)
            _ups_cache[k] = ups; return ups
        except Exception as e:
            last_err = e
            log.warning("num_conv=%s load fail: %s", nc, str(e)[:140])
    raise last_err or RuntimeError("SRVGG load failed")

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

def colorize_frame(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "off" or mode not in DDCOLOR_MODELS: return img
    model = _load_ddcolor(mode)
    if model is None: return img
    try:
        import torchvision.transforms as T
        from PIL import Image as PILImage
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        ow, oh = pil.size
        pil_r = pil.resize((512, 512), PILImage.LANCZOS)
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

# ================= CLIENT =================
app = Client("anime_upscaler_bot", api_id=API_ID, api_hash=API_HASH,
             bot_token=BOT_TOKEN, in_memory=True)
archive = ChannelArchive(app, (os.getenv("ARCHIVE_CHANNEL_ID", "") or "").strip()) if ChannelArchive else None

# ================= PANEL (submenu-safe) =================
_panel: Optional[Message] = None
_panel_lock = asyncio.Lock()
_panel_mode = "main"     # "main" | "submenu" | "job"

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
         InlineKeyboardButton("▶️ Start", callback_data="b:go"),
         InlineKeyboardButton("📊 Stats", callback_data="b:stats")],
        [InlineKeyboardButton("🧭 Help", callback_data="b:help"),
         InlineKeyboardButton("🎥 Video", callback_data="b:sendv"),
         InlineKeyboardButton("🖼 Photo", callback_data="b:sendp")],
        [InlineKeyboardButton("🎞 GIF", callback_data="b:sendg"),
         InlineKeyboardButton("⛔ Stop", callback_data="b:stop"),
         InlineKeyboardButton("🧹 Clean", callback_data="b:clean")],
    ])

def models_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎌 Anime Video (fast)", callback_data="b:m:anime_video")],
        [InlineKeyboardButton("🎌 Anime Image (crisp)", callback_data="b:m:anime_image")],
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
        [InlineKeyboardButton("🎨 Fast (tiny)", callback_data="b:col:fast")],
        [InlineKeyboardButton("💎 High (modelscope)", callback_data="b:col:high")],
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
VALUE_BUTTONS   = {"m", "q", "p", "a", "col", "c", "back"}

HELP_TEXT = (
    "🧭 **Help (v10.4)**\n\n"
    "🎥 Video / 🎞 GIF / 🖼 Photo bhejo → upscale\n"
    "🎛 **Models** → Anime Video / Anime Image / Game / Real\n"
    "🎨 **Color** → OFF / Fast (tiny) / 💎 High (DDColor)\n"
    "⚙️ **Cores** → Auto / Fixed (1T..16T, Duo, Quad, Hexa)\n"
    "🚫 No post-process — pure AI upscale\n"
    "🛡 Rotation + stride fix (explicit transpose)\n"
    "✍️ /start /panel /reset /stats /cancel"
)

def panel_text() -> str:
    m = MODELS.get(settings["model"], MODELS["anime_video"])
    lines = [_pad(f"{EMO.face()}  UPSCALER v10.4"), "─" * PW,
             _pad(f"🧠 {CPU_THREADS}c • 🛡 {mem_avail_gb():.1f}GB free • load {load1():.1f}"),
             _pad(f"{m['label']} {fmt_scale(settings['scale'])}× "
                  f"{settings['preset'][:4]} 🔊{settings['audio'][:4]}"),
             _pad(f"{_color_label()} • ⚙️ {_core_label()}"),
             _pad(f"💡 {m['best_for'][:30]}"),
             _pad("")]
    if job_state.get("active") and current_job:
        j = current_job
        if j.get("total"):
            pct = j["done"] * 100 / j["total"]
            lines += [_pad(f"{j.get('stage', '🎨')} {j['filename'][:18]}"),
                      _pad(f"{bar(pct)} {pct:.0f}%"),
                      _pad(f"🎞 {j['done']}/{j['total']} • ETA {fmt_time(j.get('eta', 0))}"),
                      _pad(j.get("ai", "")[:PW])]
        else:
            lines += [_pad(f"{j.get('stage', '📥')} {j.get('filename', '')[:18]}")] + [_pad("")] * 2
    else:
        lines += [_pad("😴 Idle — koi job nahi"),
                  _pad("🎥 video / 🖼 photo / 🎞 gif"),
                  _pad("bhejo → pure AI upscale"),
                  _pad("Models + Color buttons")]
    lines += ["─" * PW, _pad("🎨 Color: OFF / Fast / High"),
              _pad("🚫 No post-process, pure AI")]
    return "\n".join(lines)

async def send_panel(cid: int) -> Optional[Message]:
    global _panel, _panel_mode
    async with _panel_lock:
        if _panel is not None:
            try: await _panel.delete()
            except Exception: pass
            _panel = None
        for attempt in range(1, 4):
            try:
                msg = await app.send_message(cid, panel_text(), reply_markup=panel_kb())
                _panel = msg
                _panel_mode = "main"
                log.info("✅ Panel sent (msg_id=%s, attempt=%s)", msg.id, attempt)
                return msg
            except Exception as e:
                log.warning("Panel send attempt %d fail: %s", attempt, e)
                await asyncio.sleep(2)
        log.error("❌ Panel send failed")
        return None

async def ensure_panel(cid: int) -> Optional[Message]:
    global _panel
    if _panel is not None: return _panel
    try:
        _panel = await app.send_message(cid, panel_text(), reply_markup=panel_kb())
        log.info("✅ Panel ensured (msg_id=%s)", _panel.id)
        return _panel
    except Exception as e:
        log.error("Panel ensure fail: %s", e); return None

async def refresh_panel():
    global _panel
    if _panel is None: return
    try:
        await _panel.edit_text(panel_text(), reply_markup=panel_kb())
    except FloodWait as e:
        # Spam prevention check
        await asyncio.sleep(e.value)
    except Exception as e:
        err = str(e).lower()
        if "not modified" in err: return
        if "message_id_invalid" in err or "message to edit not found" in err or "deleted" in err:
            log.warning("Panel msg lost. Will re-send.")
            _panel = None

# ================= CALLBACK =================
@app.on_callback_query(filters.regex(r"^b:"))
async def btn(client, cq):
    global _panel, _panel_mode
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
            await cq.answer(f"{MODELS[v]['label']}")
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
            await cq.answer(f"🎨 {v}")
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
    elif a == "go":
        await cq.answer("▶️")
        await cq.message.reply_text(f"▶️ Bas video/GIF/photo bhejo — {_model_label()} upscale!")
        return
    elif a == "help":
        await cq.answer(); await cq.message.reply_text(HELP_TEXT); return
    elif a == "stats":
        await cq.answer(); await cq.message.reply_text(_stats_text()); return
    elif a == "sendv":
        await cq.answer(); await cq.message.reply_text("🎥 Ab **video** bhejo!"); return
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
        await send_panel(cq.message.chat.id)
        await cq.answer("🧹 Clean"); return
    else:
        await cq.answer(); return

    if archive:
        try: asyncio.create_task(archive.save_state())
        except Exception: pass

    if kb is not None:
        try:
            if _panel and cq.message.id == _panel.id:
                await _panel.edit_text(panel_text(), reply_markup=kb)
            else:
                await cq.message.edit_text(panel_text(), reply_markup=kb)
                _panel = cq.message
        except Exception as e:
            log.warning("Callback edit fail: %s", str(e)[:120])
            try: await send_panel(cq.message.chat.id)
            except Exception: pass

def _stats_text() -> str:
    m = MODELS.get(settings["model"], MODELS["anime_video"])
    lines = [f"📊 **Stats**",
             f"⚙️ Cores: {_core_label()}",
             f"🎽 Model: {m['label']} — {m['best_for']}",
             f"🎨 Colorize: {_color_label()}",
             f"🚫 Post-process: OFF (pure AI)",
             f"🧠 Governor: adaptive",
             f"🖥 Cores: {CPU_THREADS} • 🛡 RAM: {mem_avail_gb():.1f}GB free",
             f"🎯 Scale: {fmt_scale(settings['scale'])}× • Preset: {settings['preset'].title()}"]
    if archive:
        st = archive.state
        hist = st.get("history", [])
        tot_t = sum(x.get("t", 0) for x in hist)
        lines += [f"✅ Jobs done: {st.get('jobs_done', 0)}",
                  f"🕒 Total: {fmt_time(tot_t)}"]
    return "\n".join(lines)

# ================= PROBE (rotation-aware) =================
def probe_video(path: Path) -> Dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                        "-show_streams", "-show_format", str(path)],
                       capture_output=True, text=True, check=True)
    data = json.loads(r.stdout)
    vid = next(s for s in data["streams"] if s.get("codec_type") == "video")
    fs = vid.get("avg_frame_rate") or vid.get("r_frame_rate") or "30/1"
    n, d = fs.split("/"); fps = float(n) / float(d) if float(d) else 30.0
    dur = float(vid.get("duration") or data.get("format", {}).get("duration") or 0)
    frames = int(float(vid.get("nb_frames") or max(1, dur * fps)))

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

# ================= PIPELINE (STRICT Memory Alignment + Metadata Strip) =================
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
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)

def run_pipeline(job, in_path: Path, out_path: Path, info: Dict, ups, gov: Governor,
                 cancel: threading.Event, is_gif: bool, prev_dir: Path):
    w, h, fps = info["width"], info["height"], info["fps"]
    rot = info.get("rotation", 0.0)
    ow, oh = job["ow"], job["oh"]
    ff = PRESETS[settings["preset"]]
    colorize_mode = normalize_colorize(settings["colorize_mode"])
    enc_threads = 1 if gov.current[0] >= 2 else 2
    fps_g = fps if fps > 0 else 10.0
    total = min(info["frames"] or max(1, int(info["duration"] * fps)),
                MAX_GIF_FRAMES if is_gif else 10 ** 9)
    job["total"] = total
    stats = {"done": 0, "sum": 0.0}
    stats_lock = threading.Lock()
    encoded = [0]
    dec = enc = None
    in_q: queue.Queue = queue.Queue(maxsize=IN_QUEUE)
    futs = []
    futs_cond = threading.Condition()
    t_start = time.time()

    vf_chain = _build_vf_chain(w, h, rot)
    log.info("📐 Reader: vf='%s' frame=%sx%s (%d B)", vf_chain, w, h, w * h * 3)

    def reader():
        fb = w * h * 3
        try:
            dec = subprocess.Popen(
                ["ffmpeg", "-v", "error",
                 "-noautorotate",
                 "-i", str(in_path),
                 "-vsync", "0",
                 "-vf", vf_chain,
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                stdout=subprocess.PIPE)
            n = 0
            while not cancel.is_set():
                # Strict bytes reading (Fixes shifted/garbled buffer issue)
                raw = read_exact(dec.stdout, fb)
                if not raw or len(raw) != fb:
                    break
                if is_gif and n >= MAX_GIF_FRAMES: break
                in_q.put(np.frombuffer(raw, np.uint8).reshape(h, w, 3)); n += 1
            dec.wait()
        finally:
            in_q.put(None)

    def upscale_one(img, cfg):
        t0 = time.time()
        out, _ = ups.enhance(img, outscale=settings["scale"])
        if colorize_mode != "off":
            out = colorize_frame(out, colorize_mode)
            
        # FIX 1: Enforce EXACT 3-channel structure (some ESRGAN engines output 4 channels)
        if len(out.shape) == 3 and out.shape[2] == 4:
            out = out[:, :, :3]
        elif len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            
        # FIX 2: Exact matching of required FFMPEG boundaries
        if out.shape[1] != ow or out.shape[0] != oh:
            out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
            
        # FIX 3 (CRITICAL): Absolutely guarantees the bytes are packed in C-contiguous memory!
        # If memory is disjointed, ffmpeg will slant and garble every frame in the pipe.
        out = np.ascontiguousarray(out, dtype=np.uint8)
            
        dt = time.time() - t0
        with stats_lock:
            stats["done"] += 1; stats["sum"] += dt
            job["done"] = stats["done"]
            job["spf"] = stats["sum"] / stats["done"]
            job["eta"] = (total - job["done"]) * job["spf"] / max(1, gov.current[0])
            job["ai"] = gov.status(job["spf"])
        del img
        gov.on_frame(cfg, dt, stats["done"])
        return out

    def encoder():
        nonlocal enc
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{ow}x{oh}", "-r", f"{(fps_g if is_gif else fps):.6f}", "-i", "pipe:0"]
        
        if not is_gif:
            cmd += ["-i", str(in_path), "-map", "0:v:0"]
            if info["has_audio"]:
                if settings["audio"] == "keep": cmd += ["-map", "1:a?", "-c:a", "copy"]
                elif settings["audio"] == "compress": cmd += ["-map", "1:a?", "-c:a", "aac", "-b:a", "128k"]
            
            # FIX 4 (CRITICAL for mobile videos): Strip all metadata!
            # If the original file had a corrupted rotation tag, Ffmpeg would copy it here and
            # force players to squish and box the video entirely.
            cmd += ["-map_metadata", "-1"]
            
        cmd += ["-c:v", "libx264", "-preset", ff["preset"], "-crf", ff["crf"],
                "-threads", str(enc_threads), "-pix_fmt", "yuv420p"]
                
        if not is_gif and info["has_audio"] and settings["audio"] != "remove":
            cmd += ["-shortest"]
            
        cmd += ["-movflags", "+faststart", str(out_path)]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        
        i = 0
        while True:
            with futs_cond:
                while i >= len(futs):
                    futs_cond.wait(0.2)
                    if cancel.is_set() and i >= len(futs): return
                f = futs[i]
                if f is None: break
            arr = f.result()
            enc.stdin.write(arr.tobytes())
            del arr
            with futs_cond:
                futs[i] = None; encoded[0] += 1
            i += 1
        enc.stdin.close(); enc.wait()

    th_read = threading.Thread(target=reader, daemon=True); th_read.start()
    try:
        if is_gif:
            frames_raw = []
            while True:
                img = in_q.get()
                if img is None: break
                frames_raw.append(img)
            outs = []
            for fr in frames_raw:
                if cancel.is_set(): raise RuntimeError("Cancelled")
                if time.time() - t_start > MAX_JOB_SEC: raise RuntimeError("Job time-limit")
                outs.append(upscale_one(fr, gov.current))
            frames_raw.clear(); gc.collect()
            loops = min(max(1, math.ceil(GIF_MIN_SEC / (len(outs) / fps_g))),
                        max(1, 600 // max(1, len(outs))))
            job["loops"] = loops
            job["stage"] = "📦"
            
            cmd_gif = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                       "-s", f"{ow}x{oh}", "-r", f"{fps_g:.6f}", "-i", "pipe:0",
                       "-c:v", "libx264", "-preset", ff["preset"], "-crf", ff["crf"],
                       "-pix_fmt", "yuv420p", "-map_metadata", "-1", "-movflags", "+faststart", str(out_path)]
            
            enc = subprocess.Popen(cmd_gif, stdin=subprocess.PIPE)
            for _ in range(loops):
                for arr in outs: enc.stdin.write(arr.tobytes())
            enc.stdin.close(); enc.wait()
            outs.clear(); gc.collect()
        else:
            th_enc = threading.Thread(target=encoder, daemon=True); th_enc.start()
            job["stage"] = "🎨"
            i = 0
            while True:
                if cancel.is_set():
                    with futs_cond: futs.append(None); futs_cond.notify_all()
                    raise RuntimeError("Cancelled")
                if time.time() - t_start > MAX_JOB_SEC:
                    with futs_cond: futs.append(None); futs_cond.notify_all()
                    raise RuntimeError("Job time-limit")
                img = in_q.get()
                if img is None: break
                while (i - encoded[0]) >= OUT_BACKLOG:
                    time.sleep(0.01)
                    if cancel.is_set():
                        with futs_cond: futs.append(None); futs_cond.notify_all()
                        raise RuntimeError("Cancelled")
                cfg = tuple(gov.current)
                f = POOL.submit(upscale_one, img, cfg)
                with futs_cond: futs.append(f); futs_cond.notify_all()
                i += 1
            with futs_cond: futs.append(None); futs_cond.notify_all()
            th_enc.join(timeout=7200)
        th_read.join()
        if enc is not None and enc.returncode not in (0, None):
            raise RuntimeError("FFmpeg encode failed")
        job["frames_done"] = job["done"]
        job["seconds"] = time.time() - t_start
    finally:
        for p in (dec, enc):
            try:
                if p and p.poll() is None: p.kill()
            except Exception: pass

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

# ================= LOG =================
@app.on_message(filters.all & filters.private, group=-1)
async def debug_logger(client, message: Message):
    kind = "text" if message.text else "media" if (message.video or message.photo or message.animation or message.document) else "other"
    log.info("INCOMING | chat=%s | kind=%s | %r", message.chat.id, kind, (message.text or "")[:50])

# ================= JOB =================
busy_lock = asyncio.Lock()

async def _refresh_loop():
    global _panel_mode
    hb = 0
    owner = _owner_cid()
    while True:
        try:
            if job_state.get("active"):
                _panel_mode = "job"
            elif _panel_mode == "job":
                _panel_mode = "main"

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
            if hb % 24 == 0 and job_state.get("active") and current_job:
                log.info("💓 %s fr | RAM %.1fGB | load %.1f",
                         current_job.get("done", 0), mem_avail_gb(), load1())
        except Exception as e:
            log.warning("refresh_loop err: %s", e)
        
        # Ab panel 1 second ki exact delay ke saath hi update hoga (spam limit bypass)
        await asyncio.sleep(1.0)

@app.on_message((filters.video | filters.document | filters.animation) & filters.private)
async def media_handler(client, message: Message):
    global cancel_event, current_job, _panel_mode
    if not is_owner(message.chat.id):
        await message.reply_text("❌ Private bot."); return
    async with busy_lock:
        if job_state.get("active"):
            await message.reply_text("⏳ Ek job chal rahi hai."); return
        job_state["active"] = True
        cancel_event = threading.Event()
        _panel_mode = "job"
    job_dir = None; out_path = None
    try:
        media = message.video or message.document or message.animation
        filename = getattr(media, "file_name", None) or f"video_{message.id}.mp4"
        mime = getattr(media, "mime_type", "") or ""
        is_gif = filename.lower().endswith(".gif") or mime == "image/gif"
        if not filename.lower().endswith((".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".gif")):
            await message.reply_text("❌ Sirf video/GIF bhejo."); return
        if _panel is None: await ensure_panel(message.chat.id)
        job_dir = WORK_DIR / f"job_{message.id}_{int(time.time())}"
        job_dir.mkdir(parents=True, exist_ok=True)
        in_path = job_dir / filename
        current_job = {"filename": filename, "ow": 0, "oh": 0, "done": 0, "total": 0,
                       "spf": 0.0, "eta": 0.0, "ai": "", "stage": "📥", "loops": 1}

        def dl_cb(cur, tot, *a):
            current_job["stage"] = f"📥 {cur/1048576:.0f}/{tot/1048576:.0f}MB"
        await app.download_media(message, file_name=str(in_path), progress=dl_cb)

        current_job["stage"] = "🔍"
        info = await asyncio.to_thread(probe_video, in_path)
        if not is_gif and info["frames"] > MAX_FRAMES:
            await message.reply_text(f"❌ Video lambi: {info['frames']} fr (max {MAX_FRAMES}).")
            return
        scale = settings["scale"]
        ow = int(info["width"] * scale); ow += ow % 2
        oh = int(info["height"] * scale); oh += oh % 2
        capped = False
        while scale > 1.0 and ow * oh > MAX_OUT_PIXELS:
            scale = max(1.0, scale - 0.5); capped = True
            ow = int(info["width"] * scale); ow += ow % 2
            oh = int(info["height"] * scale); oh += oh % 2
        model_key = normalize_model_key(settings["model"])
        ups = await asyncio.to_thread(get_ups, model_key, choose_tile(model_key, ow * oh))
        fixed = None
        if settings["core"] != "auto":
            cm = CORE_MAP.get(settings["core"])
            if cm and cm[2] > 0: fixed = (cm[2], cm[3])
        gov = Governor(ow * oh, fixed=fixed)
        current_job.update({"ow": ow, "oh": oh, "stage": "🎨", "ai": gov.status(0.0)})
        out_path = OUTPUT_DIR / f"{Path(filename).stem}_up_{message.id}.mp4"
        t0 = time.time()
        
        # "Process shuru" message completely REMOVED -> directly updating the panel message!
        await asyncio.to_thread(run_pipeline, current_job, in_path, out_path, info,
                                ups, gov, cancel_event, is_gif, job_dir)
        current_job["stage"] = "⬆️"
        size_mb = out_path.stat().st_size / 1048576
        if size_mb > MAX_SEND_MB:
            raise RuntimeError(f"Output {size_mb:.0f}MB > 2GB")
        cap = (f"✅ **{filename}**\n"
               f"{MODELS[model_key]['label']} • {fmt_scale(scale)}× → {ow}×{oh}\n"
               f"🎨 Colorize: {_color_label()}\n"
               f"🎞 {current_job.get('frames_done', current_job['done'])} fr"
               + (f" (loop ×{current_job.get('loops', 1)})" if is_gif else "") +
               f" • ⚡ {current_job['spf']:.2f}s/fr • 🕒 {fmt_time(time.time() - t0)}\n"
               f"📦 {size_mb:.1f}MB" + (" ⚠️ 4K-cap" if capped else ""))

        def ul_cb(cur, tot, *a):
            current_job["stage"] = f"⬆️ {cur/1048576:.0f}/{tot/1048576:.0f}MB"

        # Direct final video send
        if is_gif:
            await send_with_retry(lambda: app.send_animation(
                message.chat.id, str(out_path), caption=cap, progress=ul_cb), "animation")
        else:
            await send_with_retry(lambda: app.send_video(
                message.chat.id, str(out_path), caption=cap,
                supports_streaming=True, progress=ul_cb), "video")
        if archive:
            try:
                await archive.archive_video(out_path, f"{filename} | {MODELS[model_key]['label']} | "
                                                      f"{fmt_scale(scale)}× | {fmt_time(time.time() - t0)}")
                archive.record_job(filename, scale, time.time() - t0, True,
                                   extra={"scale": settings["scale"], "preset": settings["preset"],
                                          "audio": settings["audio"], "model": model_key,
                                          "core": settings["core"], "colorize": settings["colorize_mode"]})
                await archive.save_state()
            except Exception as e:
                log.warning("Archive fail: %s", e)
        current_job["stage"] = "✅"
    except Exception as e:
        log.exception("Job failed")
        try: await message.reply_text(f"❌ Failed: {str(e)[:250]}")
        except Exception: pass
        if current_job: current_job["stage"] = "❌"
    finally:
        if job_dir: shutil.rmtree(job_dir, ignore_errors=True)
        if out_path and out_path.exists():
            try: out_path.unlink()
            except Exception: pass
        job_state["active"] = False
        current_job = None
        cancel_event = None
        _panel_mode = "main"
        gc.collect()
        EMO.set("idle")
        try: await refresh_panel()
        except Exception: pass

# ================= PHOTO =================
@app.on_message(filters.photo & filters.private)
async def photo_handler(client, message: Message):
    if not is_owner(message.chat.id):
        await message.reply_text("❌ Private bot."); return
    async with busy_lock:
        if job_state.get("active"):
            await message.reply_text("⏳ Job chal rahi hai."); return
        job_state["active"] = True
    try:
        EMO.set("work")
        tmp = WORK_DIR / f"photo_{message.id}.jpg"
        await app.download_media(message, file_name=str(tmp))
        img = cv2.imread(str(tmp))
        if img is None: raise RuntimeError("Image read fail")
        model_key = normalize_model_key(settings["model"])
        h, w = img.shape[:2]
        ow = int(w * settings["scale"]); oh = int(h * settings["scale"])
        ups = await asyncio.to_thread(get_ups, model_key, choose_tile(model_key, ow * oh))
        t0 = time.time()
        out = await asyncio.to_thread(lambda: ups.enhance(img, outscale=settings["scale"])[0])
        if normalize_colorize(settings["colorize_mode"]) != "off":
            out = await asyncio.to_thread(colorize_frame, out, normalize_colorize(settings["colorize_mode"]))
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

@app.on_message(filters.command("cancel") & filters.private)
async def cancel_cmd(client, message: Message):
    if not is_owner(message.chat.id): return
    if cancel_event: cancel_event.set()
    await message.reply_text("🛑 Cancel bhej di.")

@app.on_message(filters.forwarded & filters.private)
async def forward_id_handler(client, message: Message):
    if not is_owner(message.chat.id): return
    src = getattr(message, "forward_from_chat", None)
    if src is not None and getattr(src, "id", None):
        await message.reply_text(f"📌 Channel ID: `{src.id}`")
    else:
        await message.reply_text("❌ Forward se ID nahi mili.")

@app.on_message(filters.text & filters.private & ~filters.command(["start", "panel", "reset", "stats", "cancel"]))
async def text_handler(client, message: Message):
    if not is_owner(message.chat.id): return
    t = (message.text or "").lower()
    if any(k in t for k in ["hi", "hello", "hey", "namaste"]):
        EMO.set("happy")
        await message.reply_text(f"{EMO.one('happy')} Namaste boss! Panel se sab control hota hai.")
        if _panel is None: await ensure_panel(message.chat.id)
    elif any(k in t for k in ["game", "free fire", "pubg", "bgmi"]):
        await message.reply_text("🎮 Models → Game Fast.")
    elif "anime" in t:
        await message.reply_text("🎌 Models → Anime Video / Anime Image.")
    elif any(k in t for k in ["color", "rang"]):
        await message.reply_text("🎨 Color button → OFF / Fast / 💎 High.")
    elif any(k in t for k in ["ram", "cpu", "load"]):
        await message.reply_text(f"🛡 RAM {mem_avail_gb():.1f}GB • {CPU_THREADS}c")
    elif any(k in t for k in ["thank", "shukriya", "thx"]):
        await message.reply_text("Apna kaam hai boss!")
    else:
        await message.reply_text("🤖 v10.4: /panel se panel; Models + Color buttons se tune.")

@app.on_message(filters.command("start") & filters.private)
async def start_handler(client, message: Message):
    if not is_owner(message.chat.id):
        await message.reply_text("❌ Private bot."); return
    EMO.set("start")
    await send_panel(message.chat.id)
    await message.reply_text(f"✅ **v10.4 online!** Chat ID: `{message.chat.id}`")

# ================= BOOT =================
def notify_owner_startup():
    try:
        r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          json={"chat_id": OWNER_CHAT_ID_INT or OWNER_CHAT_ID,
                                "text": "✅ Upscaler v10.4 online!\n🎛 Panel bhej raha hoon..."},
                          timeout=15)
        log.info("Startup ping: %s", r.status_code)
    except Exception as e:
        log.warning("Ping fail: %s", e)

async def _boot():
    try: notify_owner_startup()
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

    cid = _owner_cid()
    if cid:
        log.info("🎛 Sending panel to cid=%s", cid)
        try: await send_panel(cid)
        except Exception as e: log.error("Panel send fail: %s", e)

    asyncio.create_task(_refresh_loop())
    log.info("🚀 v10.4 ready (cores=%s)", CPU_THREADS)

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
