#!/usr/bin/env python3
"""
Smart Anime/Game Upscaler v6.1 — CHUNKED-TURBO + MAXSTART AI + ALL FEATURES
- 🧩 Chunked encoding: 1000+ frames safe, no RAM blow, no timeout
- ⚙️ Core allocation: Auto (MaxStart AI) / Manual (1T..16T, Duo, Quad)
- 🎌 3 Models: Anime (SRVGG) / GameFast (SRVGG) / GameHQ (RRDB)
- 🗄 Archive: Channel memory + video storage
- 🖼 Photo upscale + ⬅️➡️ Before/After preview
- 📤 Upload auto-retry (FloodWait safe)
- 🧠 MaxStart AI: Frame-1 se MAX workers, RAM-reactive step up/down
"""
import asyncio, gc, json, logging, math, os, queue, random, shutil, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import torch
from pyrogram import Client, filters, idle
from pyrogram.errors import FloodWait
from pyrogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
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
OWNER_CHAT_ID = (os.getenv("OWNER_CHAT_ID", "") or "").strip().strip("'").strip('"')
ARCHIVE_CHANNEL_ID = (os.getenv("ARCHIVE_CHANNEL_ID", "") or "").strip()

if not API_ID or not API_HASH or not BOT_TOKEN or not OWNER_CHAT_ID:
    raise RuntimeError("Missing GitHub Secrets: API_ID, API_HASH, BOT_TOKEN, OWNER_CHAT_ID")

WORK_DIR = Path("work"); OUTPUT_DIR = Path("output"); MODEL_DIR = Path("weights")
WORK_DIR.mkdir(exist_ok=True); OUTPUT_DIR.mkdir(exist_ok=True)

MAX_FRAMES = 20000            # ~11 min @30fps — chunked so safe
MAX_GIF_FRAMES = 240
MAX_OUT_PIXELS = 3840 * 2160
MAX_SEND_MB = 1900
MAX_JOBS = 3
MAX_QUEUE = 12
CLIP_SECONDS = 2.0
CHUNK_FRAMES = 300            # ~10 sec @30fps per chunk
CPU_THREADS = os.cpu_count() or 4
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)

MODELS = {
    "anime":  {"file": "realesr-animevideov3.pth", "arch": "srvgg", "label": "🎌 Anime"},
    "game":   {"file": "realesr-general-x4v3.pth", "arch": "srvgg", "label": "🎮 GameFast"},
    "gamehq": {"file": "RealESRGAN_x4plus.pth",    "arch": "rrdb",  "label": "🎮 GameHQ"},
}
PRESETS = {"fast":     {"crf": "23", "preset": "veryfast"},
           "balanced": {"crf": "19", "preset": "veryfast"},
           "best":     {"crf": "16", "preset": "slow"}}

# ================= RAM / UTILS =================
def mem_free_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 8.0

def footprint_bytes(out_px: int) -> int:
    return out_px * 64 * 4 * 2

def fmt_time(s: float) -> str:
    s = max(0, int(s)); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h}h {m}m {sec}s" if h else (f"{m}m {sec}s" if m else f"{sec}s")

def bar(pct: float, n: int = 8) -> str:
    f = int(n * min(100, max(0, pct)) / 100)
    return "▰" * f + "▱" * (n - f)

def fmt_scale(s: float) -> str: return f"{s:g}"

PW = 34
def _pad(s: str) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) >= PW else s + " " * (PW - len(s))

# ================= EMOTE ENGINE =================
FACE_TICK = 0.85
FACES = {
    "idle":     ["(˘˘)… zZ", "(¬ᴗ¬) zZ", "(˘▽˘) ♪"],
    "work":     ["(っ⚙️_⚙️)っ⚡", "(っ⚙️_⚙️)っ✦", "(っ⚙️_⚙️)っ✧"],
    "think":    ["(◔_)…", "(◔‿◔)?", "(◕_◕)…"],
    "happy":    ["(ﾉ◕)ﾉ*:･ﾟ✧", "(◕‿)✧", "(＾▽＾)ﾉ★"],
    "error":    ["(×_×;)", "(╥_╥)…", "(⊙_)!"],
    "love":     ["(♥‿♥)", "(♡ω♡)", "(⁄⁄•⁄ω•⁄⁄)"],
    "start":    ["(ò_ó)⚡", "(◉◉)✧", "(ᐛ)و"],
    "upload":   ["(⇀↼)", "(_↼)️", "(⇀‿↼)"],
    "download": ["(⇂_⇂)📥", "(⇂_)", "(⇂_⇂)✦"],
    "wow":      ["(✧ω✧)", "(✧▽✧)", "(◍◍)✨"],
}
MOOD_ORDER = ["idle", "happy", "wow", "love", "think", "work", "start"]
BRAILLE = "⠏"

class EmoteEngine:
    def __init__(self):
        self.state = "idle"; self.t0 = time.time()
    def set(self, s: str):
        if s != self.state:
            self.state = s; self.t0 = time.time()
    def face(self) -> str:
        fr = FACES.get(self.state, FACES["idle"])
        return fr[int((time.time() - self.t0) / FACE_TICK) % len(fr)]
    def one(self, s: str) -> str:
        return random.choice(FACES.get(s, FACES["idle"]))
    def spin(self) -> str:
        return BRAILLE[int(time.time() / 0.12) % len(BRAILLE)]

EMO = EmoteEngine()

# ================= CORE PROFILES =================
def build_core_profiles() -> List[Tuple[str, str, int, int]]:
    p: List[Tuple[str, str, int, int]] = [("auto", "🤖 Auto (MaxStart AI)", 0, 0)]
    for n in (1, 2, 3, 4, 6, 8, 12, 16):
        if n <= CPU_THREADS:
            p.append((f"solo{n}", f"🐢 Solo {n}T", 1, n))
    for w, t, lbl in [(2, 2, "⚡ Duo 2×2"), (2, 3, "⚡ Duo 2×3"),
                      (3, 1, "⚡ Trio 3×1"), (4, 1, "🔥 Quad 4×1"),
                      (6, 1, "🔥🔥 Hexa 6×1")]:
        if w * t <= CPU_THREADS:
            p.append((f"m{w}x{t}", lbl, w, t))
    return p

CORE_PROFILES = build_core_profiles()
CORE_MAP = {p[0]: p for p in CORE_PROFILES}

# ================= MAXSTART AI (Adaptive) =================
class FrameAI:
    """MaxStart AI: Frame-1 se MAX feasible workers; RAM bhare → step-down;
    RAM khali + load kam → step-up. Koi probe-waste nahi."""
    def __init__(self, out_px: int):
        self.fp = footprint_bytes(out_px) / 1e9
        self.ema = 0.0
        self.done_count = 0
        
        base = [(1, 4), (2, 2), (2, 3), (3, 1), (4, 1), (6, 1), (8, 1)]
        base = [(w, t) for (w, t) in base if w * t <= CPU_THREADS]
        
        free = mem_free_gb()
        self.cands = [c for c in base if c[0] * self.fp <= max(1.0, free * 0.7)]
        if not self.cands:
            self.cands = [(1, 1)]
            
        self.cands.sort(key=lambda x: x[0], reverse=True) # Max workers first
        self.current = self.cands[0]
        self.best = self.current
        
        self.ram_ema = free
        self.last_change = 0
        self.lock = threading.Lock()
        self.apply()
        log.info("🤖 AI MaxStart: %sW×%sT | free RAM %.1fGB | fp %.2fGB",
                 self.current[0], self.current[1], free, self.fp)

    def apply(self):
        torch.set_num_threads(max(1, self.current[1]))

    def throughput(self, c) -> float:
        return (c[0] / self.ema) if self.ema > 0 and c == self.current else 0.0

    def on_frame(self, cfg, dt: float):
        with self.lock:
            self.done_count += 1
            self.ema = dt if self.done_count == 1 else self.ema * 0.7 + dt * 0.3
            self.ram_ema = self.ram_ema * 0.8 + mem_free_gb() * 0.2
            
            if self.done_count % 20 == 0:
                gc.collect()
                
            if self.ram_ema < 1.5:
                self._step(-1, "RAM pressure")
                return
                
            if self.ram_ema > 4.0 and self.done_count - self.last_change >= 30:
                self._step(+1, "RAM headroom")

    def _step(self, d: int, why: str):
        try:
            idx = self.cands.index(self.current)
        except ValueError:
            idx = 0
            
        new_idx = idx - d 
        new_idx = max(0, min(new_idx, len(self.cands) - 1))
        
        if new_idx != idx:
            self.current = self.cands[new_idx]
            self.last_change = self.done_count
            self.apply()
            log.info("🤖 AI %s -> %sW×%sT (%s)", "⬆️" if d > 0 else "⬇️", 
                     self.current[0], self.current[1], why)

    def status(self, spf: float) -> str:
        return (f"🤖 {self.current[0]}W×{self.current[1]}T | {spf:.2f}s/fr | "
                f"{self.throughput(self.current):.2f} f/s | 🛡 {self.ram_ema:.1f}GB")

class FixedCore:
    def __init__(self, workers: int, threads: int):
        self.current = (workers, threads)
        self.best = self.current
        self.lock = threading.Lock()
        self.apply()
        log.info("⚙️ Manual core: %sW×%sT", workers, threads)
    def apply(self):
        torch.set_num_threads(max(1, self.current[1]))
    def on_frame(self, cfg, dt: float): pass
    def status(self, spf: float) -> str:
        return f"⚙️ {self.current[0]}W×{self.current[1]}T | {spf:.2f}s/fr"

# ================= MODELS =================
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
        if max_idx >= 2:
            return max(16, (max_idx - 2) // 2)
    except Exception as e:
        log.warning("num_conv detect fail: %s", e)
    return 16

def choose_tile(key: str, out_px: int) -> int:
    if MODELS[key]["arch"] == "rrdb":
        return 256 if out_px > 1_000_000 else 400
    if out_px <= 1_500_000:   return 0
    if out_px <= 4_000_000:   return 400
    return 320

def get_ups(key: str, tile: int) -> RealESRGANer:
    k = (key, tile)
    if k in _ups_cache: return _ups_cache[k]
    m = MODELS[key]; path = MODEL_DIR / m["file"]
    if not path.exists(): raise FileNotFoundError(f"Model missing: {path}")

    if m["arch"] == "rrdb":
        log.info("Loading %s (RRDBNet, tile=%s)...", m["file"], tile)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        ups = RealESRGANer(scale=4, model_path=str(path), model=model, tile=tile,
                           tile_pad=10, pre_pad=0, half=False, device=torch.device("cpu"))
        _ups_cache[k] = ups
        return ups

    nconv = _detect_srvgg_num_conv(path)
    log.info("Loading %s (SRVGG num_conv=%s, tile=%s)...", m["file"], nconv, tile)
    last_err = None
    for nc in [nconv, 32, 16]:
        try:
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                    num_conv=nc, upscale=4, act_type="prelu")
            ups = RealESRGANer(scale=4, model_path=str(path), model=model, tile=tile,
                               tile_pad=10, pre_pad=0, half=False, device=torch.device("cpu"))
            if nc != nconv: log.info("✅ Fallback num_conv=%s worked!", nc)
            _ups_cache[k] = ups
            return ups
        except Exception as e:
            last_err = e
            log.warning("num_conv=%s load fail: %s", nc, str(e)[:140])
    raise last_err or RuntimeError("SRVGG load failed")

# ================= PRE-FLIGHT =================
def clean_telegram_state():
    log.info("🧹 Wiping webhooks...")
    try:
        r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook?drop_pending_updates=True", timeout=10)
        log.info("Webhook: %s", r.text[:120])
        time.sleep(2)
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      json={"chat_id": OWNER_CHAT_ID,
                            "text": "✅ Upscaler v6.1 (Chunked+MaxStart+Archive) online!\n🎛 Panel boot par hi aa raha hai..."},
                      timeout=10)
    except Exception as e:
        log.error("Pre-flight: %s", e)

clean_telegram_state()

app = Client("anime_upscaler_bot", api_id=API_ID, api_hash=API_HASH,
             bot_token=BOT_TOKEN, in_memory=True)
archive = ChannelArchive(app, ARCHIVE_CHANNEL_ID) if (ChannelArchive and ARCHIVE_CHANNEL_ID) else None

# ================= SESSION =================
class Session:
    def __init__(self):
        self.scale = 2.0
        self.preset = "balanced"
        self.audio = "keep"
        self.model = "anime"
        self.core_mode = "auto"
        self.jobs: List[Dict[str, Any]] = []
        self.history: List[str] = []
        self.panel: Optional[Message] = None
        self.refresher: Optional[asyncio.Task] = None

SESSIONS: Dict[int, Session] = {}
def get_sess(cid: int) -> Session:
    if cid not in SESSIONS: SESSIONS[cid] = Session()
    return SESSIONS[cid]

def is_owner(m) -> bool:
    cid = m.chat.id if hasattr(m, "chat") else m.from_user.id
    return str(cid) == OWNER_CHAT_ID or cid == (int(OWNER_CHAT_ID) if OWNER_CHAT_ID.lstrip("-").isdigit() else 0)

STAGE_EMO = {"queued": "⏳", "dl": "⬇️", "probe": "🔍", "up": "🎨",
             "enc": "📦", "concat": "🧩", "mux": "🎬", "upload": "⬆️",
             "done": "✅", "fail": "❌"}

# ================= PANEL =================
HELP_TEXT = (
    "🧭 **Help (v6.1)**\n\n"
    "🎥 Video / 🎞 GIF / 🖼 Photo bhejo → upscale\n"
    "🎛 Panel boot par hi — Model / Scale / Preset / Audio / Cores / Stats\n"
    "⚙️ **Cores button**: Auto (MaxStart AI) ya fixed (1T..16T, Duo, Quad)\n"
    "🧠 MaxStart AI: frame-1 se MAX workers, RAM-reactive tuning\n"
    "🧩 **Chunked encoding**: 1000+ frames bhi safe\n"
    "🗄 Archive: Channel me save hota hai\n"
    "✍️ /start /stats"
)

def panel_kb(s: Session) -> InlineKeyboardMarkup:
    running = any(j["status"] not in ("done", "fail", "queued") for j in s.jobs)
    if running:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("⛔ Stop All", callback_data="b:stop")],
            [InlineKeyboardButton("🧹 Clean", callback_data="b:clean")],
        ])
    cm = CORE_MAP.get(s.core_mode, CORE_PROFILES[0])
    core_lbl = "🤖 Auto" if s.core_mode == "auto" else cm[1]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{EMO.face()}", callback_data="b:mood"),
         InlineKeyboardButton(MODELS[s.model]["label"], callback_data="b:mmenu"),
         InlineKeyboardButton(f"🎯 {fmt_scale(s.scale)}×", callback_data="b:qmenu")],
        [InlineKeyboardButton(f"⚡ {s.preset.title()}", callback_data="b:pmenu"),
         InlineKeyboardButton(f"🔊 {s.audio.title()}", callback_data="b:amenu"),
         InlineKeyboardButton(core_lbl, callback_data="b:cmenu")],
        [InlineKeyboardButton("▶️ Start", callback_data="b:go"),
         InlineKeyboardButton("📊 Stats", callback_data="b:stats"),
         InlineKeyboardButton("🧭 Help", callback_data="b:help")],
        [InlineKeyboardButton("🎥 Video", callback_data="b:sendv"),
         InlineKeyboardButton("🖼 Photo", callback_data="b:sendp"),
         InlineKeyboardButton("🎞 GIF", callback_data="b:sendg")],
        [InlineKeyboardButton("⛔ Stop", callback_data="b:stop"),
         InlineKeyboardButton("🧹 Clean", callback_data="b:clean")],
    ])

def model_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎌 Anime (fast)", callback_data="b:m:anime")],
        [InlineKeyboardButton("🎮 Game Fast (FF/PUBG)", callback_data="b:m:game")],
        [InlineKeyboardButton("🎮 Game HQ (best, slow)", callback_data="b:m:gamehq")],
        [InlineKeyboardButton("🔙 Panel", callback_data="b:back")]])

def quality_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("1.5×", callback_data="b:q:1.5"), InlineKeyboardButton("2×", callback_data="b:q:2"),
         InlineKeyboardButton("3×", callback_data="b:q:3"), InlineKeyboardButton("4×", callback_data="b:q:4")],
        [InlineKeyboardButton("🔙 Panel", callback_data="b:back")]])

def preset_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚡ Fast", callback_data="b:p:fast"),
         InlineKeyboardButton("⚖️ Balanced", callback_data="b:p:balanced"),
         InlineKeyboardButton("💎 Best", callback_data="b:p:best")],
        [InlineKeyboardButton("🔙 Panel", callback_data="b:back")]])

def audio_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔊 Keep", callback_data="b:a:keep"),
         InlineKeyboardButton("🗜 Compress", callback_data="b:a:compress"),
         InlineKeyboardButton("🔇 Remove", callback_data="b:a:remove")],
        [InlineKeyboardButton("🔙 Panel", callback_data="b:back")]])

def core_kb() -> InlineKeyboardMarkup:
    rows = []
    row = []
    for key, label, w, t in CORE_PROFILES:
        row.append(InlineKeyboardButton(label, callback_data=f"b:c:{key}"))
        if len(row) == 2:
            rows.append(row); row = []
    if row: rows.append(row)
    rows.append([InlineKeyboardButton("🔙 Panel", callback_data="b:back")])
    return InlineKeyboardMarkup(rows)

def panel_text(s: Session) -> str:
    L = [_pad(f"{EMO.face()}  UPSCALER v6.1"), "─" * PW,
         _pad(f"🧠 {CPU_THREADS}c • 🛡 {mem_free_gb():.1f}GB free"),
         _pad(f"{MODELS[s.model]['label']} {fmt_scale(s.scale)}× "
              f"{s.preset[:4]} 🔊{s.audio[:4]}")]

    cm = CORE_MAP.get(s.core_mode, CORE_PROFILES[0])
    core_line = "🤖 Auto (MaxStart)" if s.core_mode == "auto" else f"{cm[1]}"
    L.append(_pad(f"⚙️ Cores: {core_line}"))
    L.append(_pad(""))

    active = [j for j in s.jobs if j["status"] not in ("done", "fail")]
    if active:
        for j in active:
            st = j["status"]
            head = f"{STAGE_EMO.get(st, '•')} {j['filename'][:18]}"
            if st == "queued":
                L.append(_pad(head + " — queued"))
            elif st == "dl":
                L.append(_pad(head + f" {j.get('dl_done',0)/1048576:.0f}/{j.get('dl_total',0)/1048576:.0f}MB"))
            elif st == "probe":
                L.append(_pad(head + " — analyzing"))
            elif st == "up":
                pct = j["done"] * 100 / j["total"] if j["total"] else 0
                eta = (j["total"] - j["done"]) * j["spf"] / max(1, j.get("conc", 1)) if j["spf"] else 0
                L.append(_pad(head + f" {bar(pct)} {pct:.0f}%"))
                L.append(_pad(f"  {j['done']}/{j['total']} • {j['spf']:.2f}s/fr • ETA {fmt_time(eta)}"))
                L.append(_pad(f"  🧩 chunk {j.get('chunk',1)}/{j.get('chunks',1)} | {j.get('ai','')[:20]}"))
            elif st == "enc":
                L.append(_pad(head + " — 📦 encoding"))
            elif st == "concat":
                L.append(_pad(head + " — 🧩 concat chunks"))
            elif st == "mux":
                L.append(_pad(head + " — 🎬 muxing audio"))
            elif st == "upload":
                L.append(_pad(head + f" ⬆️ {j.get('ul_done',0)/1048576:.0f}/{j.get('ul_total',0)/1048576:.0f}MB"))
    else:
        L += [_pad("😴 Idle — koi job nahi"),
              _pad("🎥 video / 🖼 photo / 🎞 gif"),
              _pad("bhejo → AI MAX speed se"),
              _pad("settings buttons se")]
    if s.history:
        L += ["─" * PW, _pad("📜 " + " | ".join(s.history[-2:]))]
    L += ["─" * PW, _pad("🧩 chunked • 🛡 RAM-safe • 🔥 no-timeout")]
    return "\n".join(L)

async def ensure_panel(cid: int) -> Message:
    s = get_sess(cid)
    if s.panel is None:
        s.panel = await app.send_message(cid, panel_text(s), reply_markup=panel_kb(s))
    return s.panel

async def refresh_panel(cid: int):
    s = get_sess(cid)
    try:
        if s.panel:
            await s.panel.edit_text(panel_text(s), reply_markup=panel_kb(s))
    except Exception: pass

async def refresher_loop(cid: int):
    s = get_sess(cid)
    while any(j["status"] not in ("done", "fail") for j in s.jobs):
        st = next((j["status"] for j in s.jobs if j["status"] not in ("done", "fail")), "")
        if st == "dl": EMO.set("download")
        elif st == "up": EMO.set("work")
        elif st == "upload": EMO.set("upload")
        elif st == "probe": EMO.set("think")
        else: EMO.set("work")
        await refresh_panel(cid)
        await asyncio.sleep(2.5)
    EMO.set("idle")
    await refresh_panel(cid)
    s.refresher = None

def kick_refresher(cid: int):
    s = get_sess(cid)
    if s.refresher is None or s.refresher.done():
        s.refresher = asyncio.create_task(refresher_loop(cid))

# ================= SCHEDULER =================
POOL = ThreadPoolExecutor(max_workers=8)

def pump(cid: int):
    s = get_sess(cid)
    running = [j for j in s.jobs if j["status"] not in ("done", "fail", "queued")]
    running_long = [j for j in running if not j["clip"]]
    for j in s.jobs:
        if j["status"] != "queued": continue
        if len(running) >= MAX_JOBS: break
        if not j["clip"] and running_long: continue
        if not j["clip"]: running_long.append(j)
        running.append(j); j["status"] = "dl"
        asyncio.create_task(run_job(cid, j))
    kick_refresher(cid)

# ================= UPLOAD RETRY =================
async def send_with_retry(fn, desc: str):
    for attempt in range(1, 4):
        try:
            return await fn()
        except FloodWait as fw:
            log.warning("⏳ FloodWait %ss (%s)", fw.value, desc)
            await asyncio.sleep(fw.value + 2)
        except Exception as e:
            log.warning("📤 %s attempt %s fail: %s", desc, attempt, e)
            if attempt == 3: raise
            await asyncio.sleep(5 * attempt)

# ================= PREVIEW =================
def make_compare(job_dir: Path) -> Optional[Path]:
    a = cv2.imread(str(job_dir / "prev_in.png"))
    b = cv2.imread(str(job_dir / "prev_out.png"))
    if a is None or b is None: return None
    b = cv2.resize(b, (a.shape[1], a.shape[0]))
    sep = np.full((a.shape[0], 6, 3), 255, np.uint8)
    comp = np.hstack([a, sep, b])
    out = job_dir / "compare.png"
    cv2.imwrite(str(out), comp)
    return out

# ================= PIPELINE (CHUNKED) =================
def _compute_out_dims(w: int, h: int, want_scale: float) -> Tuple[int, int, float]:
    scale = want_scale
    ow = int(w * scale); ow += ow % 2
    oh = int(h * scale); oh += oh % 2
    while scale > 1.0 and ow * oh > MAX_OUT_PIXELS:
        scale = max(1.0, scale - 0.25)
        ow = int(w * scale); ow += ow % 2
        oh = int(h * scale); oh += oh % 2
    return ow, oh, scale

def _ffmpeg_encoder_cmd(ow: int, oh: int, fps: float, crf: str, preset: str,
                        enc_threads: int, out_path: Path) -> List[str]:
    return ["ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
            "-r", f"{fps:.6f}", "-i", "pipe:0",
            "-c:v", "libx264", "-preset", preset, "-crf", crf,
            "-threads", str(enc_threads), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out_path)]

def run_sync_job(job, in_path: Path, out_path: Path, info: Dict, s: Session):
    w, h, fps = info["width"], info["height"], info["fps"]
    ow, oh, scale = _compute_out_dims(w, h, s.scale)
    job["ow"], job["oh"] = ow, oh
    out_px = ow * oh
    ff = PRESETS.get(s.preset, PRESETS["balanced"])
    is_gif = job["is_gif"]
    fps_g = fps if fps > 0 else 10.0

    if s.core_mode == "auto":
        core = FrameAI(out_px)
    else:
        _, _, cw, ct = CORE_MAP.get(s.core_mode, CORE_PROFILES[0])
        if cw == 0: cw, ct = 2, 2
        fp = footprint_bytes(out_px) / 1e9
        max_w = max(1, int(mem_free_gb() * 0.6 / max(fp, 0.1)))
        if cw > max_w:
            log.warning("⚙️ Manual core clamped %d→%d (RAM)", cw, max_w)
            cw = max_w
        core = FixedCore(cw, ct)

    job["conc"] = core.current[0]
    ups = get_ups(s.model, choose_tile(s.model, out_px))

    total = min(info["frames"] or max(1, int(info["duration"] * fps)),
                MAX_GIF_FRAMES if is_gif else 10 ** 9)
    job["total"] = total

    cancel: threading.Event = job["cancel"]
    in_q: queue.Queue = queue.Queue(maxsize=6)
    fut_q: queue.Queue = queue.Queue(maxsize=core.current[0] + 2)
    stats = {"done": 0, "sum": 0.0}
    stats_lock = threading.Lock()
    enc_threads = max(1, CPU_THREADS - core.current[0] * core.current[1] + 1)

    chunk_files: List[Path] = []
    chunk_size = CHUNK_FRAMES if total > CHUNK_FRAMES * 1.5 else total
    num_chunks = max(1, math.ceil(total / chunk_size))
    job["chunks"] = num_chunks
    job["chunk"] = 1

    def reader():
        fb = w * h * 3
        try:
            dec = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-i", str(in_path), "-vsync", "0",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                stdout=subprocess.PIPE)
            n = 0
            while not cancel.is_set():
                raw = dec.stdout.read(fb)
                if not raw or len(raw) != fb: break
                if is_gif and n >= MAX_GIF_FRAMES: break
                in_q.put(np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()); n += 1
            try: dec.wait(timeout=5)
            except Exception: dec.kill()
        finally:
            in_q.put(None)

    def upscale_one(img, cfg):
        t0 = time.time()
        out, _ = ups.enhance(img, outscale=scale)
        dt = time.time() - t0
        core.on_frame(cfg, dt)
        with stats_lock:
            first = (stats["done"] == 0)
            stats["done"] += 1; stats["sum"] += dt
            job["done"] = stats["done"]
            job["spf"] = dt if stats["done"] == 1 else job.get("spf", dt) * 0.7 + dt * 0.3
            job["ai"] = core.status(job["spf"])
            job["conc"] = core.current[0]
        if first:
            try: cv2.imwrite(str(job["_job_dir"] / "prev_out.png"), out)
            except Exception: pass
        del img
        return out

    def encoder_chunked():
        try:
            if is_gif:
                outs = []
                while True:
                    if cancel.is_set(): raise RuntimeError("Cancelled")
                    fut = fut_q.get()
                    if fut is None: break
                    arr = fut.result()
                    outs.append(arr)
                    if len(outs) % 30 == 0: gc.collect()
                job["done"] = len(outs); job["total"] = len(outs)
                loops = min(max(1, math.ceil(CLIP_SECONDS / (max(1, len(outs)) / fps_g))),
                            max(1, 600 // max(1, len(outs))))
                job["loops"] = loops
                job["status"] = "enc"
                enc = subprocess.Popen(_ffmpeg_encoder_cmd(ow, oh, fps_g, ff["crf"], ff["preset"],
                                                            enc_threads, out_path),
                                       stdin=subprocess.PIPE)
                for _ in range(loops):
                    for arr in outs: enc.stdin.write(arr.tobytes())
                enc.stdin.close(); enc.wait()
                outs.clear(); gc.collect()
                return

            chunk_idx = 0
            in_chunk = 0
            frames_written_total = 0
            enc = subprocess.Popen(
                _ffmpeg_encoder_cmd(ow, oh, fps_g, ff["crf"], ff["preset"],
                                     enc_threads,
                                     job["_job_dir"] / f"chunk_{chunk_idx:04d}.mp4"),
                stdin=subprocess.PIPE)
            chunk_files.append(job["_job_dir"] / f"chunk_{chunk_idx:04d}.mp4")
            job["chunk"] = 1

            while True:
                if cancel.is_set():
                    try: enc.kill()
                    except Exception: pass
                    raise RuntimeError("Cancelled")
                fut = fut_q.get()
                if fut is None: break
                arr = fut.result()
                try:
                    enc.stdin.write(arr.tobytes())
                except BrokenPipeError:
                    raise RuntimeError("Encoder pipe broken")
                del arr, fut
                in_chunk += 1; frames_written_total += 1
                if (in_chunk >= chunk_size) and (frames_written_total < total):
                    try: enc.stdin.close()
                    except Exception: pass
                    enc.wait()
                    chunk_idx += 1; in_chunk = 0
                    p = job["_job_dir"] / f"chunk_{chunk_idx:04d}.mp4"
                    chunk_files.append(p)
                    enc = subprocess.Popen(
                        _ffmpeg_encoder_cmd(ow, oh, fps_g, ff["crf"], ff["preset"],
                                             enc_threads, p),
                        stdin=subprocess.PIPE)
                    job["chunk"] = chunk_idx + 1
                    gc.collect()

            try: enc.stdin.close()
            except Exception: pass
            enc.wait()
            if enc.returncode not in (0, None):
                raise RuntimeError(f"Chunk encode fail rc={enc.returncode}")
        except Exception:
            raise

    th_read = threading.Thread(target=reader, daemon=True)
    th_enc = threading.Thread(target=encoder_chunked, daemon=True)

    th_read.start(); th_enc.start()

    cfg = tuple(core.current)
    try:
        i = 0
        while True:
            if cancel.is_set(): break
            img = in_q.get()
            if img is None: break
            if i == 0:
                try: cv2.imwrite(str(job["_job_dir"] / "prev_in.png"), img)
                except Exception: pass
            while fut_q.qsize() >= core.current[0] + 2:
                time.sleep(0.02)
                if cancel.is_set(): break
            if cancel.is_set(): break
            cfg = tuple(core.current)
            f = POOL.submit(upscale_one, img, cfg)
            try:
                fut_q.put(f, timeout=30)
            except queue.Full:
                fut_q.put(f)
            i += 1
    finally:
        try: fut_q.put(None, timeout=5)
        except queue.Full: fut_q.put(None)

    th_read.join(timeout=600)
    th_enc.join(timeout=7200)

    if cancel.is_set():
        raise RuntimeError("Cancelled")

    if not is_gif and len(chunk_files) > 1:
        job["status"] = "concat"
        concat_txt = job["_job_dir"] / "concat.txt"
        concat_txt.write_text("\n".join(f"file '{p.name}'" for p in chunk_files))
        video_only = job["_job_dir"] / "video_only.mp4"
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(concat_txt), "-c", "copy",
             "-movflags", "+faststart", str(video_only)],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Concat fail: {r.stderr[:200]}")
        src_video = video_only
    else:
        src_video = chunk_files[0] if chunk_files else None

    if src_video is None:
        raise RuntimeError("No chunks produced")

    if not is_gif and info.get("has_audio") and s.audio != "remove":
        job["status"] = "mux"
        codec = "copy" if s.audio == "keep" else "aac"
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-i", str(src_video), "-i", str(in_path),
               "-map", "0:v:0", "-map", "1:a:0?",
               "-c:v", "copy", "-c:a", codec]
        if codec == "aac": cmd += ["-b:a", "128k"]
        cmd += ["-shortest", "-movflags", "+faststart", str(out_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log.warning("Audio mux fail, video-only output: %s", r.stderr[:150])
            shutil.copy(str(src_video), str(out_path))
    else:
        shutil.copy(str(src_video), str(out_path))

    for p in chunk_files:
        try: p.unlink()
        except Exception: pass

    job["frames_done"] = job["done"]
    job["ai"] = core.status(job.get("spf", 0))
    log.info("✅ Job complete: %s frames → %s", job["done"], out_path.name)

# ================= JOB RUNNER =================
async def run_job(cid: int, job):
    s = get_sess(cid)
    job_dir = None; out_path = None
    try:
        job_dir = WORK_DIR / f"job_{job['mid']}_{int(time.time())}"
        job_dir.mkdir(parents=True, exist_ok=True)
        job["_job_dir"] = job_dir
        in_path = job_dir / job["filename"]

        def dl_cb(cur, tot, *a):
            job["dl_done"], job["dl_total"] = cur, tot
        await app.download_media(job["msg"], file_name=str(in_path), progress=dl_cb)

        job["status"] = "probe"; await refresh_panel(cid)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", str(in_path)],
            capture_output=True, text=True)
        data = json.loads(probe.stdout)
        vid = next(x for x in data["streams"] if x.get("codec_type") == "video")
        fs = vid.get("avg_frame_rate") or vid.get("r_frame_rate") or "30/1"
        n, d = fs.split("/"); fps = float(n) / float(d) if float(d) else 30.0
        dur = float(vid.get("duration") or data.get("format", {}).get("duration") or 0)
        frames = int(float(vid.get("nb_frames") or max(1, dur * fps)))
        info = {"width": int(vid["width"]), "height": int(vid["height"]), "fps": fps,
                "duration": dur, "frames": frames,
                "has_audio": any(x.get("codec_type") == "audio" for x in data["streams"])}

        if not job["is_gif"] and frames > MAX_FRAMES:
            raise RuntimeError(f"Video bahut lambi: {frames} fr (max {MAX_FRAMES})")

        out_path = OUTPUT_DIR / f"{Path(job['filename']).stem}_up_{job['mid']}.mp4"
        job["status"] = "up"; job["t0"] = time.time()
        await asyncio.to_thread(run_sync_job, job, in_path, out_path, info, s)

        size_mb = out_path.stat().st_size / 1048576
        if size_mb > MAX_SEND_MB:
            raise RuntimeError(f"Output {size_mb:.0f}MB > 2GB")

        job["status"] = "upload"

        def ul_cb(cur, tot, *a):
            job["ul_done"], job["ul_total"] = cur, tot

        cap = (f"✅ **{job['filename']}**\n"
               f"{MODELS[s.model]['label']} • 🎯 {fmt_scale(s.scale)}× → {job['ow']}×{job['oh']}\n"
               f"⚡ {job.get('spf',0):.2f}s/fr\n{job.get('ai','')}\n"
               f"🎞 {job.get('frames_done',0)} fr"
               + (f" (loop ×{job.get('loops',1)})" if job["is_gif"] else "") +
               f" • 🕒 {fmt_time(time.time() - job['t0'])} • 📦 {size_mb:.1f}MB")

        comp = await asyncio.to_thread(make_compare, job_dir)
        if comp:
            try: await app.send_photo(cid, str(comp), caption=f"{EMO.one('wow')} ⬅️ Before | ➡️ After")
            except Exception: pass

        if job["is_gif"]:
            await send_with_retry(lambda: app.send_animation(cid, str(out_path), caption=cap, progress=ul_cb), "animation")
        else:
            await send_with_retry(lambda: app.send_video(cid, str(out_path), caption=cap,
                                 supports_streaming=True, progress=ul_cb), "video")
        
        if archive:
            await archive.archive_video(out_path, f"{job['filename']} | {MODELS[s.model]['label']} | {fmt_scale(s.scale)}×")
            archive.record_job(job['filename'], s.scale, time.time() - job['t0'], True,
                               extra={"scale": s.scale, "preset": s.preset, "audio": s.audio, "model": s.model})
            await archive.save_state()

        job["status"] = "done"
        s.history.append(f"✅ {job['filename'][:12]}")
    except Exception as e:
        log.exception("Job fail %s", job["filename"])
        job["status"] = "fail"
        s.history.append(f"❌ {job['filename'][:12]}")
        try:
            await app.send_message(cid, f"{EMO.one('error')} ❌ {job['filename'][:30]}: {str(e)[:200]}")
        except Exception: pass
    finally:
        if job_dir: shutil.rmtree(job_dir, ignore_errors=True)
        if out_path and out_path.exists():
            try: out_path.unlink()
            except Exception: pass
        gc.collect()
        pump(cid); kick_refresher(cid)

# ================= INTAKE =================
@app.on_message((filters.video | filters.document | filters.animation) & filters.private)
async def media_handler(client, message: Message):
    if not is_owner(message): return
    s = get_sess(message.chat.id)
    media = message.video or message.document or message.animation
    fn = getattr(media, "file_name", None) or f"video_{message.id}.mp4"
    mime = getattr(media, "mime_type", "") or ""
    is_gif = fn.lower().endswith(".gif") or mime == "image/gif"
    if not fn.lower().endswith((".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".gif")):
        await message.reply_text(f"{EMO.one('error')} ❌ Sirf video/GIF bhejo."); return
    dur = getattr(media, "duration", 0) or 0
    clip = is_gif or (0 < dur < CLIP_SECONDS)
    if len([j for j in s.jobs if j["status"] not in ("done", "fail")]) >= MAX_QUEUE:
        await message.reply_text("⚠️ Queue full (12)."); return
    s.jobs = [j for j in s.jobs if j["status"] not in ("done", "fail")][-9:] + [{
        "mid": message.id, "msg": message, "filename": fn, "is_gif": is_gif, "clip": clip,
        "status": "queued", "cancel": threading.Event(), "done": 0, "total": 0, "spf": 0.0,
        "conc": 1, "ow": 0, "oh": 0, "dl_done": 0, "dl_total": 0, "ul_done": 0, "ul_total": 0,
        "chunk": 1, "chunks": 1, "_job_dir": None}]
    EMO.set("download")
    await ensure_panel(message.chat.id); pump(message.chat.id)

# ================= PHOTO =================
@app.on_message(filters.photo & filters.private)
async def photo_handler(client, message: Message):
    if not is_owner(message): return
    s = get_sess(message.chat.id)
    if any(j["status"] not in ("done", "fail") for j in s.jobs):
        await message.reply_text(f"{EMO.one('think')} ⏳ Job chal rahi hai, photo baad me bhejo."); return
    
    try:
        EMO.set("work")
        tmp = WORK_DIR / f"photo_{message.id}.jpg"
        await app.download_media(message, file_name=str(tmp))
        img = cv2.imread(str(tmp))
        if img is None: raise RuntimeError("Image read fail")
        
        h, w = img.shape[:2]
        ow, oh, scale = _compute_out_dims(w, h, s.scale)
        ups = get_ups(s.model, choose_tile(s.model, ow * oh))
        
        t0 = time.time()
        out, _ = ups.enhance(img, outscale=scale)
        dt = time.time() - t0
        outp = WORK_DIR / f"photo_{message.id}_up.png"
        cv2.imwrite(str(outp), out)
        
        EMO.set("happy")
        await send_with_retry(lambda: app.send_photo(
            message.chat.id, str(outp),
            caption=f"{EMO.one('happy')} ✅ Photo {MODELS[s.model]['label']} • "
                    f"{fmt_scale(scale)}× • {w}×{h} → {out.shape[1]}×{out.shape[0]} • {dt:.1f}s"), "photo")
    except Exception as e:
        log.exception("Photo fail")
        EMO.set("error")
        try: await message.reply_text(f"{EMO.one('error')} ❌ Photo fail: {str(e)[:200]}")
        except Exception: pass
    finally:
        for f in WORK_DIR.glob(f"photo_{message.id}*"):
            try: f.unlink()
            except Exception: pass

# ================= BUTTONS =================
@app.on_callback_query(filters.regex(r"^b:"))
async def btn(client, cq: CallbackQuery):
    if not is_owner(cq.message):
        await cq.answer("Private bot!", show_alert=True); return
    s = get_sess(cq.message.chat.id)
    parts = cq.data[2:].split(":"); a = parts[0]; v = parts[1] if len(parts) > 1 else ""
    kb = None
    if a == "mmenu": kb = model_kb(); await cq.answer("🎽 Model chuno")
    elif a == "qmenu": kb = quality_kb(); await cq.answer("🎯 Scale chuno")
    elif a == "pmenu": kb = preset_kb(); await cq.answer("⚡ Preset chuno")
    elif a == "amenu": kb = audio_kb(); await cq.answer("🔊 Audio chuno")
    elif a == "cmenu": kb = core_kb(); await cq.answer("⚙️ Cores chuno")
    elif a == "back": kb = panel_kb(s); await cq.answer("🔙")
    elif a == "m":
        s.model = v; kb = panel_kb(s); await cq.answer(f"{EMO.one('wow')} {MODELS[v]['label']}")
    elif a == "q":
        s.scale = float(v); kb = panel_kb(s); await cq.answer(f"{EMO.one('start')} {v}×")
    elif a == "p":
        s.preset = v; kb = panel_kb(s); await cq.answer(f"{EMO.one('think')} {v}")
    elif a == "a":
        s.audio = v; kb = panel_kb(s); await cq.answer(f"{EMO.one('happy')} {v}")
    elif a == "c":
        if v in CORE_MAP:
            s.core_mode = v
            lbl = "Auto (MaxStart)" if v == "auto" else CORE_MAP[v][1]
            kb = core_kb()
            await cq.answer(f"⚙️ {lbl}")
        else:
            kb = panel_kb(s); await cq.answer()
    elif a == "go":
        await cq.answer(EMO.one("start"))
        await cq.message.reply_text(f"{EMO.one('start')} Bas video/GIF/photo bhejo — MAX speed se start!")
    elif a == "help":
        await cq.answer(EMO.one("think")); await cq.message.reply_text(HELP_TEXT)
    elif a == "stats":
        await cq.answer(EMO.one("wow")); await cq.message.reply_text(stats_text(s))
    elif a == "sendv":
        await cq.answer(EMO.one("download")); await cq.message.reply_text(f"{EMO.one('download')} Ab **video** bhejo!")
    elif a == "sendp":
        await cq.answer(EMO.one("download")); await cq.message.reply_text(f"{EMO.one('download')} Ab **photo** bhejo!")
    elif a == "sendg":
        await cq.answer(EMO.one("download")); await cq.message.reply_text(f"{EMO.one('download')} Ab **GIF** bhejo!")
    elif a == "mood":
        cur = EMO.state if EMO.state in MOOD_ORDER else "idle"
        EMO.set(MOOD_ORDER[(MOOD_ORDER.index(cur) + 1) % len(MOOD_ORDER)])
        kb = panel_kb(s); await cq.answer(EMO.face())
    elif a == "stop":
        n = sum(1 for j in s.jobs if j["status"] not in ("done", "fail", "queued") and not j["cancel"].is_set())
        for j in s.jobs:
            if j["status"] not in ("done", "fail", "queued"): j["cancel"].set()
        await cq.answer(f"⛔ {n} jobs ruki")
    elif a == "clean":
        try:
            if s.panel: await s.panel.delete()
        except Exception: pass
        s.panel = None; s.jobs = []; s.history = []
        await ensure_panel(cq.message.chat.id); await cq.answer("🧹 Clean!")
        return
    else:
        await cq.answer(); return
    
    if archive:
        asyncio.create_task(archive.save_state())
        
    if kb is not None:
        try:
            if s.panel and cq.message.id == s.panel.id:
                await s.panel.edit_text(panel_text(s), reply_markup=kb)
            else:
                await cq.message.edit_text(panel_text(s), reply_markup=kb); s.panel = cq.message
        except Exception: pass

def stats_text(s: Session) -> str:
    running = len([j for j in s.jobs if j["status"] not in ("done", "fail")])
    cm = CORE_MAP.get(s.core_mode, CORE_PROFILES[0])
    core_lbl = "Auto (MaxStart)" if s.core_mode == "auto" else cm[1]
    return (f"📊 **Stats** {EMO.one('wow')}\n"
            f"⚙️ Cores: {core_lbl}\n"
            f"🧠 AI: MaxStart (RAM-reactive)\n"
            f"🖥 CPU: {CPU_THREADS} cores • 🛡 RAM: {mem_free_gb():.1f}GB free\n"
            f"🧩 Chunk size: {CHUNK_FRAMES} frames\n"
            f"⚡ Running: {running}/{MAX_JOBS} • Queue cap: {MAX_QUEUE}\n"
            f"🎽 Model: {MODELS[s.model]['label']} • 🎯 {fmt_scale(s.scale)}×")

# ================= COMMANDS / TEXT =================
@app.on_message(filters.command("stats") & filters.private)
async def stats_cmd(client, message: Message):
    if not is_owner(message): return
    s = get_sess(message.chat.id)
    await message.reply_text(stats_text(s))

@app.on_message(filters.text & filters.private & ~filters.command(["start", "stats"]))
async def text_handler(client, message: Message):
    if not is_owner(message): return
    t = (message.text or "").lower()
    if any(k in t for k in ["hi", "hello", "hey", "namaste"]):
        EMO.set("happy")
        await message.reply_text(f"{EMO.one('happy')} Namaste boss! Panel buttons se sab control hota hai.")
    elif any(k in t for k in ["ram", "cpu", "load"]):
        await message.reply_text(f"{EMO.one('think')} 🛡 RAM {mem_free_gb():.1f}GB • Cores {CPU_THREADS}")
    else:
        await message.reply_text(f"{EMO.one('think')} 🤖 v6.1: video/GIF/photo bhejo; chunked encoding; MaxStart AI.")

@app.on_message(filters.command("start") & filters.private)
async def start_handler(client, message: Message):
    if not is_owner(message): return
    s = get_sess(message.chat.id)
    try:
        if s.panel: await s.panel.delete()
    except Exception: pass
    s.panel = None
    await ensure_panel(message.chat.id); await refresh_panel(message.chat.id)
    EMO.set("start")
    await message.reply_text(f"{EMO.one('start')} **v6.1 online!** Chat ID: `{message.chat.id}`")

# ================= BOOT =================
async def _boot():
    log.info("🚀 v6.1 boot (cores=%s)", CPU_THREADS)
    if archive:
        await archive.load()
        if archive.channel_id:
            try:
                m = await app.send_message(archive.channel_id, "🧪 Archive self-test...")
                await m.delete()
                log.info("✅ Archive WRITE test OK")
            except Exception as e:
                log.error("❌ Archive WRITE FAIL: %s", e)
                
    try: cid = int(OWNER_CHAT_ID)
    except Exception: cid = 0
    if cid:
        try:
            await ensure_panel(cid)
            await refresh_panel(cid)
            log.info("🎛 Panel + buttons boot par bhej diye")
        except Exception as e:
            log.warning("Panel boot fail: %s", e)

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
    app.run(_main())
