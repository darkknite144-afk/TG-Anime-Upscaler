#!/usr/bin/env python3
"""
Smart Anime/Game Upscaler v13 — MAX-START EDITION
- 🎛 Panel + FULL buttons BOOT par hi (video se pehle), v4 jaisa
- 🧠 Governor v4 MaxStart: start MAX workers se; RAM bhare → step-down;
  RAM khali + load kam → step-up. Koi probe-waste nahi
- ⛔ Job me sirf Cancel; complete hote hi full panel wapas
- 🎭 Quality faces + 📏 fixed-size panel + 📚 archive + 📤 retry + 🖼 photo
"""
import asyncio
import gc
import json
import logging
import math
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

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

if not API_ID or not API_HASH or not BOT_TOKEN or not OWNER_CHAT_ID:
    raise RuntimeError("Missing GitHub Secrets: API_ID, API_HASH, BOT_TOKEN, OWNER_CHAT_ID")

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

MODELS = {
    "anime":  {"file": "realesr-animevideov3.pth",  "arch": "srvgg", "label": "🎌 Anime"},
    "game":   {"file": "realesr-general-x4v3.pth",  "arch": "srvgg", "label": "🎮 GameFast"},
    "gamehq": {"file": "RealESRGAN_x4plus.pth",     "arch": "rrdb",  "label": "🎮 GameHQ"},
}
PRESETS = {"fast": {"crf": "23", "preset": "veryfast"},
           "balanced": {"crf": "19", "preset": "veryfast"},
           "best": {"crf": "16", "preset": "slow"}}

settings = {"scale": 2.0, "preset": "balanced", "audio": "keep", "model": "anime"}
job_state = {"active": False}
current_job: Optional[Dict[str, Any]] = None
cancel_event: Optional[threading.Event] = None

# ================= EMOTE ENGINE =================
FACE_TICK = 0.85
FACES = {
    "idle":     ["(˘˘)… zZ", "(¬ᴗ¬) zZ", "(˘▽˘) ♪"],
    "work":     ["(っ⚙️_⚙️)っ⚡", "(っ⚙️_⚙️)っ✦", "(っ⚙️_⚙️)っ✧"],
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
BRAILLE = "⠋⠧⠏"

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

# ================= SYSTEM =================
def mem_avail_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 8.0

def load1() -> float:
    try:
        return os.getloadavg()[0]
    except Exception:
        return 0.0

def fmt_time(s: float) -> str:
    s = max(0, int(s)); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h}h {m}m {sec}s" if h else (f"{m}m {sec}s" if m else f"{sec}s")

def fmt_scale(s: float) -> str: return f"{s:g}"

PW = 30
def _pad(s: str) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) >= PW else s + " " * (PW - len(s))

def playbar(pct: float, n: int = 10) -> str:
    f = int(n * min(100, max(0, pct)) / 100)
    cells = ["▰"] * f + ["▱"] * (n - f)
    if 0 < f < n: cells[f] = "▶"
    return "".join(cells)

# ================= GOVERNOR v4 — MAX-START, RAM-REACTIVE =================
PROFILES = [
    ("solo", 1, 4, "Solo Turbo", "1 frame × 4 threads"),
    ("duo2", 2, 2, "Duo Balanced", "2 frames × 2 threads"),
    ("duo3", 2, 3, "Duo Wide", "2 frames × 3 threads"),
    ("trio", 3, 1, "Trio Fast", "3 frames × 1 thread"),
    ("quad", 4, 1, "Quad Max", "4 frames × 1 thread"),
]
P_BY_KEY = {p[0]: p for p in PROFILES}
ORDER = ["solo", "duo2", "duo3", "trio", "quad"]   # workers 1→4

class Governor:
    """v4 MaxStart: frame-1 se MAX feasible workers; RAM bhare → step-down;
    RAM khali + load kam → step-up. Koi probe/exploration waste nahi."""
    def __init__(self, out_px: int):
        self.fp = out_px * 512 / 1e9
        self.ema = 0.0
        self.key = next((k for k in reversed(ORDER)
                         if self._ram_ok(P_BY_KEY[k][1], 0.7)), "solo")
        self.load_ema = load1(); self.ram_ema = mem_avail_gb()
        self.last_change = 0
        self.lock = threading.Lock()
        self.apply()
        log.info("🧠 Governor v4 MaxStart -> %s | fp %.2fGB/fr | RAM %.1fGB | load %.1f",
                 self.label(), self.fp, self.ram_ema, self.load_ema)

    def _ram_ok(self, workers: int, margin: float) -> bool:
        return workers * self.fp <= max(1.0, mem_avail_gb() * margin)

    def workers(self) -> int: return P_BY_KEY[self.key][1]
    def threads(self) -> int: return P_BY_KEY[self.key][2]
    def label(self) -> str:
        p = P_BY_KEY[self.key]; return f"{p[3]} • {p[4]}"
    def apply(self): torch.set_num_threads(self.threads())
    def thr(self) -> float:
        return self.workers() / self.ema if self.ema else 0.0
    def short(self, spf: float) -> str:
        return f"🤖 {P_BY_KEY[self.key][3][:6]} {spf:.2f}s {self.thr():.2f}f/s 🛡{self.ram_ema:.0f}G"

    def _step(self, d: int, why: str, done: int):
        i = ORDER.index(self.key)
        j = min(max(i + d, 0), len(ORDER) - 1)
        if j != i:
            self.key = ORDER[j]; self.last_change = done; self.apply()
            log.info("🧠 Governor %s -> %s (%s)", "⬇️" if d < 0 else "⬆️", self.label(), why)

    def on_frame(self, key: str, dt: float, done: int):
        with self.lock:
            self.ema = dt if done == 1 else self.ema * 0.7 + dt * 0.3
            self.load_ema = self.load_ema * 0.8 + load1() * 0.2
            self.ram_ema = self.ram_ema * 0.8 + mem_avail_gb() * 0.2
            if done % 20 == 0:
                gc.collect()
            # RAM/CPU pressure → turant ek step down
            if self.ram_ema < 1.5 or self.load_ema > CPU_THREADS * 1.5:
                self._step(-1, "RAM/load pressure", done)
                return
            # headroom hai aur thoda settle ho chuka hai → ek step up (max ki taraf)
            if self.ram_ema > 4.0 and self.load_ema < CPU_THREADS * 0.9 \
                    and done - self.last_change >= 30:
                i = ORDER.index(self.key)
                if i < len(ORDER) - 1:
                    nxt = ORDER[i + 1]
                    if self._ram_ok(P_BY_KEY[nxt][1], 0.6):
                        self._step(+1, "RAM headroom", done)

# ================= MODELS =================
_ups_cache: Dict[Any, RealESRGANer] = {}

def choose_tile(key: str, out_px: int) -> int:
    if MODELS[key]["arch"] == "rrdb": return 256
    return 0 if out_px <= 2_600_000 else 320

def get_ups(key: str, tile: int) -> RealESRGANer:
    k = (key, tile)
    if k in _ups_cache: return _ups_cache[k]
    m = MODELS[key]; path = MODEL_DIR / m["file"]
    if not path.exists(): raise FileNotFoundError(f"Model missing: {path}")
    log.info("Loading %s (tile=%s)...", m["file"], tile)
    if m["arch"] == "rrdb":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    else:
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
    ups = RealESRGANer(scale=4, model_path=str(path), model=model, tile=tile,
                       tile_pad=10, pre_pad=0, half=False, device=torch.device("cpu"))
    _ups_cache[k] = ups
    return ups

# ================= CLIENT + ARCHIVE =================
app = Client("anime_upscaler_bot", api_id=API_ID, api_hash=API_HASH,
             bot_token=BOT_TOKEN, in_memory=True)
archive = ChannelArchive(app, (os.getenv("ARCHIVE_CHANNEL_ID", "") or "").strip()) if ChannelArchive else None

async def react(message, emoji: str):
    try: await app.send_reaction(message.chat.id, message.id, emoji)
    except Exception: pass

# ================= PANEL =================
_panel: Optional[Message] = None

def job_active() -> bool:
    return bool(job_state.get("active") and current_job)

HELP_TEXT = (
    "🧭 **Help (v13)**\n\n"
    "🎥 Video / 🎞 GIF / 🖼 Photo bhejo → upscale\n"
    "🎛 Panel boot par hi mil jaata hai — model/scale/preset/audio/stats/help\n"
    "⛔ Job ke dauran sirf Cancel; complete hote hi full panel wapas\n"
    "🧠 AI v4 MaxStart: frame-1 se MAX workers; RAM bhare to step-down,\n"
    "   RAM khali to step-up — koi time-waste probing nahi\n"
    "🗄 Archive: settings+history pinned message me, videos channel me\n"
    "✍️ /start /stats /cancel"
)

def panel_kb(active: bool = False) -> InlineKeyboardMarkup:
    if active:
        return InlineKeyboardMarkup([[InlineKeyboardButton("⛔ Cancel Job", callback_data="b:stop")]])
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{EMO.face()}", callback_data="b:mood"),
         InlineKeyboardButton(MODELS[settings["model"]]["label"], callback_data="b:mmenu"),
         InlineKeyboardButton(f"🎯 {fmt_scale(settings['scale'])}×", callback_data="b:qmenu")],
        [InlineKeyboardButton(f"⚡ {settings['preset'].title()}", callback_data="b:pmenu"),
         InlineKeyboardButton(f"🔊 {settings['audio'].title()}", callback_data="b:amenu"),
         InlineKeyboardButton("📊 Stats", callback_data="b:stats")],
        [InlineKeyboardButton("▶️ Start", callback_data="b:go"),
         InlineKeyboardButton("🧭 Help", callback_data="b:help"),
         InlineKeyboardButton("🔄 Re-learn", callback_data="b:relearn")],
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

def panel_text() -> str:
    L = [_pad(f"{EMO.face()}  UPSCALER v13"), "─" * PW,
         _pad(f"🧠 {CPU_THREADS}c • 🛡 {mem_avail_gb():.1f}GB • load {load1():.1f}"),
         _pad(f"{MODELS[settings['model']]['label']} {fmt_scale(settings['scale'])}× "
              f"{settings['preset'][:4]} 🔊{settings['audio'][:4]}"),
         _pad("")]
    if job_active():
        j = current_job
        if j.get("total"):
            pct = j["done"] * 100 / j["total"]
            L += [_pad(f"{EMO.spin()} {j.get('stage', '🎨')} {j['filename'][:15]}"),
                  _pad(f"{playbar(pct)} {pct:.0f}%"),
                  _pad(f"🎞 {j['done']}/{j['total']} • ETA {fmt_time(j.get('eta', 0))}"),
                  _pad(j.get("ai", "")[:PW])]
        else:
            L += [_pad(f"{EMO.spin()} {j.get('stage', '📥')} {j.get('filename', '')[:15]}")] + [_pad("")] * 3
    else:
        L += [_pad("😴 Idle — koi job nahi"),
              _pad("🎥 video / 🖼 photo / 🎞 gif"),
              _pad("bhejo → MAX speed se start"),
              _pad(" settings buttons se")]
    L += ["─" * PW, _pad("⛔ job me sirf cancel dikhta"), _pad("🗄 archive channel me save")]
    return "\n".join(L)

async def ensure_panel(cid: int) -> Message:
    global _panel
    if _panel is None:
        _panel = await app.send_message(cid, panel_text(), reply_markup=panel_kb(job_active()))
    return _panel

async def refresh_panel(with_kb: bool = False):
    global _panel
    if not _panel: return
    try:
        if with_kb:
            await _panel.edit_text(panel_text(), reply_markup=panel_kb(job_active()))
        else:
            await _panel.edit_text(panel_text())
    except Exception:
        pass

async def _panel_loop():
    hb = 0
    while True:
        try:
            if job_active():
                st = current_job.get("stage", "")
                if st.startswith("📥"): EMO.set("download")
                elif st.startswith("🔍"): EMO.set("think")
                elif st.startswith("⬆️"): EMO.set("upload")
                elif st.startswith("✅"): EMO.set("happy")
                elif st.startswith("❌"): EMO.set("error")
                else: EMO.set("work")
            else:
                EMO.set("idle")
            await refresh_panel(False)
            hb += 1
            if hb % 40 == 0 and job_active():
                log.info("💓 heartbeat | %s fr | RAM %.1fGB | load %.1f",
                         current_job.get("done", 0), mem_avail_gb(), load1())
        except Exception:
            pass
        await asyncio.sleep(2.0)

def stats_text() -> str:
    if not archive: return "ℹ️ Archive channel set nahi hai."
    st = archive.state
    hist = st.get("history", [])
    tot_t = sum(x.get("t", 0) for x in hist)
    lines = [f"📊 **Stats** {EMO.one('wow')}", f"✅ Jobs: {st.get('jobs_done', 0)}",
             f"🕒 Total: {fmt_time(tot_t)}", "🧠 Policy: MaxStart (RAM-reactive)"]
    for x in hist[-3:]:
        lines.append(f"• {x.get('f', '?')[:22]} | {x.get('s', 0)}× | {fmt_time(x.get('t', 0))}")
    return "\n".join(lines)

# ================= CALLBACKS =================
@app.on_callback_query(filters.regex(r"^b:"))
async def btn(client, cq):
    global _panel
    if not is_owner(cq.message.chat.id):
        await cq.answer("Private bot!", show_alert=True); return
    parts = cq.data[2:].split(":"); a = parts[0]; v = parts[1] if len(parts) > 1 else ""
    kb = None
    if a == "mmenu": kb = model_kb(); await cq.answer("🎽 Model chuno")
    elif a == "qmenu": kb = q_kb(); await cq.answer("🎯 Scale chuno")
    elif a == "pmenu": kb = p_kb(); await cq.answer("⚡ Preset chuno")
    elif a == "amenu": kb = a_kb(); await cq.answer("🔊 Audio chuno")
    elif a == "back": kb = panel_kb(job_active()); await cq.answer("🔙")
    elif a == "m":
        settings["model"] = v
        if archive: archive.state["model"] = v
        kb = panel_kb(job_active()); await cq.answer(f"{EMO.one('wow')} {MODELS[v]['label']}")
    elif a == "q":
        settings["scale"] = float(v)
        if archive: archive.state["scale"] = float(v)
        kb = panel_kb(job_active()); await cq.answer(f"{EMO.one('start')} {v}×")
    elif a == "p":
        settings["preset"] = v
        if archive: archive.state["preset"] = v
        kb = panel_kb(job_active()); await cq.answer(f"{EMO.one('think')} {v}")
    elif a == "a":
        settings["audio"] = v
        if archive: archive.state["audio"] = v
        kb = panel_kb(job_active()); await cq.answer(f"{EMO.one('happy')} {v}")
    elif a == "go":
        await cq.answer(EMO.one("start"))
        if not job_active():
            await cq.message.reply_text(f"{EMO.one('start')} Bas video/GIF/photo bhejo — MAX speed se start!")
    elif a == "help":
        await cq.answer(EMO.one("think")); await cq.message.reply_text(HELP_TEXT)
    elif a == "stats":
        await cq.answer(EMO.one("wow")); await cq.message.reply_text(stats_text())
    elif a == "relearn":
        await cq.answer(f"{EMO.one('think')} v4 MaxStart hi policy hai — koi learning waste nahi")
    elif a == "sendv":
        await cq.answer(EMO.one("download")); await cq.message.reply_text(f"{EMO.one('download')} Ab **video** bhejo!")
    elif a == "sendp":
        await cq.answer(EMO.one("download")); await cq.message.reply_text(f"{EMO.one('download')} Ab **photo** bhejo!")
    elif a == "sendg":
        await cq.answer(EMO.one("download")); await cq.message.reply_text(f"{EMO.one('download')} Ab **GIF** bhejo!")
    elif a == "mood":
        cur = EMO.state if EMO.state in MOOD_ORDER else "idle"
        EMO.set(MOOD_ORDER[(MOOD_ORDER.index(cur) + 1) % len(MOOD_ORDER)])
        await cq.answer(EMO.face())
    elif a == "stop":
        if cancel_event: cancel_event.set()
        await cq.answer(f"{EMO.one('error')} Cancel!")
    elif a == "clean":
        try:
            if _panel: await _panel.delete()
        except Exception: pass
        _panel = None
        await ensure_panel(cq.message.chat.id)
        await cq.answer("🧹 Clean"); return
    else:
        await cq.answer(); return
    if archive:
        asyncio.create_task(archive.save_state())
    if kb is not None:
        try:
            if _panel and cq.message.id == _panel.id:
                await _panel.edit_text(panel_text(), reply_markup=kb)
            else:
                await cq.message.edit_text(panel_text(), reply_markup=kb)
                _panel = cq.message
        except Exception:
            pass

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
    frames = int(float(vid.get("nb_frames") or max(1, dur * fps)))
    return {"width": int(vid["width"]), "height": int(vid["height"]), "fps": fps,
            "duration": dur, "frames": frames,
            "has_audio": any(s.get("codec_type") == "audio" for s in data["streams"])}

# ================= PIPELINE =================
def run_pipeline(job, in_path: Path, out_path: Path, info: Dict, ups, gov: Governor,
                 cancel: threading.Event, is_gif: bool, prev_dir: Path):
    w, h, fps = info["width"], info["height"], info["fps"]
    ow, oh = job["ow"], job["oh"]
    ff = PRESETS[settings["preset"]]
    enc_threads = 1 if gov.workers() >= 2 else 2
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

    def reader():
        fb = w * h * 3
        try:
            dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(in_path), "-vsync", "0",
                                    "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                                   stdout=subprocess.PIPE)
            n = 0
            while not cancel.is_set():
                raw = dec.stdout.read(fb)
                if not raw or len(raw) != fb: break
                if is_gif and n >= MAX_GIF_FRAMES: break
                in_q.put(np.frombuffer(raw, np.uint8).reshape(h, w, 3)); n += 1
            dec.wait()
        finally:
            in_q.put(None)

    def upscale_one(img, key):
        t0 = time.time()
        out, _ = ups.enhance(img, outscale=settings["scale"])
        dt = time.time() - t0
        with stats_lock:
            first = (stats["done"] == 0)
            stats["done"] += 1; stats["sum"] += dt
            job["done"] = stats["done"]
            job["spf"] = stats["sum"] / stats["done"]
            job["eta"] = (total - job["done"]) * job["spf"] / max(1, gov.workers())
            job["ai"] = gov.short(job["spf"])
        if first:
            try: cv2.imwrite(str(prev_dir / "prev_out.png"), out)
            except Exception: pass
        del img
        gov.on_frame(key, dt, stats["done"])
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
            enc.stdin.write(arr.tobytes()); del arr
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
            if frames_raw:
                try: cv2.imwrite(str(prev_dir / "prev_in.png"), frames_raw[0])
                except Exception: pass
            outs = []
            for fr in frames_raw:
                if cancel.is_set(): raise RuntimeError("Cancelled")
                if time.time() - t_start > MAX_JOB_SEC: raise RuntimeError("Job time-limit cross")
                outs.append(upscale_one(fr, gov.key))
            frames_raw.clear(); gc.collect()
            loops = min(max(1, math.ceil(GIF_MIN_SEC / (len(outs) / fps_g))),
                        max(1, 600 // max(1, len(outs))))
            job["loops"] = loops
            job["stage"] = "📦"
            enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                    "-s", f"{ow}x{oh}", "-r", f"{fps_g:.6f}", "-i", "pipe:0",
                                    "-c:v", "libx264", "-preset", ff["preset"], "-crf", ff["crf"],
                                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
                                   stdin=subprocess.PIPE)
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
                    raise RuntimeError("Job time-limit cross")
                img = in_q.get()
                if img is None: break
                if i == 0:
                    try: cv2.imwrite(str(prev_dir / "prev_in.png"), img)
                    except Exception: pass
                while (i - encoded[0]) >= OUT_BACKLOG:
                    time.sleep(0.01)
                    if cancel.is_set():
                        with futs_cond: futs.append(None); futs_cond.notify_all()
                        raise RuntimeError("Cancelled")
                key = gov.key
                f = POOL.submit(upscale_one, img, key)
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

def make_compare(prev_dir: Path) -> Optional[Path]:
    a = cv2.imread(str(prev_dir / "prev_in.png"))
    b = cv2.imread(str(prev_dir / "prev_out.png"))
    if a is None or b is None: return None
    b = cv2.resize(b, (a.shape[1], a.shape[0]))
    sep = np.full((a.shape[0], 6, 3), 255, np.uint8)
    comp = np.hstack([a, sep, b])
    out = prev_dir / "compare.png"
    cv2.imwrite(str(out), comp)
    return out

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

# ================= JOB =================
busy_lock = asyncio.Lock()

@app.on_message((filters.video | filters.document | filters.animation) & filters.private)
async def media_handler(client, message: Message):
    global cancel_event, current_job
    if not is_owner(message.chat.id):
        await message.reply_text("❌ Private bot."); return
    async with busy_lock:
        if job_state.get("active"):
            await message.reply_text(f"{EMO.one('think')} ⏳ Ek job chal rahi hai — wait!"); return
        job_state["active"] = True
        cancel_event = threading.Event()
    job_dir = None; out_path = None
    try:
        media = message.video or message.document or message.animation
        filename = getattr(media, "file_name", None) or f"video_{message.id}.mp4"
        mime = getattr(media, "mime_type", "") or ""
        is_gif = filename.lower().endswith(".gif") or mime == "image/gif"
        if not filename.lower().endswith((".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".gif")):
            await message.reply_text(f"{EMO.one('error')} ❌ Sirf video/GIF bhejo."); return
        await react(message, "🔥")
        EMO.set("download")
        await ensure_panel(message.chat.id)
        await refresh_panel(True)
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
            await message.reply_text(f"{EMO.one('error')} ❌ Lambi video: {info['frames']} fr (max {MAX_FRAMES}).")
            return
        scale = settings["scale"]
        ow = int(info["width"] * scale); ow += ow % 2
        oh = int(info["height"] * scale); oh += oh % 2
        capped = False
        while scale > 1.0 and ow * oh > MAX_OUT_PIXELS:
            scale = max(1.0, scale - 0.5); capped = True
            ow = int(info["width"] * scale); ow += ow % 2
            oh = int(info["height"] * scale); oh += oh % 2
        model_key = settings["model"]
        ups = await asyncio.to_thread(get_ups, model_key, choose_tile(model_key, ow * oh))
        gov = Governor(ow * oh)
        current_job.update({"ow": ow, "oh": oh, "stage": "🎨", "ai": gov.short(0.0)})
        out_path = OUTPUT_DIR / f"{Path(filename).stem}_up_{message.id}.mp4"
        t0 = time.time()
        await message.reply_text(f"{EMO.one('work')} **MAX speed se shuru!** {info['width']}×{info['height']} → "
                                 f"{ow}×{oh} • {info['frames']} fr • 🧠 {gov.label()}")
        await asyncio.to_thread(run_pipeline, current_job, in_path, out_path, info,
                                ups, gov, cancel_event, is_gif, job_dir)
        current_job["stage"] = "⬆️"
        size_mb = out_path.stat().st_size / 1048576
        if size_mb > MAX_SEND_MB:
            raise RuntimeError(f"Output {size_mb:.0f}MB > 2GB limit")
        cap = (f"{EMO.one('happy')} ✅ **{filename}**\n"
               f"{MODELS[model_key]['label']} • {fmt_scale(scale)}× → {ow}×{oh}\n"
               f"🎞 {current_job.get('frames_done', current_job['done'])} fr"
               + (f" (loop ×{current_job.get('loops', 1)})" if is_gif else "") +
               f" • ⚡ {current_job['spf']:.2f}s/fr • 🕒 {fmt_time(time.time() - t0)}\n"
               f"📦 {size_mb:.1f}MB" + (" ⚠️ 4K-cap" if capped else ""))

        def ul_cb(cur, tot, *a):
            current_job["stage"] = f"⬆️ {cur/1048576:.0f}/{tot/1048576:.0f}MB"
        comp = await asyncio.to_thread(make_compare, job_dir)
        if comp:
            try: await app.send_photo(message.chat.id, str(comp), caption=f"{EMO.one('wow')} ⬅️ Before | ➡️ After")
            except Exception: pass
        if is_gif:
            await send_with_retry(lambda: app.send_animation(
                message.chat.id, str(out_path), caption=cap, progress=ul_cb), "animation")
        else:
            await send_with_retry(lambda: app.send_video(
                message.chat.id, str(out_path), caption=cap,
                supports_streaming=True, progress=ul_cb), "video")
        await react(message, "❤️")
        EMO.set("happy")
        if archive:
            await archive.archive_video(out_path, f"{filename} | {MODELS[model_key]['label']} | "
                                                  f"{fmt_scale(scale)}× | {fmt_time(time.time() - t0)}")
            archive.record_job(filename, scale, time.time() - t0, True,
                               extra={"scale": settings["scale"], "preset": settings["preset"],
                                      "audio": settings["audio"], "model": model_key})
            await archive.save_state()
        current_job["stage"] = "✅"
    except Exception as e:
        log.exception("Job failed")
        EMO.set("error")
        try: await message.reply_text(f"{EMO.one('error')} ❌ Failed: {str(e)[:220]}")
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
        gc.collect()
        await refresh_panel(True)

# ================= PHOTO =================
@app.on_message(filters.photo & filters.private)
async def photo_handler(client, message: Message):
    if not is_owner(message.chat.id):
        await message.reply_text("❌ Private bot."); return
    async with busy_lock:
        if job_state.get("active"):
            await message.reply_text(f"{EMO.one('think')} ⏳ Job chal rahi hai."); return
        job_state["active"] = True
    try:
        await react(message, "🔥")
        EMO.set("work")
        tmp = WORK_DIR / f"photo_{message.id}.jpg"
        await app.download_media(message, file_name=str(tmp))
        img = cv2.imread(str(tmp))
        if img is None: raise RuntimeError("Image read fail")
        model_key = settings["model"]
        h, w = img.shape[:2]
        ow = int(w * settings["scale"]); oh = int(h * settings["scale"])
        ups = await asyncio.to_thread(get_ups, model_key, choose_tile(model_key, ow * oh))
        t0 = time.time()
        out = await asyncio.to_thread(lambda: ups.enhance(img, outscale=settings["scale"])[0])
        dt = time.time() - t0
        outp = WORK_DIR / f"photo_{message.id}_up.png"
        cv2.imwrite(str(outp), out)
        EMO.set("happy")
        await send_with_retry(lambda: app.send_photo(
            message.chat.id, str(outp),
            caption=f"{EMO.one('happy')} ✅ Photo {MODELS[model_key]['label']} • "
                    f"{fmt_scale(settings['scale'])}× • {w}×{h} → {out.shape[1]}×{out.shape[0]} • {dt:.1f}s"), "photo")
        await react(message, "❤️")
    except Exception as e:
        log.exception("Photo fail")
        EMO.set("error")
        try: await message.reply_text(f"{EMO.one('error')} ❌ Photo fail: {str(e)[:200]}")
        except Exception: pass
    finally:
        job_state["active"] = False
        for f in WORK_DIR.glob(f"photo_{message.id}*"):
            try: f.unlink()
            except Exception: pass

# ================= COMMANDS / TEXT =================
@app.on_message(filters.command("stats") & filters.private)
async def stats_cmd(client, message: Message):
    if not is_owner(message.chat.id): return
    await message.reply_text(stats_text())

@app.on_message(filters.command("cancel") & filters.private)
async def cancel_cmd(client, message: Message):
    if not is_owner(message.chat.id): return
    if cancel_event: cancel_event.set()
    await message.reply_text(f"{EMO.one('error')} 🛑 Cancel bhej di.")

@app.on_message(filters.forwarded & filters.private)
async def forward_id_handler(client, message: Message):
    if not is_owner(message.chat.id): return
    src = getattr(message, "forward_from_chat", None)
    if src is not None and getattr(src, "id", None):
        if archive:
            archive.candidates = [src.id]
            archive.channel_id = src.id
            archive.state_msg_id = None
            asyncio.create_task(archive.save_state())
        await message.reply_text(f"{EMO.one('wow')} 📌 Channel ID set: `{src.id}`")
    else:
        await message.reply_text("❌ Channel ID nahi mili forward se.")

@app.on_message(filters.text & filters.private & ~filters.command(["start", "stats", "cancel"]))
async def text_handler(client, message: Message):
    if not is_owner(message.chat.id): return
    t = (message.text or "").lower()
    if any(k in t for k in ["hi", "hello", "hey", "namaste"]):
        await react(message, "👋")
        await message.reply_text(f"{EMO.one('happy')} Namaste boss! Panel buttons se sab control hota hai.")
    elif any(k in t for k in ["game", "free fire", "pubg", "bgmi"]):
        await message.reply_text(f"{EMO.one('wow')} 🎮 Model button → Game Fast / Game HQ.")
    elif any(k in t for k in ["ram", "cpu", "load"]):
        await message.reply_text(f"{EMO.one('think')} 🛡 RAM {mem_avail_gb():.1f}GB • load {load1():.1f} • {CPU_THREADS}c")
    elif any(k in t for k in ["thank", "shukriya", "thx"]):
        await message.reply_text(f"{EMO.one('love')} Apna kaam hai boss!")
    else:
        await message.reply_text(f"{EMO.one('think')} 🤖 v13: video/GIF/photo bhejo; MAX-start AI; buttons se settings.")

@app.on_message(filters.command("start") & filters.private)
async def start_handler(client, message: Message):
    global _panel
    if not is_owner(message.chat.id):
        await message.reply_text("❌ Private bot."); return
    await react(message, "👋")
    EMO.set("start")
    try:
        if _panel: await _panel.delete()
    except Exception: pass
    _panel = None
    await ensure_panel(message.chat.id)
    await message.reply_text(f"{EMO.one('start')} **v13 MaxStart online!** Chat ID: `{message.chat.id}`")

# ================= BOOT =================
def notify_owner_startup():
    try:
        r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          json={"chat_id": OWNER_CHAT_ID_INT or OWNER_CHAT_ID,
                                "text": "✅ Upscaler v13 online!\n🎛 Panel+buttons ABHI aa rahe hain + 🧠 MaxStart AI."},
                          timeout=15)
        log.info("Startup ping: %s", r.status_code)
    except Exception as e:
        log.warning("Ping fail: %s", e)

async def _boot():
    notify_owner_startup()
    if archive:
        await archive.load()
        st = archive.state
        settings["scale"] = float(st.get("scale", settings["scale"]))
        settings["preset"] = st.get("preset", settings["preset"])
        settings["audio"] = st.get("audio", settings["audio"])
        settings["model"] = st.get("model", settings["model"])
        log.info("📚 Archive settings: %s", settings)
        if archive.channel_id:
            try:
                m = await app.send_message(archive.channel_id, "🧪 Archive self-test...")
                await m.delete()
                log.info("✅ Archive WRITE test OK")
            except Exception as e:
                log.error("❌ Archive WRITE FAIL: %s", e)
    # 🎛 v4 jaisa: panel + FULL buttons BOOT par hi, video se pehle
    try:
        cid = OWNER_CHAT_ID_INT or int(OWNER_CHAT_ID)
        await ensure_panel(cid)
        await refresh_panel(True)
        log.info("🎛 Panel + buttons boot par bhej diye")
    except Exception as e:
        log.warning("Panel boot fail: %s", e)
    asyncio.create_task(_panel_loop())
    log.info("🚀 v13 ready (cores=%s)", CPU_THREADS)

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
