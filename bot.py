#!/usr/bin/env python3
"""
Telegram Anime Video Upscaler - SMART EDITION v2
Engine: Pyrogram app.run() + Webhook hard-reset + in_memory session
NEW: /specs (live system specs), /workers (2-3 videos parallel), RAM guard
"""
import asyncio
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import requests
import torch
from pyrogram import Client, filters
from pyrogram.types import (
    Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
)
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# ============================================================
# 1. CONFIGURATION (UNCHANGED - WORKING)
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("anime-upscaler")

API_ID = int(os.getenv("API_ID", "0") or 0)
API_HASH = (os.getenv("API_HASH", "")).strip()
BOT_TOKEN = (os.getenv("BOT_TOKEN", "")).strip()

_raw_owner = (os.getenv("OWNER_CHAT_ID", "") or "").strip().strip("'").strip('"')
OWNER_CHAT_ID = _raw_owner
try:
    OWNER_CHAT_ID_INT = int(_raw_owner)
except ValueError:
    OWNER_CHAT_ID_INT = 0

if not API_ID or not API_HASH or not BOT_TOKEN or not OWNER_CHAT_ID:
    raise RuntimeError("❌ Missing GitHub Secrets.")

WORK_DIR = Path("work")
OUTPUT_DIR = Path("output")
MODEL_PATH = Path("weights/realesr-animevideov3.pth")
WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_FRAMES = 3600
MAX_GIF_FRAMES = 240
MAX_OUT_PIXELS = 3840 * 2160
GIF_MIN_SEC = 2.0
MAX_SEND_MB = 1900
MAX_BATCH_SIZE = 10
MAX_WORKERS_CAP = 3
MIN_RAM_PER_JOB_MB = 1800      # RAM guard: naya job tabhi agar itni RAM free ho
CPU_THREADS = os.cpu_count() or 4
torch.set_num_threads(CPU_THREADS)
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)

# ============================================================
# 2. PRE-FLIGHT (UNCHANGED - FIXES GHOST ISSUE)
# ============================================================
def clean_telegram_state():
    log.info("🧹 Wiping old webhooks and pending updates...")
    try:
        res = requests.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook?drop_pending_updates=True",
            timeout=10
        )
        log.info("Webhook Wipe Response: %s", res.text)
        time.sleep(3)
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": OWNER_CHAT_ID,
                "text": (
                    "✅ **Smart Upscaler v2 Boot!**\n\n"
                    "🆕 /specs — live system specs\n"
                    "🆕 /workers — 1/2/3 videos parallel\n"
                    "🎬 /start se shuru karo"
                ),
                "parse_mode": "Markdown"
            },
            timeout=10
        )
    except Exception as e:
        log.error("Pre-flight failed: %s", e)

clean_telegram_state()

# ============================================================
# 3. PYROGRAM CLIENT (UNCHANGED - WORKING)
# ============================================================
app = Client(
    "anime_upscaler_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    in_memory=True
)

# ============================================================
# 4. SESSION STATE + PARALLEL WORKER POOL
# ============================================================
class UserSession:
    def __init__(self):
        self.mode: str = "idle"
        self.queue: List[Dict[str, Any]] = []
        self.running_jobs: List[Dict[str, Any]] = []
        self.current_job: Optional[Dict[str, Any]] = None
        self.workers: int = 2                     # default 2 parallel
        self.default_scale: float = 2.0
        self.default_preset: str = "balanced"
        self.default_audio: str = "keep"

sessions: Dict[int, UserSession] = {}
queue_lock = asyncio.Lock()

def get_session(uid: int) -> UserSession:
    if uid not in sessions:
        sessions[uid] = UserSession()
    return sessions[uid]

# ============================================================
# 5. SYSTEM SPECS (NEW)
# ============================================================
def mem_info_mb():
    total = avail = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) // 1024
    except Exception:
        pass
    return total, avail

def get_system_specs() -> str:
    cpu = os.cpu_count() or 0
    try:
        load = os.getloadavg()
        load_txt = f"{load[0]:.2f} / {load[1]:.2f} / {load[2]:.2f}"
    except Exception:
        load_txt = "n/a"
    total_mb, avail_mb = mem_info_mb()
    disk = shutil.disk_usage(".")
    sess = get_session(OWNER_CHAT_ID_INT or 0)
    return (
        "🖥 **System Specs (live)**\n\n"
        f"🧠 CPU cores: **{cpu}**\n"
        f"📈 Load avg (1/5/15m): {load_txt}\n"
        f"💾 RAM: **{avail_mb} MB free** / {total_mb} MB total\n"
        f"💿 Disk: **{disk.free // (1024**2)} GB free** / {disk.total // (1024**2)} GB\n"
        f"🔥 Torch: {torch.__version__} • threads/worker: **{max(1, CPU_THREADS // max(1, sess.workers))}**\n"
        f"🐍 Python: {os.sys.version.split()[0]}\n\n"
        f"⚙️ Workers: **{sess.workers}** parallel (max {MAX_WORKERS_CAP})\n"
        f"🏃 Running jobs: {len(sess.running_jobs)}\n"
        f"📚 Queue: {len(sess.queue)} videos\n"
        f"🛡 RAM guard: {MIN_RAM_PER_JOB_MB} MB/job"
    )

def apply_worker_threads(n: int):
    t = max(1, CPU_THREADS // max(1, n))
    torch.set_num_threads(t)
    log.info("🧵 Torch threads per worker set to %s (workers=%s)", t, n)

# ============================================================
# 6. HELPERS
# ============================================================
def is_owner(message) -> bool:
    cid = message.chat.id if hasattr(message, 'chat') else message.from_user.id
    return str(cid) == OWNER_CHAT_ID or cid == OWNER_CHAT_ID_INT

def fmt_scale(s: float) -> str: return f"{s:g}"

def fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

def progress_bar(pct: float, length: int = 10) -> str:
    filled = int(length * min(100, max(0, pct)) / 100)
    return "▰" * filled + "▱" * (length - filled) + f" {pct:.0f}%"

SPINNERS = ["⠋", "", "", "⠸", "⠼", "", "", "⠧", "⠇", "⠏"]
def spinner(tick: int) -> str: return SPINNERS[tick % len(SPINNERS)]

# ============================================================
# 7. KEYBOARDS
# ============================================================
def mode_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎬 Single Video", callback_data="mode:single")],
        [InlineKeyboardButton("📚 Batch Mode (Parallel)", callback_data="mode:batch")],
        [InlineKeyboardButton("ℹ️ Help / Info", callback_data="mode:help")],
    ])

def options_keyboard(filename: str, is_gif: bool) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("📐 Scale Options", callback_data="noop")],
        [InlineKeyboardButton("1.5× ⚡", callback_data="scale:1.5"),
         InlineKeyboardButton("2× ⚖️", callback_data="scale:2.0")],
        [InlineKeyboardButton("3× ✨", callback_data="scale:3.0"),
         InlineKeyboardButton("4× 💎", callback_data="scale:4.0")],
        [InlineKeyboardButton("🔊 Audio Options", callback_data="noop")],
        [InlineKeyboardButton("🔊 Keep", callback_data="audio:keep"),
         InlineKeyboardButton("🗜 Compress", callback_data="audio:compress"),
         InlineKeyboardButton("🔇 Remove", callback_data="audio:remove")],
        [InlineKeyboardButton("⚙️ Quality Preset", callback_data="noop")],
        [InlineKeyboardButton("⚡ Fast", callback_data="preset:fast"),
         InlineKeyboardButton("⚖️ Balanced", callback_data="preset:balanced"),
         InlineKeyboardButton("✨ Best", callback_data="preset:best")],
    ]
    if is_gif:
        rows.append([InlineKeyboardButton("🔁 GIF Loop Options", callback_data="noop")])
        rows.append([InlineKeyboardButton("1×", callback_data="loop:1"),
                     InlineKeyboardButton("3×", callback_data="loop:3"),
                     InlineKeyboardButton("5×", callback_data="loop:5"),
                     InlineKeyboardButton("∞", callback_data="loop:10")])
    rows.append([
        InlineKeyboardButton("✅ START UPSCALE", callback_data="action:start"),
        InlineKeyboardButton("❌ Cancel", callback_data="action:cancel_opts"),
    ])
    return InlineKeyboardMarkup(rows)

def workers_keyboard(current: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{'✅ ' if current == 1 else ''}1× (sequential)", callback_data="workers:1"),
         InlineKeyboardButton(f"{'✅ ' if current == 2 else ''}2× parallel", callback_data="workers:2"),
         InlineKeyboardButton(f"{'✅ ' if current == 3 else ''}3× parallel", callback_data="workers:3")],
    ])

def queue_keyboard(sess: UserSession) -> InlineKeyboardMarkup:
    rows = []
    for i, job in enumerate(sess.queue[:5]):
        name = job.get("filename", f"video_{i+1}")[:25]
        rows.append([InlineKeyboardButton(f"❌ Remove #{i+1}: {name}", callback_data=f"qrem:{i}")])
    if len(sess.queue) > 5:
        rows.append([InlineKeyboardButton(f"... aur {len(sess.queue)-5} videos", callback_data="noop")])
    rows.append([
        InlineKeyboardButton("🚀 PROCESS NOW", callback_data="qproc"),
        InlineKeyboardButton("🗑 Clear Queue", callback_data="qclear"),
    ])
    return InlineKeyboardMarkup(rows)

# ============================================================
# 8. LIVE STATUS (Animated, per-job label)
# ============================================================
class LiveStatus:
    STAGES = [
        ("📥", "Downloading"), ("🔍", "Analyzing video"), ("🎨", "Upscaling frames"),
        ("🔊", "Processing audio"), ("📦", "Packaging"), ("📤", "Uploading to Telegram"),
    ]
    def __init__(self, msg, loop, label: str = ""):
        self.msg, self.loop, self.last, self.last_text, self.tick = msg, loop, 0.0, "", 0
        self.label = label
    def _format(self, stage_idx, extra, pct=0):
        emoji, name = self.STAGES[min(stage_idx, len(self.STAGES)-1)]
        parts = [f"{spinner(self.tick)} **{emoji} {name}**" + (f" — {self.label}" if self.label else "")]
        self.tick += 1
        if pct > 0: parts.append(progress_bar(pct))
        if extra: parts.append(extra)
        parts.append("\n❌ /cancel to stop all")
        return "\n".join(parts)
    def update(self, stage_idx, extra="", pct=0, force=False):
        text = self._format(stage_idx, extra, pct)
        def _do(): asyncio.create_task(self._edit(text, force))
        self.loop.call_soon_threadsafe(_do)
    async def _edit(self, text, force):
        now = time.time()
        if not force and (now - self.last < 3 or text == self.last_text): return
        self.last, self.last_text = now, text
        try: await self.msg.edit_text(text, parse_mode="Markdown")
        except Exception: pass
    async def set_text(self, text):
        try: await self.msg.edit_text(text, parse_mode="Markdown")
        except Exception: pass

job_state = {"active": False, "text": "😴 Idle"}

# ============================================================
# 9. COMMANDS
# ============================================================
@app.on_message(filters.command("start") & filters.private)
async def start_handler(client, message: Message):
    if not is_owner(message): return
    sess = get_session(message.chat.id)
    sess.mode = "idle"
    await message.reply_text(
        "🎬 **Anime Video Upscaler — Smart v2**\n\n"
        "🎬 **Single Video** — options dialog ke saath\n"
        "📚 **Batch Mode** — videos bhejo, **ek saath 2-3 process** hongi\n\n"
        "🆕 /specs — live system specs (CPU/RAM/disk)\n"
        "🆕 /workers — parallel workers chuno (1/2/3)\n"
        "/queue /cancel /reset /settings /help\n\n"
        f"⚙️ Abhi workers: **{sess.workers}** parallel",
        reply_markup=mode_keyboard()
    )

@app.on_message(filters.command("specs") & filters.private)
async def specs_handler(client, message: Message):
    if not is_owner(message): return
    await message.reply_text(get_system_specs())

@app.on_message(filters.command("workers") & filters.private)
async def workers_handler(client, message: Message):
    if not is_owner(message): return
    sess = get_session(message.chat.id)
    await message.reply_text(
        f"⚙️ **Parallel workers chuno** (abhi: {sess.workers})\n\n"
        f"🧠 CPU cores: {CPU_THREADS} • RAM free: {mem_info_mb()[1]} MB\n\n"
        "💡 4-core runner par **2× best** hai; 3× chhoti videos ke liye.\n"
        "🛡 RAM kam ho to extra job khud ruk jayegi.",
        reply_markup=workers_keyboard(sess.workers)
    )

@app.on_callback_query(filters.regex(r"^workers:"))
async def workers_callback(client, cq: CallbackQuery):
    if not is_owner(cq.message): return
    sess = get_session(cq.message.chat.id)
    n = int(cq.data.split(":")[1])
    sess.workers = min(max(1, n), MAX_WORKERS_CAP)
    apply_worker_threads(sess.workers)
    await cq.answer(f"Workers: {sess.workers} parallel")
    await cq.message.edit_text(
        f"✅ Workers set: **{sess.workers} parallel**\n"
        f"🧵 Threads per worker: {max(1, CPU_THREADS // sess.workers)}\n\n"
        "Queue ab isi hisaab se chalegi.",
        reply_markup=workers_keyboard(sess.workers)
    )
    asyncio.create_task(pump_queue(cq.message.chat.id))

@app.on_message(filters.command("help") & filters.private)
async def help_handler(client, message: Message):
    if not is_owner(message): return
    await message.reply_text(
        "📚 **Commands**\n\n"
        "/start — mode chuno\n/specs — live system specs\n"
        "/workers — 1/2/3 parallel workers\n/queue — queue + running jobs\n"
        "/settings — defaults\n/cancel — sab running jobs roko\n/reset — clear\n/ping — status\n\n"
        "**Formats:** MP4/MKV/MOV/WEBM/AVI/M4V + GIF (loop video)\n"
        "**Options per video:** scale 1.5-4×, preset Fast/Balanced/Best, audio keep/compress/remove, GIF loops\n"
        "**Parallel:** batch mode me 2-3 videos ek saath, RAM guard ke saath"
    )

@app.on_message(filters.command("settings") & filters.private)
async def settings_handler(client, message: Message):
    if not is_owner(message): return
    sess = get_session(message.chat.id)
    await message.reply_text(
        "⚙️ **Defaults**\n\n"
        f"🎯 Scale: **{fmt_scale(sess.default_scale)}×**\n"
        f"⚡ Preset: **{sess.default_preset.title()}**\n"
        f"🔊 Audio: **{sess.default_audio.Title() if False else sess.default_audio.title()}**\n"
        f"⚙️ Workers: **{sess.workers}**\n",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🎯 Scale", callback_data="def:scale"),
             InlineKeyboardButton("⚡ Preset", callback_data="def:preset"),
             InlineKeyboardButton("🔊 Audio", callback_data="def:audio")],
            [InlineKeyboardButton("🔄 Reset defaults", callback_data="def:reset")],
        ])
    )

@app.on_message(filters.command("queue") & filters.private)
async def queue_handler(client, message: Message):
    if not is_owner(message): return
    sess = get_session(message.chat.id)
    lines = ["📚 **Queue & Running**\n"]
    if sess.running_jobs:
        lines.append("🏃 **Running:**")
        for j in sess.running_jobs:
            lines.append(f"   • {j.get('filename','?')[:30]}")
    if sess.queue:
        lines.append("\n⏳ **Pending:**")
        for i, j in enumerate(sess.queue, 1):
            lines.append(f"{i}. {j.get('filename','?')[:30]} — {fmt_scale(j.get('scale',2.0))}×")
    if not sess.running_jobs and not sess.queue:
        lines.append("(khaali)")
    lines.append(f"\n⚙️ Workers: {sess.workers} • /workers se badlo")
    await message.reply_text("\n".join(lines), reply_markup=queue_keyboard(sess))

@app.on_message(filters.command("cancel") & filters.private)
async def cancel_handler(client, message: Message):
    if not is_owner(message): return
    sess = get_session(message.chat.id)
    n = 0
    for j in sess.running_jobs:
        ev = j.get("cancel")
        if ev: ev.set(); n += 1
    await message.reply_text(f"🛑 {n} running job(s) ko cancel kiya." if n else "😴 Koi job nahi chal rahi.")

@app.on_message(filters.command("reset") & filters.private)
async def reset_handler(client, message: Message):
    if not is_owner(message): return
    sess = get_session(message.chat.id)
    for j in sess.running_jobs:
        ev = j.get("cancel")
        if ev: ev.set()
    sess.queue.clear()
    sess.current_job = None
    sess.mode = "idle"
    await message.reply_text("🔄 Sab clear! /start se shuru karo.")

@app.on_message(filters.command("ping") & filters.private)
async def ping_handler(client, message: Message):
    if not is_owner(message): return
    sess = get_session(message.chat.id)
    await message.reply_text(
        f"🏓 **Pong!**\n🏃 Running: {len(sess.running_jobs)} • ⏳ Queue: {len(sess.queue)}\n"
        f"⚙️ Workers: {sess.workers} • Mode: {sess.mode}"
    )

# ============================================================
# 10. MODE CALLBACKS
# ============================================================
@app.on_callback_query(filters.regex(r"^mode:"))
async def mode_callback(client, cq: CallbackQuery):
    if not is_owner(cq.message):
        await cq.answer("Private bot!", show_alert=True); return
    sess = get_session(cq.message.chat.id)
    action = cq.data.split(":", 1)[1]
    if action == "single":
        sess.mode = "single"
        await cq.answer("Single Video Mode")
        await cq.message.edit_text(
            "🎬 **Single Mode** — video/GIF bhejo, options dialog aayega.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="mode:back")]])
        )
    elif action == "batch":
        sess.mode = "batch"
        await cq.answer("Batch Mode (parallel)")
        await cq.message.edit_text(
            f"📚 **Batch Mode** — 10 tak videos bhejo.\n"
            f"⚙️ **{sess.workers} videos ek saath** process hongi (RAM guard ke saath).\n"
            "/queue se manage karo.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="mode:back")]])
        )
    elif action == "help":
        await cq.answer()
        await cq.message.edit_text(
            "ℹ️ Real-ESRGAN AnimeVideo-v3 • CPU • 2GB tak file • 6h run\n"
            "Single = options dialog • Batch = parallel queue",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="mode:back")]])
        )
    elif action == "back":
        sess.mode = "idle"
        await cq.message.edit_text("🎬 **Smart Upscaler v2** — mode chuno:", reply_markup=mode_keyboard())

# ============================================================
# 11. SETTINGS CALLBACKS
# ============================================================
@app.on_callback_query(filters.regex(r"^def:"))
async def def_callback(client, cq: CallbackQuery):
    if not is_owner(cq.message): return
    sess = get_session(cq.message.chat.id)
    a = cq.data.split(":", 1)[1]
    if a == "scale":
        await cq.message.edit_text("🎯 Default scale:", reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("1.5×", callback_data="dscale:1.5"),
             InlineKeyboardButton("2×", callback_data="dscale:2.0"),
             InlineKeyboardButton("3×", callback_data="dscale:3.0"),
             InlineKeyboardButton("4×", callback_data="dscale:4.0")],
            [InlineKeyboardButton("🔙", callback_data="def:back")]]))
    elif a == "preset":
        await cq.message.edit_text("⚡ Default preset:", reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("⚡ Fast", callback_data="dpreset:fast"),
             InlineKeyboardButton("⚖️ Balanced", callback_data="dpreset:balanced"),
             InlineKeyboardButton("✨ Best", callback_data="dpreset:best")],
            [InlineKeyboardButton("🔙", callback_data="def:back")]]))
    elif a == "audio":
        await cq.message.edit_text("🔊 Default audio:", reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔊 Keep", callback_data="daudio:keep"),
             InlineKeyboardButton("🗜 Compress", callback_data="daudio:compress"),
             InlineKeyboardButton("🔇 Remove", callback_data="daudio:remove")],
            [InlineKeyboardButton("🔙", callback_data="def:back")]]))
    elif a == "reset":
        sess.default_scale, sess.default_preset, sess.default_audio = 2.0, "balanced", "keep"
        await cq.answer("Defaults reset!")
        await settings_handler(client, cq.message)
    elif a == "back":
        await settings_handler(client, cq.message)

@app.on_callback_query(filters.regex(r"^dscale:"))
async def dscale_cb(client, cq: CallbackQuery):
    if not is_owner(cq.message): return
    get_session(cq.message.chat.id).default_scale = float(cq.data.split(":")[1])
    await cq.answer("Saved!"); await settings_handler(client, cq.message)

@app.on_callback_query(filters.regex(r"^dpreset:"))
async def dpreset_cb(client, cq: CallbackQuery):
    if not is_owner(cq.message): return
    get_session(cq.message.chat.id).default_preset = cq.data.split(":")[1]
    await cq.answer("Saved!"); await settings_handler(client, cq.message)

@app.on_callback_query(filters.regex(r"^daudio:"))
async def daudio_cb(client, cq: CallbackQuery):
    if not is_owner(cq.message): return
    get_session(cq.message.chat.id).default_audio = cq.data.split(":")[1]
    await cq.answer("Saved!"); await settings_handler(client, cq.message)

# ============================================================
# 12. MEDIA INTAKE
# ============================================================
@app.on_message((filters.video | filters.document | filters.animation) & filters.private)
async def media_handler(client, message: Message):
    if not is_owner(message): return
    sess = get_session(message.chat.id)
    if sess.mode == "idle":
        await message.reply_text("⚠️ Pehle /start se mode chuno!", reply_markup=mode_keyboard())
        return
    media = message.video or message.document or message.animation
    filename = getattr(media, "file_name", None) or f"video_{message.id}.mp4"
    is_gif = filename.lower().endswith(".gif") or getattr(media, "mime_type", "") == "image/gif"
    if not filename.lower().endswith((".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".gif")):
        await message.reply_text("❌ Unsupported format.\nMP4/MKV/MOV/WEBM/AVI/M4V/GIF")
        return
    job = {
        "message_id": message.id, "chat_id": message.chat.id, "filename": filename,
        "is_gif": is_gif, "scale": sess.default_scale, "preset": sess.default_preset,
        "audio": sess.default_audio, "loop_count": 1, "message": message,
        "cancel": threading.Event(),
    }
    if sess.mode == "single":
        sess.current_job = job
        await message.reply_text(
            f"🎬 **{filename}**\n\n{'🎞 GIF — loop video banega' if is_gif else '🎥 Video detected'}\n\n"
            "Options chuno, phir START:",
            reply_markup=options_keyboard(filename, is_gif)
        )
    else:
        if len(sess.queue) + len(sess.running_jobs) >= MAX_BATCH_SIZE:
            await message.reply_text(f"⚠️ Max {MAX_BATCH_SIZE} jobs ek saath.")
            return
        sess.queue.append(job)
        pos = len(sess.queue)
        await message.reply_text(
            f"✅ Queue me add (#{pos}): **{filename}**\n"
            f"⚙️ Workers: {sess.workers} — slot khulte hi start hogi.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🚀 PROCESS NOW", callback_data="qproc"),
                 InlineKeyboardButton("📋 Queue", callback_data="qview")]
            ])
        )
        asyncio.create_task(pump_queue(message.chat.id))

# ============================================================
# 13. OPTIONS + QUEUE CALLBACKS
# ============================================================
@app.on_callback_query(filters.regex(r"^(scale|audio|preset|loop|action|noop|qrem|qproc|qclear|qview):"))
async def options_callback(client, cq: CallbackQuery):
    if not is_owner(cq.message):
        await cq.answer("Private bot!", show_alert=True); return
    sess = get_session(cq.message.chat.id)
    parts = cq.data.split(":", 1)
    action, value = parts[0], (parts[1] if len(parts) > 1 else "")

    if action == "qrem":
        i = int(value)
        if 0 <= i < len(sess.queue):
            r = sess.queue.pop(i)
            await cq.answer(f"Removed: {r['filename'][:25]}")
            if sess.queue: await queue_handler(client, cq.message)
            else: await cq.message.edit_text("📭 Queue khaali!")
        return
    if action == "qclear":
        sess.queue.clear(); await cq.answer("Cleared!")
        await cq.message.edit_text("📭 Queue khaali!"); return
    if action == "qview":
        await queue_handler(client, cq.message); await cq.answer(); return
    if action == "qproc":
        await cq.answer("Pumping queue...")
        asyncio.create_task(pump_queue(cq.message.chat.id)); return
    if action == "noop":
        await cq.answer(); return

    if not sess.current_job:
        await cq.answer("Koi video select nahi!", show_alert=True); return
    job = sess.current_job
    if action == "scale": job["scale"] = float(value); await cq.answer(f"Scale {fmt_scale(job['scale'])}×")
    elif action == "audio": job["audio"] = value; await cq.answer(f"Audio {value}")
    elif action == "preset": job["preset"] = value; await cq.answer(f"Preset {value}")
    elif action == "loop": job["loop_count"] = int(value); await cq.answer(f"Loop {value}×")
    elif action == "cancel_opts":
        sess.current_job = None
        await cq.message.edit_text("❌ Cancelled."); return
    elif action == "start":
        sess.current_job = None
        sess.queue.insert(0, job)
        await cq.message.edit_text("🚀 Job shuru ho rahi hai...")
        asyncio.create_task(pump_queue(cq.message.chat.id))
        return
    await cq.message.edit_text(
        f"🎬 **{job['filename']}**\n\n"
        f"🎯 Scale: **{fmt_scale(job['scale'])}×**\n⚡ Preset: **{job['preset'].title()}**\n"
        f"🔊 Audio: **{job['audio'].title()}**\n"
        + (f"🔁 Loop: **{job['loop_count']}×**\n" if job["is_gif"] else "") +
        "\nSTART dabao jab ready ho:",
        reply_markup=options_keyboard(job["filename"], job["is_gif"])
    )

# ============================================================
# 14. PARALLEL SCHEDULER (NEW)
# ============================================================
async def pump_queue(chat_id: int):
    sess = get_session(chat_id)
    async with queue_lock:
        while sess.queue and len(sess.running_jobs) < sess.workers:
            free_mb = mem_info_mb()[1]
            if free_mb < MIN_RAM_PER_JOB_MB:
                log.warning("🛡 RAM kam (%s MB) — naya job hold", free_mb)
                break
            job = sess.queue.pop(0)
            task = asyncio.create_task(run_job_slot(chat_id, job))
            job["task"] = task
            sess.running_jobs.append(job)
            log.info("🏃 Job start: %s (running=%s/%s)", job["filename"], len(sess.running_jobs), sess.workers)

async def run_job_slot(chat_id: int, job: Dict[str, Any]):
    try:
        await run_job_worker(chat_id, job)
    finally:
        sess = get_session(chat_id)
        if job in sess.running_jobs:
            sess.running_jobs.remove(job)
        asyncio.create_task(pump_queue(chat_id))

# ============================================================
# 15. PIPELINE (core unchanged)
# ============================================================
_models = {}
def get_upsampler(tile: int) -> RealESRGANer:
    if tile not in _models:
        log.info("Loading Real-ESRGAN tile=%s", tile)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
        _models[tile] = RealESRGANer(scale=4, model_path=str(MODEL_PATH), model=model,
                                     tile=tile, tile_pad=10, pre_pad=0, half=False, device=torch.device("cpu"))
    return _models[tile]

def probe_video(path: Path) -> Dict:
    probe = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                            "-show_streams", "-show_format", str(path)], capture_output=True, text=True)
    data = json.loads(probe.stdout)
    vid = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if not vid: raise RuntimeError("No video stream")
    fs = vid.get("avg_frame_rate") or vid.get("r_frame_rate") or "30/1"
    n, d = fs.split("/")
    fps = float(n) / float(d) if float(d) else 30.0
    dur = float(vid.get("duration") or data.get("format", {}).get("duration") or 0)
    frames = int(float(vid.get("nb_frames") or max(1, dur * fps)))
    return {"width": int(vid["width"]), "height": int(vid["height"]), "fps": fps,
            "duration": dur, "frames": frames,
            "has_audio": any(s.get("codec_type") == "audio" for s in data["streams"]),
            "codec": vid.get("codec_name", "unknown")}

def run_pipeline_sync(input_path, output_path, info, opts, status, cancel, is_gif):
    w, h, fps = info["width"], info["height"], info["fps"]
    scale = opts["scale"]; preset_name = opts.get("preset", "balanced")
    audio_opt = opts.get("audio", "keep"); loop_count = opts.get("loop_count", 1)
    preset_map = {"fast": {"crf": "23", "preset": "veryfast", "tile": 320},
                  "balanced": {"crf": "19", "preset": "veryfast", "tile": 0},
                  "best": {"crf": "16", "preset": "slow", "tile": 0}}
    ff = preset_map.get(preset_name, preset_map["balanced"])
    ow, oh = w * scale, h * scale
    while scale > 1.0 and (int(ow + ow % 2) * int(oh + oh % 2)) > MAX_OUT_PIXELS:
        scale = max(1.0, scale - 0.5); ow, oh = w * scale, h * scale
    ow, oh = int(ow + (ow % 2)), int(oh + (oh % 2))
    tile = ff["tile"] if ff["tile"] > 0 else (0 if (ow * oh) <= 2_600_000 else 320)
    ups = get_upsampler(tile)
    t_start = time.time(); dec = enc = None
    stats = {"frames": 0, "avg_spf": 0.0, "seconds": 0.0, "out_w": ow, "out_h": oh, "loops": 1, "scale": scale}
    try:
        if not is_gif:
            total = info["frames"] or max(1, int(info["duration"] * fps))
            dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(input_path), "-vsync", "0",
                                    "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE)
            cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{ow}x{oh}", "-r", f"{fps:.6f}", "-i", "pipe:0", "-i", str(input_path), "-map", "0:v:0"]
            if info["has_audio"]:
                if audio_opt == "keep": cmd += ["-map", "1:a?", "-c:a", "copy"]
                elif audio_opt == "compress": cmd += ["-map", "1:a?", "-c:a", "aac", "-b:a", "128k"]
            cmd += ["-c:v", "libx264", "-preset", ff["preset"], "-crf", ff["crf"], "-pix_fmt", "yuv420p"]
            if info["has_audio"] and audio_opt != "remove": cmd += ["-shortest"]
            cmd += ["-movflags", "+faststart", str(output_path)]
            enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            done, sum_t, fb = 0, 0.0, w * h * 3
            while True:
                if cancel.is_set(): raise Exception("Cancelled by user")
                raw = dec.stdout.read(fb)
                if not raw or len(raw) != fb: break
                t0 = time.time()
                img = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
                out, _ = ups.enhance(img, outscale=scale)
                enc.stdin.write(out.tobytes())
                sum_t += time.time() - t0; done += 1
                spf = sum_t / done
                status.update(2, (f"🎞 Frame: **{done}/{total}**\n⚡ Per-frame: **{spf:.2f}s**\n"
                                  f"⏱ ETA: **{fmt_time((total - done) * spf)}**\n"
                                  f"🕒 Elapsed: {fmt_time(time.time() - t_start)}\n"
                                  f"🖼 {ow}×{oh} • {preset_name}"), done * 100 / total if total else 0)
                if done % 60 == 0: gc.collect()
            enc.stdin.close(); dec.wait(); enc.wait()
            stats.update(frames=done, avg_spf=(sum_t / done if done else 0.0))
        else:
            fps_g = fps if fps > 0 else 10.0
            dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(input_path), "-vsync", "0",
                                    "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE)
            frames_raw = []
            while len(frames_raw) < MAX_GIF_FRAMES:
                raw = dec.stdout.read(w * h * 3)
                if not raw or len(raw) != w * h * 3: break
                frames_raw.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy())
            dec.wait()
            total = len(frames_raw)
            if not total: raise RuntimeError("No frames in GIF")
            ups_list, sum_t = [], 0.0
            for i, fr in enumerate(frames_raw, 1):
                if cancel.is_set(): raise Exception("Cancelled")
                t0 = time.time()
                out, _ = ups.enhance(fr, outscale=scale)
                sum_t += time.time() - t0; ups_list.append(out)
                status.update(2, f"🎞 GIF Frame: **{i}/{total}**\n⚡ {sum_t/i:.2f}s/frame", i * 100 / total)
            frames_raw.clear(); gc.collect()
            loops = loop_count if 0 < loop_count <= 20 else min(max(1, math.ceil(GIF_MIN_SEC / (total / fps_g))), max(1, 600 // total))
            enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                    "-s", f"{ow}x{oh}", "-r", f"{fps_g:.6f}", "-i", "pipe:0",
                                    "-c:v", "libx264", "-preset", ff["preset"], "-crf", ff["crf"],
                                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)], stdin=subprocess.PIPE)
            for _ in range(loops):
                for fr in ups_list: enc.stdin.write(fr.tobytes())
            enc.stdin.close(); enc.wait()
            stats.update(frames=total, avg_spf=(sum_t / total if total else 0.0), loops=loops)
        stats["seconds"] = time.time() - t_start
        return stats
    finally:
        for p in (dec, enc):
            try:
                if p and p.poll() is None: p.kill()
            except Exception: pass

# ============================================================
# 16. JOB WORKER (per-job status + cleanup)
# ============================================================
async def run_job_worker(chat_id: int, job: Dict[str, Any]):
    label = job["filename"][:22]
    job_dir = None
    out_path = None
    try:
        status_msg = await app.send_message(chat_id, f"📥 **Downloading:** {label}...")
        status = LiveStatus(status_msg, asyncio.get_running_loop(), label)
        job_dir = WORK_DIR / f"job_{job['message_id']}_{int(time.time())}"
        job_dir.mkdir(exist_ok=True)
        in_path = job_dir / job["filename"]
        status.update(0, f"File: **{job['filename']}**")
        await app.download_media(job["message"], file_name=str(in_path))

        status.update(1, "🔍 FFprobe analysis...")
        info = await asyncio.to_thread(probe_video, in_path)
        if not job["is_gif"] and info["frames"] > MAX_FRAMES:
            await status.set_text(f"❌ **{label}** bahut badi: {info['frames']} frames (max {MAX_FRAMES})")
            return
        status.update(1, (f"📐 **{info['width']}×{info['height']}** • 🎞 {info['fps']:.2f} fps\n"
                          f"⏱ {fmt_time(info['duration'])} • 🎬 {info['codec']}\n"
                          f"🎯 {fmt_scale(job['scale'])}× • ⚡ {job['preset'].title()} • 🔊 {job['audio'].title()}"))
        await asyncio.sleep(2)

        out_path = OUTPUT_DIR / f"{Path(job['filename']).stem}_up_{job['message_id']}.mp4"
        stats = await asyncio.to_thread(run_pipeline_sync, in_path, out_path, info,
                                        job, status, job["cancel"], job["is_gif"])
        size_mb = out_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_SEND_MB:
            raise RuntimeError(f"Output {size_mb:.0f} MB > 2GB limit")
        status.update(5, f"Size: **{size_mb:.1f} MB**")
        cap = (f"✅ **Done:** {job['filename']}\n\n"
               f"🎯 {fmt_scale(stats['scale'])}× → {stats['out_w']}×{stats['out_h']}\n"
               f"🎞 Frames: {stats['frames']}" + (f" (loop ×{stats['loops']})" if job["is_gif"] else "") + "\n"
               f"⚡ Avg/frame: {stats['avg_spf']:.2f}s • 🕒 {fmt_time(stats['seconds'])}\n"
               f"📦 {size_mb:.1f} MB • ⚡ {job['preset'].title()}")
        if job["is_gif"]:
            await app.send_animation(chat_id, str(out_path), caption=cap)
        else:
            await app.send_video(chat_id, str(out_path), caption=cap, supports_streaming=True)
        await status.set_text(f"✅ **{label}** complete! 🎉")
    except Exception as e:
        log.exception("Job failed: %s", label)
        try: await app.send_message(chat_id, f"❌ **{label}** failed:\n`{str(e)[:250]}`")
        except Exception: pass
    finally:
        if job_dir: shutil.rmtree(job_dir, ignore_errors=True)
        if out_path and out_path.exists():
            try: out_path.unlink()
            except Exception: pass

# ============================================================
# 17. TEXT FALLBACK
# ============================================================
@app.on_message(filters.text & filters.private & ~filters.command(
    ["start", "help", "settings", "queue", "cancel", "reset", "ping", "specs", "workers"]))
async def text_fallback(client, message: Message):
    if not is_owner(message): return
    t = (message.text or "").lower()
    if any(w in t for w in ["hi", "hello", "hey", "namaste"]):
        await message.reply_text("🙏 Namaste! /start se shuru karo, ya video bhejo.", reply_markup=mode_keyboard())
    elif "spec" in t or "system" in t or "ram" in t or "cpu" in t:
        await message.reply_text(get_system_specs())
    elif "worker" in t or "parallel" in t:
        await workers_handler(client, message)
    elif "status" in t:
        sess = get_session(message.chat.id)
        await message.reply_text(f"🏃 Running: {len(sess.running_jobs)} • ⏳ Queue: {len(sess.queue)} • ⚙️ Workers: {sess.workers}")
    else:
        await message.reply_text(
            "🤖 **Smart Upscaler v2**\n/specs • /workers • /queue • /start",
            reply_markup=mode_keyboard()
        )

# ============================================================
# 18. START ENGINE (UNCHANGED - WORKING)
# ============================================================
if __name__ == "__main__":
    log.info("🚀 Pyrogram Engine (app.run) + parallel workers ready...")
    app.run()
