#!/usr/bin/env python3
"""
Smart Anime Upscaler v3 - Panel UI + Math-based parallelism
Engine: webhook wipe + in_memory + app.run() (UNCHANGED, WORKING)
"""
import asyncio
import gc
import json
import logging
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import requests
import torch
from pyrogram import Client, filters
from pyrogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# ================= CONFIG (UNCHANGED) =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("anime-upscaler")

API_ID = int(os.getenv("API_ID", "0") or 0)
API_HASH = (os.getenv("API_HASH", "")).strip()
BOT_TOKEN = (os.getenv("BOT_TOKEN", "")).strip()
OWNER_CHAT_ID = (os.getenv("OWNER_CHAT_ID", "") or "").strip().strip("'").strip('"')

if not API_ID or not API_HASH or not BOT_TOKEN or not OWNER_CHAT_ID:
    raise RuntimeError("Missing GitHub Secrets.")

WORK_DIR = Path("work"); OUTPUT_DIR = Path("output")
MODEL_PATH = Path("weights/realesr-animevideov3.pth")
WORK_DIR.mkdir(exist_ok=True); OUTPUT_DIR.mkdir(exist_ok=True)

MAX_FRAMES = 3600
MAX_GIF_FRAMES = 240
MAX_OUT_PIXELS = 3840 * 2160
MAX_SEND_MB = 1900
MAX_JOBS = 3          # max concurrent jobs (clips)
MAX_QUEUE = 12
CLIP_SECONDS = 2.0    # isse chhoti video/GIF = clip (parallel eligible)
CPU_THREADS = os.cpu_count() or 4
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)

# ================= RAM / MATH =================
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
    """Ek frame ke upscale me lagbhag RAM (model activations)."""
    return out_px * 64 * 4 * 2

def spf_single(out_px: int) -> float:
    """Ek frame ka time agar single-thread-style chale."""
    return 4.5e-7 * out_px + 0.06

FREE_GB = mem_free_gb()
K_POOL = max(1, min(3, CPU_THREADS - 1, int(FREE_GB * 0.6 // 1.3)))
torch.set_num_threads(max(1, CPU_THREADS // K_POOL))
POOL = ThreadPoolExecutor(max_workers=K_POOL)
log.info("🧮 Pool workers=%s (cores=%s, free RAM=%.1fGB)", K_POOL, CPU_THREADS, FREE_GB)

def job_conc_for(out_px: int) -> int:
    free = mem_free_gb()
    by_ram = max(1, int(free * 0.6 * 1e9 // max(1, footprint_bytes(out_px))))
    return max(1, min(K_POOL, by_ram))

# ================= PRE-FLIGHT (UNCHANGED) =================
def clean_telegram_state():
    log.info("🧹 Wiping webhooks...")
    try:
        r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook?drop_pending_updates=True", timeout=10)
        log.info("Webhook: %s", r.text[:120])
        time.sleep(3)
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      json={"chat_id": OWNER_CHAT_ID,
                            "text": "✅ Upscaler v3 online!\n🎛 Sab kuch buttons par — /start karo."}, timeout=10)
    except Exception as e:
        log.error("Pre-flight: %s", e)

clean_telegram_state()

app = Client("anime_upscaler_bot", api_id=API_ID, api_hash=API_HASH,
             bot_token=BOT_TOKEN, in_memory=True)

# ================= SESSION / JOBS =================
class Session:
    def __init__(self):
        self.scale = 2.0
        self.preset = "balanced"
        self.audio = "keep"
        self.jobs: List[Dict[str, Any]] = []
        self.history: List[str] = []
        self.panel: Optional[Message] = None
        self.refresher: Optional[asyncio.Task] = None
        self.lock = threading.Lock()

SESSIONS: Dict[int, Session] = {}
def get_sess(cid: int) -> Session:
    if cid not in SESSIONS: SESSIONS[cid] = Session()
    return SESSIONS[cid]

def is_owner(m) -> bool:
    cid = m.chat.id if hasattr(m, "chat") else m.from_user.id
    return str(cid) == OWNER_CHAT_ID or cid == (int(OWNER_CHAT_ID) if OWNER_CHAT_ID.lstrip("-").isdigit() else 0)

# ================= HELPERS =================
def fmt_time(s: float) -> str:
    s = max(0, int(s)); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h}h {m}m {sec}s" if h else (f"{m}m {sec}s" if m else f"{sec}s")

def bar(pct: float, n: int = 8) -> str:
    f = int(n * min(100, max(0, pct)) / 100)
    return "▰" * f + "▱" * (n - f)

STAGE_EMO = {"queued": "⏳", "dl": "⬇️", "probe": "🔍", "up": "🎨", "enc": "📦", "upload": "⬆️", "done": "✅", "fail": "❌"}

PRESETS = {"fast": {"crf": "23", "preset": "veryfast"},
           "balanced": {"crf": "19", "preset": "veryfast"},
           "best": {"crf": "16", "preset": "slow"}}

# ================= MODEL (tile=0 => crash fix) =================
_ups = None
def get_ups() -> RealESRGANer:
    global _ups
    if _ups is None:
        log.info("Loading AnimeVideo-v3 (tile=0, no tiling => no tensor bug)")
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
        _ups = RealESRGANer(scale=4, model_path=str(MODEL_PATH), model=model,
                            tile=0, tile_pad=0, pre_pad=0, half=False, device=torch.device("cpu"))
    return _ups

# ================= PANEL UI =================
def panel_kb(s: Session) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🎯 Quality: {s.scale:g}×", callback_data="b:qmenu"),
         InlineKeyboardButton(f"⚡ {s.preset.title()}", callback_data="b:pmenu"),
         InlineKeyboardButton(f"🔊 {s.audio.Title() if False else s.audio.title()}", callback_data="b:amenu")],
        [InlineKeyboardButton("⛔ Stop All", callback_data="b:stop"),
         InlineKeyboardButton("🧹 Clean Chat", callback_data="b:clean")],
    ])

def quality_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("1.5×", callback_data="b:q:1.5"), InlineKeyboardButton("2×", callback_data="b:q:2"),
         InlineKeyboardButton("3×", callback_data="b:q:3"), InlineKeyboardButton("4×", callback_data="b:q:4")],
        [InlineKeyboardButton("🔙 Panel", callback_data="b:back")],
    ])

def preset_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚡ Fast", callback_data="b:p:fast"),
         InlineKeyboardButton("⚖️ Balanced", callback_data="b:p:balanced"),
         InlineKeyboardButton("💎 Best", callback_data="b:p:best")],
        [InlineKeyboardButton("🔙 Panel", callback_data="b:back")],
    ])

def audio_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔊 Keep", callback_data="b:a:keep"),
         InlineKeyboardButton("🗜 Compress", callback_data="b:a:compress"),
         InlineKeyboardButton("🔇 Remove", callback_data="b:a:remove")],
        [InlineKeyboardButton("🔙 Panel", callback_data="b:back")],
    ])

def panel_text(s: Session) -> str:
    lines = [f"🎛 **Upscaler Panel** • 🧠 {K_POOL} workers • 💾 {mem_free_gb():.1f}GB free", ""]
    running = [j for j in s.jobs if j["status"] not in ("done", "fail")]
    for j in running:
        st = j["status"]
        head = f"{STAGE_EMO[st]} **{j['filename'][:24]}**"
        if st == "dl":
            dl = j.get("dl_done", 0) / 1048576; dt = j.get("dl_total", 0) / 1048576
            lines += [head + " — ⬇️ Download", f"   {dl:.1f} / {dt:.1f} MB"]
        elif st == "probe":
            lines += [head + " — 🔍 Analyzing..."]
        elif st == "up":
            pct = j["done"] * 100 / j["total"] if j["total"] else 0
            eta = (j["total"] - j["done"]) * j["spf"] / max(1, j["conc"]) if j["spf"] else 0
            lines += [head + f" — 🎨 {bar(pct)} {pct:.0f}%",
                      f"   {j['done']}/{j['total']} frames • ⚡ {j['spf']:.2f}s/fr ×{j['conc']} concurrent",
                      f"   ⏱ ETA {fmt_time(eta)} • 🖼 {j['ow']}×{j['oh']}"]
        elif st == "enc":
            lines += [head + " — 📦 Encoding..."]
        elif st == "upload":
            ul = j.get("ul_done", 0) / 1048576; ut = j.get("ul_total", 0) / 1048576
            lines += [head + " — ⬆️ Upload", f"   {ul:.1f} / {ut:.1f} MB"]
        else:
            lines += [head]
    queued = [j for j in s.jobs if j["status"] == "queued"]
    if queued:
        lines += ["", "⏳ Queue: " + ", ".join(j["filename"][:16] for j in queued[:4])]
    if s.history:
        lines += ["", "📜 " + " | ".join(s.history[-3:])]
    if not running and not queued:
        lines += ["😴 Idle — video/GIF bhejo; options upar buttons se."]
    return "\n".join(lines)

async def ensure_panel(cid: int) -> Message:
    s = get_sess(cid)
    if s.panel is None:
        s.panel = await app.send_message(cid, panel_text(s), reply_markup=panel_kb(s))
    return s.panel

async def refresh_panel(cid: int):
    s = get_sess(cid)
    try:
        if s.panel is None: return
        await s.panel.edit_text(panel_text(s), reply_markup=panel_kb(s))
    except Exception:
        pass

async def refresher_loop(cid: int):
    s = get_sess(cid)
    while any(j["status"] not in ("done", "fail") for j in s.jobs):
        await refresh_panel(cid)
        await asyncio.sleep(2.5)
    await refresh_panel(cid)
    s.refresher = None

def kick_refresher(cid: int):
    s = get_sess(cid)
    if s.refresher is None or s.refresher.done():
        s.refresher = asyncio.create_task(refresher_loop(cid))

# ================= SCHEDULER (auto batch/parallel math) =================
def pump(cid: int):
    s = get_sess(cid)
    running = [j for j in s.jobs if j["status"] not in ("done", "fail", "queued")]
    running_long = [j for j in running if not j["clip"]]
    for j in s.jobs:
        if j["status"] != "queued": continue
        if len(running) >= MAX_JOBS: break
        if not j["clip"] and running_long: continue   # lambi video akeli chalti hai
        if not j["clip"]: running_long.append(j)
        running.append(j)
        j["status"] = "dl"
        asyncio.create_task(run_job(cid, j))
    kick_refresher(cid)

# ================= PIPELINE (3 frames ek saath, ordered encode) =================
def run_sync_job(job: Dict[str, Any], in_path: Path, out_path: Path, info: Dict, s: Session):
    w, h, fps = info["width"], info["height"], info["fps"]
    scale = s.scale
    ow, oh = int(w * scale + (int(w * scale) % 2)), int(h * scale + (int(h * scale) % 2))
    while scale > 1.0 and ow * oh > MAX_OUT_PIXELS:
        scale = max(1.0, scale - 0.5)
        ow, oh = int(w * scale + (int(w * scale) % 2)), int(h * scale + (int(h * scale) % 2))
    job["ow"], job["oh"] = ow, oh
    out_px = ow * oh
    conc = job_conc_for(out_px)
    job["conc"] = conc
    ups = get_ups()
    ff = PRESETS.get(s.preset, PRESETS["balanced"])
    is_gif = job["is_gif"]
    fps_g = fps if fps > 0 else 10.0

    total = info["frames"] or max(1, int(info["duration"] * fps))
    if is_gif: total = min(total, MAX_GIF_FRAMES)
    job["total"] = total
    # Math display: single vs parallel estimate
    est1 = total * spf_single(out_px)
    job["est_par"] = est1 / (conc * 0.8)

    cancel: threading.Event = job["cancel"]
    in_q: queue.Queue = queue.Queue(maxsize=conc * 2)
    futs: List = []
    futs_cond = threading.Condition()
    stats = {"done": 0, "sum": 0.0}
    stats_lock = threading.Lock()

    def reader():
        fb = w * h * 3
        try:
            dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(in_path), "-vsync", "0",
                                    "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE)
            n = 0
            while not cancel.is_set():
                raw = dec.stdout.read(fb)
                if not raw or len(raw) != fb: break
                if is_gif and n >= MAX_GIF_FRAMES: break
                in_q.put(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
                n += 1
            dec.wait()
        finally:
            in_q.put(None)
    job["_dec"] = None

    def upscale_one(img):
        t0 = time.time()
        out, _ = ups.enhance(img, outscale=scale)
        dt = time.time() - t0
        with stats_lock:
            stats["done"] += 1; stats["sum"] += dt
            job["done"] = stats["done"]; job["spf"] = stats["sum"] / stats["done"]
        del img
        return out

    enc = None
    def encoder(fut_list):
        nonlocal enc
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{ow}x{oh}", "-r", f"{(fps_g if is_gif else fps):.6f}", "-i", "pipe:0"]
        if not is_gif:
            cmd += ["-i", str(in_path), "-map", "0:v:0"]
            if info["has_audio"]:
                if s.audio == "keep": cmd += ["-map", "1:a?", "-c:a", "copy"]
                elif s.audio == "compress": cmd += ["-map", "1:a?", "-c:a", "aac", "-b:a", "128k"]
        cmd += ["-c:v", "libx264", "-preset", ff["preset"], "-crf", ff["crf"], "-pix_fmt", "yuv420p"]
        if not is_gif and info["has_audio"] and s.audio != "remove": cmd += ["-shortest"]
        cmd += ["-movflags", "+faststart", str(out_path)]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        i = 0
        while True:
            with futs_cond:
                while len(fut_list) <= i:
                    if fut_list and fut_list[-1] is None and len(fut_list) == i + 1:
                        break
                    futs_cond.wait(0.2)
                    if cancel.is_set() and len(fut_list) <= i: return
                f = fut_list[i]
                if f is None: break
            arr = f.result()
            enc.stdin.write(arr.tobytes())
            del arr
            i += 1
        enc.stdin.close(); enc.wait()

    th_read = threading.Thread(target=reader, daemon=True); th_read.start()
    ups_list: List = [] if is_gif else None
    if is_gif:
        # GIF: collect ordered, phir loop encode
        i = 0
        while True:
            if cancel.is_set(): raise RuntimeError("Cancelled")
            img = in_q.get()
            if img is None: break
            if i >= conc and futs[i - conc] is not None: futs[i - conc].result()
            f = POOL.submit(upscale_one, img)
            with futs_cond:
                futs.append(f); futs_cond.notify_all()
            i += 1
        th_read.join()
        for f in futs:
            ups_list.append(f.result())
        job["done"] = len(ups_list); job["total"] = len(ups_list)
        loops = job.get("loops", max(1, math.ceil(CLIP_SECONDS / (len(ups_list) / fps_g))))
        loops = min(loops, max(1, 600 // max(1, len(ups_list))))
        job["loops"] = loops
        job["status"] = "enc"
        enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", f"{ow}x{oh}", "-r", f"{fps_g:.6f}", "-i", "pipe:0",
                                "-c:v", "libx264", "-preset", ff["preset"], "-crf", ff["crf"],
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)], stdin=subprocess.PIPE)
        for _ in range(loops):
            for arr in ups_list:
                enc.stdin.write(arr.tobytes())
        enc.stdin.close(); enc.wait()
    else:
        th_enc = threading.Thread(target=encoder, args=(futs,), daemon=True); th_enc.start()
        i = 0
        while True:
            if cancel.is_set():
                with futs_cond:
                    futs.append(None); futs_cond.notify_all()
                raise RuntimeError("Cancelled")
            img = in_q.get()
            if img is None: break
            if i >= conc: futs[i - conc].result()   # RAM bound: sirf conc frames in-flight
            f = POOL.submit(upscale_one, img)
            with futs_cond:
                futs.append(f); futs_cond.notify_all()
            i += 1
        with futs_cond:
            futs.append(None); futs_cond.notify_all()
        th_enc.join(timeout=3600)
        th_read.join()
    if enc is not None and enc.returncode not in (0, None):
        raise RuntimeError("FFmpeg encode failed")
    job["frames_done"] = job["done"]

async def run_job(cid: int, job: Dict[str, Any]):
    s = get_sess(cid)
    job_dir = None; out_path = None
    try:
        job_dir = WORK_DIR / f"job_{job['mid']}_{int(time.time())}"
        job_dir.mkdir(parents=True, exist_ok=True)
        in_path = job_dir / job["filename"]

        def dl_cb(cur, tot, *a):
            job["dl_done"], job["dl_total"] = cur, tot
        await app.download_media(job["msg"], file_name=str(in_path), progress=dl_cb)

        job["status"] = "probe"
        await refresh_panel(cid)
        probe = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                                "-show_streams", "-show_format", str(in_path)], capture_output=True, text=True)
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
            raise RuntimeError(f"Video bahut lambi: {frames} frames (max {MAX_FRAMES})")

        out_path = OUTPUT_DIR / f"{Path(job['filename']).stem}_up_{job['mid']}.mp4"
        job["status"] = "up"
        await asyncio.to_thread(run_sync_job, job, in_path, out_path, info, s)

        size_mb = out_path.stat().st_size / 1048576
        if size_mb > MAX_SEND_MB: raise RuntimeError(f"Output {size_mb:.0f}MB > 2GB limit")
        job["status"] = "upload"

        def ul_cb(cur, tot, *a):
            job["ul_done"], job["ul_total"] = cur, tot
        cap = (f"✅ **{job['filename']}**\n🎯 {s.scale:g}× → {job['ow']}×{job['oh']} • "
               f"⚡ {job.get('spf', 0):.2f}s/fr ×{job['conc']} workers\n"
               f"🎞 {job.get('frames_done', 0)} frames" + (f" (loop ×{job.get('loops', 1)})" if job["is_gif"] else "") +
               f" • 🕒 {fmt_time(job.get('t0', 0) and (time.time() - job['t0']) or 0)} • 📦 {size_mb:.1f}MB")
        job["t0"] = job.get("t0") or time.time()
        if job["is_gif"]:
            await app.send_animation(cid, str(out_path), caption=cap, progress=ul_cb)
        else:
            await app.send_video(cid, str(out_path), caption=cap, supports_streaming=True, progress=ul_cb)
        job["status"] = "done"
        s.history.append(f"✅ {job['filename'][:14]}")
    except Exception as e:
        log.exception("Job fail %s", job["filename"])
        job["status"] = "fail"; job["err"] = str(e)[:180]
        s.history.append(f"❌ {job['filename'][:14]}")
    finally:
        if job_dir: shutil.rmtree(job_dir, ignore_errors=True)
        if out_path and out_path.exists():
            try: out_path.unlink()
            except Exception: pass
        pump(cid)
        kick_refresher(cid)

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
        await message.reply_text("❌ Sirf video/GIF: MP4 MKV MOV WEBM AVI M4V GIF")
        return
    dur = getattr(media, "duration", 0) or 0
    clip = is_gif or (0 < dur < CLIP_SECONDS)
    if len([j for j in s.jobs if j["status"] not in ("done", "fail")]) >= MAX_QUEUE:
        await message.reply_text("⚠️ Queue full (12). Thoda wait karo.")
        return
    job = {"mid": message.id, "msg": message, "filename": fn, "is_gif": is_gif,
           "clip": clip, "status": "queued", "cancel": threading.Event(),
           "done": 0, "total": 0, "spf": 0.0, "conc": 1, "ow": 0, "oh": 0,
           "dl_done": 0, "dl_total": 0, "ul_done": 0, "ul_total": 0}
    s.jobs = [j for j in s.jobs if j["status"] not in ("done", "fail")][-9:] + [job]
    await ensure_panel(message.chat.id)
    pump(message.chat.id)

# ================= BUTTONS =================
@app.on_callback_query(filters.regex(r"^b:"))
async def btn(client, cq: CallbackQuery):
    if not is_owner(cq.message):
        await cq.answer("Private bot!", show_alert=True); return
    s = get_sess(cq.message.chat.id)
    parts = cq.data[2:].split(":")
    a = parts[0]; v = parts[1] if len(parts) > 1 else ""
    if a == "qmenu": kb = quality_kb()
    elif a == "pmenu": kb = preset_kb()
    elif a == "amenu": kb = audio_kb()
    elif a == "back": kb = panel_kb(s)
    elif a == "q":
        s.scale = float(v); kb = panel_kb(s); await cq.answer(f"Quality {s.scale:g}×")
    elif a == "p":
        s.preset = v; kb = panel_kb(s); await cq.answer(f"Preset {v}")
    elif a == "a":
        s.audio = v; kb = panel_kb(s); await cq.answer(f"Audio {v}")
    elif a == "stop":
        n = 0
        for j in s.jobs:
            if j["status"] not in ("done", "fail", "queued"): j["cancel"].set(); n += 1
        kb = panel_kb(s); await cq.answer(f"⛔ {n} jobs roki")
    elif a == "clean":
        try:
            if s.panel: await s.panel.delete()
        except Exception: pass
        s.panel = None; s.jobs = []; s.history = []
        await cq.answer("🧹 Clean!")
        await ensure_panel(cq.message.chat.id)
        return
    else:
        await cq.answer(); return
    try:
        if s.panel and cq.message.id == s.panel.id:
            await s.panel.edit_text(panel_text(s), reply_markup=kb)
        else:
            await cq.message.edit_text(panel_text(s), reply_markup=kb)
            s.panel = cq.message
    except Exception:
        pass
    await cq.answer()

# ================= START =================
@app.on_message(filters.command("start") & filters.private)
async def start_handler(client, message: Message):
    if not is_owner(message): return
    s = get_sess(message.chat.id)
    try:
        if s.panel: await s.panel.delete()
    except Exception: pass
    s.panel = None
    await ensure_panel(message.chat.id)
    await refresh_panel(message.chat.id)

# ================= ENGINE (UNCHANGED) =================
if __name__ == "__main__":
    log.info("🚀 v3 engine start (pool=%s)", K_POOL)
    app.run()
