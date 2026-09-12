#!/usr/bin/env python3
"""
Smart Anime Upscaler v4 - AI TURBO
4-core optimal + RAM max use + adaptive per-frame AI optimizer
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
MAX_JOBS = 3
MAX_QUEUE = 12
CLIP_SECONDS = 2.0
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
    return out_px * 64 * 4 * 2

def spf_single(out_px: int) -> float:
    return 4.5e-7 * out_px + 0.06

# ================= AI FRAME OPTIMIZER =================
CONFIGS = [(1, 4), (2, 2), (2, 3), (3, 1), (4, 1)]   # (workers, torch threads)
PROBE_FRAMES = 6
EXPLOIT_FRAMES = 12

class FrameAI:
    """Har frame ke time se seekh kar workers/threads live tune karta hai."""
    def __init__(self, out_px: int):
        free = mem_free_gb()
        fp = footprint_bytes(out_px) / 1e9
        self.cands = [c for c in CONFIGS if c[0] * fp <= max(1.0, free * 0.75)]
        if not self.cands:
            self.cands = [(1, 1)]
        self.ema = {c: 0.0 for c in CONFIGS}
        self.cnt = {c: 0 for c in CONFIGS}
        self.current = (2, 2) if (2, 2) in self.cands else self.cands[0]
        self.best = self.current
        self.probe_left = PROBE_FRAMES
        self.exploit_left = 0
        self.lock = threading.Lock()
        self.apply()
        log.info("🤖 AI start: %s (free RAM %.1fGB, footprint %.2fGB)", self.current, free, fp)

    def apply(self):
        torch.set_num_threads(self.current[1])

    def throughput(self, c) -> float:
        e = self.ema.get(c, 0.0)
        return (c[0] / e) if e else 0.0

    def on_frame(self, cfg, dt: float):
        with self.lock:
            c = tuple(cfg)
            if c not in self.ema:
                return
            self.ema[c] = dt if self.cnt[c] == 0 else self.ema[c] * 0.7 + dt * 0.3
            self.cnt[c] += 1
            # decision logic
            if self.probe_left > 0:
                self.probe_left -= 1
                if self.probe_left == 0:
                    self._pick_best()
                    self.exploit_left = EXPLOIT_FRAMES
                return
            if self.exploit_left > 0:
                self.exploit_left -= 1
                if self.exploit_left == 0:
                    self._next_probe()

    def _pick_best(self):
        tested = [c for c in self.cands if self.cnt[c] >= 3]
        if not tested:
            return
        b = max(tested, key=self.throughput)
        if b != self.best:
            log.info("🤖 AI naya best: %sW×%sT (%.2f f/s)", b[0], b[1], self.throughput(b))
        self.best = b
        self.current = b
        self.apply()

    def _next_probe(self):
        untested = [c for c in self.cands if self.cnt[c] < 3]
        if untested:
            target = untested[0]
        else:
            target = min(self.cands, key=lambda c: self.cnt[c])
        if target != self.current:
            self.current = target
            self.apply()
            log.info("🤖 AI probe: %sW×%sT", target[0], target[1])
        self.probe_left = PROBE_FRAMES

    def status(self, spf: float) -> str:
        return (f"🤖 AI: {self.current[0]}W×{self.current[1]}T | {spf:.2f}s/fr | "
                f"{self.throughput(self.current):.2f} f/s | best {self.best[0]}W×{self.best[1]}T "
                f"({self.throughput(self.best):.2f} f/s)")

# ================= PRE-FLIGHT (UNCHANGED) =================
def clean_telegram_state():
    log.info("🧹 Wiping webhooks...")
    try:
        r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook?drop_pending_updates=True", timeout=10)
        log.info("Webhook: %s", r.text[:120])
        time.sleep(3)
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      json={"chat_id": OWNER_CHAT_ID,
                            "text": "✅ Upscaler v4 AI-Turbo online!\n🎛 Panel buttons se sab control — /start karo."}, timeout=10)
    except Exception as e:
        log.error("Pre-flight: %s", e)

clean_telegram_state()

app = Client("anime_upscaler_bot", api_id=API_ID, api_hash=API_HASH,
             bot_token=BOT_TOKEN, in_memory=True)

# ================= SESSION =================
class Session:
    def __init__(self):
        self.scale = 2.0
        self.preset = "balanced"
        self.audio = "keep"
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

_ups = None
def get_ups() -> RealESRGANer:
    global _ups
    if _ups is None:
        log.info("Loading AnimeVideo-v3 (tile=0 => RAM max, no tiling bugs)")
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
        _ups = RealESRGANer(scale=4, model_path=str(MODEL_PATH), model=model,
                            tile=0, tile_pad=0, pre_pad=0, half=False, device=torch.device("cpu"))
    return _ups

# ================= PANEL =================
def panel_kb(s: Session) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🎯 {s.scale:g}×", callback_data="b:qmenu"),
         InlineKeyboardButton(f"⚡ {s.preset.title()}", callback_data="b:pmenu"),
         InlineKeyboardButton(f"🔊 {s.audio.title()}", callback_data="b:amenu")],
        [InlineKeyboardButton("⛔ Stop All", callback_data="b:stop"),
         InlineKeyboardButton("🧹 Clean", callback_data="b:clean")],
    ])

def quality_kb(): return InlineKeyboardMarkup([
    [InlineKeyboardButton("1.5×", callback_data="b:q:1.5"), InlineKeyboardButton("2×", callback_data="b:q:2"),
     InlineKeyboardButton("3×", callback_data="b:q:3"), InlineKeyboardButton("4×", callback_data="b:q:4")],
    [InlineKeyboardButton("🔙", callback_data="b:back")]])

def preset_kb(): return InlineKeyboardMarkup([
    [InlineKeyboardButton("⚡ Fast", callback_data="b:p:fast"),
     InlineKeyboardButton("⚖️ Balanced", callback_data="b:p:balanced"),
     InlineKeyboardButton("💎 Best", callback_data="b:p:best")],
    [InlineKeyboardButton("🔙", callback_data="b:back")]])

def audio_kb(): return InlineKeyboardMarkup([
    [InlineKeyboardButton("🔊 Keep", callback_data="b:a:keep"),
     InlineKeyboardButton("🗜 Compress", callback_data="b:a:compress"),
     InlineKeyboardButton("🔇 Remove", callback_data="b:a:remove")],
    [InlineKeyboardButton("🔙", callback_data="b:back")]])

def panel_text(s: Session) -> str:
    lines = [f"🎛 **Upscaler v4 AI-Turbo** • 🧠 {CPU_THREADS} cores • 💾 {mem_free_gb():.1f}GB free", ""]
    for j in [j for j in s.jobs if j["status"] not in ("done", "fail")]:
        st = j["status"]
        head = f"{STAGE_EMO[st]} **{j['filename'][:22]}**"
        if st == "dl":
            lines += [head + " — ⬇️ Download", f"   {j.get('dl_done',0)/1048576:.1f}/{j.get('dl_total',0)/1048576:.1f} MB"]
        elif st == "probe":
            lines += [head + " — 🔍 Analyzing..."]
        elif st == "up":
            pct = j["done"] * 100 / j["total"] if j["total"] else 0
            eta = (j["total"] - j["done"]) * j["spf"] / max(1, j.get("conc", 1)) if j["spf"] else 0
            lines += [head + f" — 🎨 {bar(pct)} {pct:.0f}%",
                      f"   {j['done']}/{j['total']} fr • ⚡ {j['spf']:.2f}s/fr • ⏱ ETA {fmt_time(eta)}",
                      "   " + j.get("ai", "🤖 AI: warmup...")]
        elif st == "enc":
            lines += [head + " — 📦 Encoding..."]
        elif st == "upload":
            lines += [head + " — ⬆️ Upload", f"   {j.get('ul_done',0)/1048576:.1f}/{j.get('ul_total',0)/1048576:.1f} MB"]
    q = [j for j in s.jobs if j["status"] == "queued"]
    if q: lines += ["", "⏳ Queue: " + ", ".join(j["filename"][:14] for j in q[:4])]
    if s.history: lines += ["", "📜 " + " | ".join(s.history[-3:])]
    if not q and not [j for j in s.jobs if j["status"] not in ("done", "fail")]:
        lines += ["😴 Idle — video/GIF bhejo; AI har frame ke saath khud fast hota jayega."]
    return "\n".join(lines)

async def ensure_panel(cid: int) -> Message:
    s = get_sess(cid)
    if s.panel is None:
        s.panel = await app.send_message(cid, panel_text(s), reply_markup=panel_kb(s))
    return s.panel

async def refresh_panel(cid: int):
    s = get_sess(cid)
    try:
        if s.panel: await s.panel.edit_text(panel_text(s), reply_markup=panel_kb(s))
    except Exception: pass

async def refresher_loop(cid: int):
    s = get_sess(cid)
    while any(j["status"] not in ("done", "fail") for j in s.jobs):
        await refresh_panel(cid); await asyncio.sleep(2.5)
    await refresh_panel(cid); s.refresher = None

def kick_refresher(cid: int):
    s = get_sess(cid)
    if s.refresher is None or s.refresher.done():
        s.refresher = asyncio.create_task(refresher_loop(cid))

# ================= SCHEDULER =================
POOL = ThreadPoolExecutor(max_workers=4)

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

# ================= PIPELINE (AI-TURBO) =================
def run_sync_job(job, in_path: Path, out_path: Path, info: Dict, s: Session):
    w, h, fps = info["width"], info["height"], info["fps"]
    scale = s.scale
    ow, oh = int(w * scale + (int(w * scale) % 2)), int(h * scale + (int(h * scale) % 2))
    while scale > 1.0 and ow * oh > MAX_OUT_PIXELS:
        scale = max(1.0, scale - 0.5)
        ow, oh = int(w * scale + (int(w * scale) % 2)), int(h * scale + (int(h * scale) % 2))
    job["ow"], job["oh"] = ow, oh
    out_px = ow * oh
    ai = FrameAI(out_px)
    job["conc"] = ai.current[0]
    ups = get_ups()
    ff = PRESETS.get(s.preset, PRESETS["balanced"])
    is_gif = job["is_gif"]
    fps_g = fps if fps > 0 else 10.0
    total = min(info["frames"] or max(1, int(info["duration"] * fps)), MAX_GIF_FRAMES if is_gif else 10**9)
    job["total"] = total
    cancel: threading.Event = job["cancel"]
    in_q: queue.Queue = queue.Queue(maxsize=6)      # RAM-max prefetch
    futs: List = []
    futs_cond = threading.Condition()
    stats = {"done": 0, "sum": 0.0}
    stats_lock = threading.Lock()
    enc_threads = max(1, CPU_THREADS - ai.current[0] * ai.current[1] + 1)

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
                in_q.put(np.frombuffer(raw, np.uint8).reshape(h, w, 3)); n += 1
            dec.wait()
        finally:
            in_q.put(None)

    def upscale_one(img, cfg):
        t0 = time.time()
        out, _ = ups.enhance(img, outscale=scale)
        dt = time.time() - t0
        ai.on_frame(cfg, dt)                       # 🤖 AI seekhta hai
        with stats_lock:
            stats["done"] += 1; stats["sum"] += dt
            job["done"] = stats["done"]
            job["spf"] = ai.ema.get(tuple(cfg), dt)
            job["ai"] = ai.status(job["spf"])
            job["conc"] = ai.current[0]
        del img
        return out

    enc = None
    def encoder():
        nonlocal enc
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{ow}x{oh}", "-r", f"{(fps_g if is_gif else fps):.6f}", "-i", "pipe:0"]
        if not is_gif:
            cmd += ["-i", str(in_path), "-map", "0:v:0"]
            if info["has_audio"]:
                if s.audio == "keep": cmd += ["-map", "1:a?", "-c:a", "copy"]
                elif s.audio == "compress": cmd += ["-map", "1:a?", "-c:a", "aac", "-b:a", "128k"]
        cmd += ["-c:v", "libx264", "-preset", ff["preset"], "-crf", ff["crf"],
                "-threads", str(enc_threads), "-pix_fmt", "yuv420p"]
        if not is_gif and info["has_audio"] and s.audio != "remove": cmd += ["-shortest"]
        cmd += ["-movflags", "+faststart", str(out_path)]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        i = 0
        while True:
            with futs_cond:
                while len(futs) <= i:
                    futs_cond.wait(0.2)
                    if cancel.is_set() and len(futs) <= i: return
                f = futs[i]
                if f is None: break
            arr = f.result()
            enc.stdin.write(arr.tobytes()); del arr; i += 1
        enc.stdin.close(); enc.wait()

    th_read = threading.Thread(target=reader, daemon=True); th_read.start()

    if is_gif:
        i = 0
        while True:
            if cancel.is_set(): raise RuntimeError("Cancelled")
            img = in_q.get()
            if img is None: break
            cfg = tuple(ai.current)
            f = POOL.submit(upscale_one, img, cfg)
            with futs_cond: futs.append(f); futs_cond.notify_all()
            i += 1
        th_read.join()
        outs = [f.result() for f in futs]
        job["done"] = len(outs); job["total"] = len(outs)
        loops = min(max(1, math.ceil(CLIP_SECONDS / (len(outs) / fps_g))), max(1, 600 // max(1, len(outs))))
        job["loops"] = loops
        job["status"] = "enc"
        enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", f"{ow}x{oh}", "-r", f"{fps_g:.6f}", "-i", "pipe:0",
                                "-c:v", "libx264", "-preset", ff["preset"], "-crf", ff["crf"],
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)], stdin=subprocess.PIPE)
        for _ in range(loops):
            for arr in outs: enc.stdin.write(arr.tobytes())
        enc.stdin.close(); enc.wait()
    else:
        th_enc = threading.Thread(target=encoder, daemon=True); th_enc.start()
        i = 0
        while True:
            if cancel.is_set():
                with futs_cond: futs.append(None); futs_cond.notify_all()
                raise RuntimeError("Cancelled")
            img = in_q.get()
            if img is None: break
            while (i - stats["done"]) >= ai.current[0]:     # in-flight cap = AI workers
                time.sleep(0.02)
                if cancel.is_set():
                    with futs_cond: futs.append(None); futs_cond.notify_all()
                    raise RuntimeError("Cancelled")
            cfg = tuple(ai.current)
            f = POOL.submit(upscale_one, img, cfg)
            with futs_cond: futs.append(f); futs_cond.notify_all()
            i += 1
        with futs_cond: futs.append(None); futs_cond.notify_all()
        th_enc.join(timeout=7200); th_read.join()
    if enc is not None and enc.returncode not in (0, None):
        raise RuntimeError("FFmpeg encode failed")
    job["frames_done"] = job["done"]
    job["ai"] = ai.status(job.get("spf", 0))
    log.info("🤖 AI final: best=%s thr=%.2f f/s", ai.best, ai.throughput(ai.best))

async def run_job(cid: int, job):
    s = get_sess(cid)
    job_dir = None; out_path = None
    try:
        job_dir = WORK_DIR / f"job_{job['mid']}_{int(time.time())}"
        job_dir.mkdir(parents=True, exist_ok=True)
        in_path = job_dir / job["filename"]
        def dl_cb(cur, tot, *a): job["dl_done"], job["dl_total"] = cur, tot
        await app.download_media(job["msg"], file_name=str(in_path), progress=dl_cb)
        job["status"] = "probe"; await refresh_panel(cid)
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
        job["status"] = "up"; job["t0"] = time.time()
        await asyncio.to_thread(run_sync_job, job, in_path, out_path, info, s)
        size_mb = out_path.stat().st_size / 1048576
        if size_mb > MAX_SEND_MB: raise RuntimeError(f"Output {size_mb:.0f}MB > 2GB")
        job["status"] = "upload"
        def ul_cb(cur, tot, *a): job["ul_done"], job["ul_total"] = cur, tot
        cap = (f"✅ **{job['filename']}**\n🎯 {s.scale:g}× → {job['ow']}×{job['oh']} • "
               f"⚡ {job.get('spf',0):.2f}s/fr\n{job.get('ai','')}\n"
               f"🎞 {job.get('frames_done',0)} fr" + (f" (loop ×{job.get('loops',1)})" if job["is_gif"] else "") +
               f" • 🕒 {fmt_time(time.time() - job['t0'])} • 📦 {size_mb:.1f}MB")
        if job["is_gif"]: await app.send_animation(cid, str(out_path), caption=cap, progress=ul_cb)
        else: await app.send_video(cid, str(out_path), caption=cap, supports_streaming=True, progress=ul_cb)
        job["status"] = "done"; s.history.append(f"✅ {job['filename'][:12]}")
    except Exception as e:
        log.exception("Job fail %s", job["filename"])
        job["status"] = "fail"; s.history.append(f"❌ {job['filename'][:12]}")
    finally:
        if job_dir: shutil.rmtree(job_dir, ignore_errors=True)
        if out_path and out_path.exists():
            try: out_path.unlink()
            except Exception: pass
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
        await message.reply_text("❌ Sirf video/GIF bhejo."); return
    dur = getattr(media, "duration", 0) or 0
    clip = is_gif or (0 < dur < CLIP_SECONDS)
    if len([j for j in s.jobs if j["status"] not in ("done", "fail")]) >= MAX_QUEUE:
        await message.reply_text("⚠️ Queue full (12)."); return
    s.jobs = [j for j in s.jobs if j["status"] not in ("done", "fail")][-9:] + [{
        "mid": message.id, "msg": message, "filename": fn, "is_gif": is_gif, "clip": clip,
        "status": "queued", "cancel": threading.Event(), "done": 0, "total": 0, "spf": 0.0,
        "conc": 1, "ow": 0, "oh": 0, "dl_done": 0, "dl_total": 0, "ul_done": 0, "ul_total": 0}]
    await ensure_panel(message.chat.id); pump(message.chat.id)

# ================= BUTTONS =================
@app.on_callback_query(filters.regex(r"^b:"))
async def btn(client, cq: CallbackQuery):
    if not is_owner(cq.message):
        await cq.answer("Private bot!", show_alert=True); return
    s = get_sess(cq.message.chat.id)
    parts = cq.data[2:].split(":"); a = parts[0]; v = parts[1] if len(parts) > 1 else ""
    kb = panel_kb(s)
    if a == "qmenu": kb = quality_kb()
    elif a == "pmenu": kb = preset_kb()
    elif a == "amenu": kb = audio_kb()
    elif a == "q": s.scale = float(v); await cq.answer(f"Scale {s.scale:g}×")
    elif a == "p": s.preset = v; await cq.answer(f"Preset {v}")
    elif a == "a": s.audio = v; await cq.answer(f"Audio {v}")
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
        await ensure_panel(cq.message.chat.id); await cq.answer("🧹 Clean!"); return
    else:
        await cq.answer(); return
    try:
        if s.panel and cq.message.id == s.panel.id:
            await s.panel.edit_text(panel_text(s), reply_markup=kb)
        else:
            await cq.message.edit_text(panel_text(s), reply_markup=kb); s.panel = cq.message
    except Exception: pass
    await cq.answer()

@app.on_message(filters.command("start") & filters.private)
async def start_handler(client, message: Message):
    if not is_owner(message): return
    s = get_sess(message.chat.id)
    try:
        if s.panel: await s.panel.delete()
    except Exception: pass
    s.panel = None
    await ensure_panel(message.chat.id); await refresh_panel(message.chat.id)

# ================= ENGINE (UNCHANGED) =================
if __name__ == "__main__":
    log.info("🚀 v4 AI-Turbo start (cores=%s)", CPU_THREADS)
    app.run()
