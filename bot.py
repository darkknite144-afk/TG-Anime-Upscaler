#!/usr/bin/env python3
"""
FRESH REWRITE: Telegram Anime Video Upscaler
- Uses strict Pyrogram app.run() engine (No custom loops)
- Webhook hard-reset before MTProto initialization
- Real-ESRGAN AnimeVideo-v3 / FFmpeg Pipeline
"""
import asyncio
import gc
import json
import logging
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import requests
import torch
from pyrogram import Client, filters
from pyrogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# ============================================================
# 1. CONFIGURATION & CLEANUP
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("anime-upscaler")

API_ID = int(os.getenv("API_ID", "0") or 0)
API_HASH = (os.getenv("API_HASH", "")).strip()
BOT_TOKEN = (os.getenv("BOT_TOKEN", "")).strip()

# Clean OWNER_CHAT_ID completely
_raw_owner = (os.getenv("OWNER_CHAT_ID", "") or "").strip().strip("'").strip('"')
OWNER_CHAT_ID = _raw_owner
try:
    OWNER_CHAT_ID_INT = int(_raw_owner)
except ValueError:
    OWNER_CHAT_ID_INT = 0

if not API_ID or not API_HASH or not BOT_TOKEN or not OWNER_CHAT_ID:
    raise RuntimeError("❌ Missing API_ID, API_HASH, BOT_TOKEN, or OWNER_CHAT_ID in GitHub Secrets.")

# Ensure Directories
WORK_DIR = Path("work")
OUTPUT_DIR = Path("output")
MODEL_PATH = Path("weights/realesr-animevideov3.pth")
WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Processing Limits & Settings
MAX_FRAMES = 3600
MAX_GIF_FRAMES = 240
MAX_OUT_PIXELS = 3840 * 2160
GIF_MIN_SEC = 2.0
MAX_SEND_MB = 1900
CPU_THREADS = os.cpu_count() or 4
torch.set_num_threads(CPU_THREADS)
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)

# Global States
settings = {"scale": 2.0}
job_state = {"active": False, "text": "😴 Idle — koi job nahi."}
cancel_event = None


# ============================================================
# 2. PRE-FLIGHT (Fixing the Ghost & Webhook Issue)
# ============================================================
def clean_telegram_state():
    """Forces Telegram to drop any webhook and pending messages so MTProto can connect cleanly."""
    log.info("🧹 Wiping old webhooks and pending updates...")
    try:
        res = requests.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook?drop_pending_updates=True",
            timeout=10
        )
        log.info("Webhook Wipe Response: %s", res.text)
        
        # Give Telegram servers a moment to route traffic back to long-polling/MTProto
        time.sleep(3) 
        
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": OWNER_CHAT_ID,
                "text": "✅ System Fresh Boot!\n\nPyrogram engine is starting...\nSend /ping to test connection."
            },
            timeout=10
        )
    except Exception as e:
        log.error("Pre-flight failed: %s", e)

clean_telegram_state()


# ============================================================
# 3. PYROGRAM CLIENT SETUP
# ============================================================
app = Client(
    "anime_upscaler_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    in_memory=True # Avoids session file locks on GitHub Actions
)
busy_lock = asyncio.Lock()


# ============================================================
# 4. BOT HANDLERS & LOGIC
# ============================================================
def is_owner(message: Message) -> bool:
    return str(message.chat.id) == OWNER_CHAT_ID or message.chat.id == OWNER_CHAT_ID_INT

def fmt_scale(s: float) -> str: return f"{s:g}"

def fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")

def quality_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{'✅ ' if abs(settings['scale'] - sc) < 0.01 else ''}{fmt_scale(sc)}×", callback_data=f"scale:{sc}")]
        for sc in [1.5, 2.0, 3.0, 4.0]
    ])

@app.on_message(filters.command("ping") & filters.private)
async def ping_handler(client, message: Message):
    if is_owner(message):
        await message.reply_text("🏓 Pong! Bot zinda hai aur messages padh raha hai.")

@app.on_message(filters.command("start") & filters.private)
async def start_handler(client, message: Message):
    if not is_owner(message): return
    await message.reply_text(
        "🎬 **Anime Video Upscaler Bot** (Naya Engine)\n\n"
        "Bhejo koi video ya GIF. Main ready hoon!\n"
        f"🎯 Current quality: {fmt_scale(settings['scale'])}×",
        reply_markup=quality_keyboard()
    )

@app.on_message(filters.command("quality") & filters.private)
async def quality_cmd(client, message: Message):
    if not is_owner(message): return
    await message.reply_text("🎛 Upscale quality chuno:", reply_markup=quality_keyboard())

@app.on_callback_query(filters.regex(r"^scale:"))
async def scale_callback(client, cq):
    if not is_owner(cq.message): return
    val = float(cq.data.split(":")[1])
    settings["scale"] = val
    await cq.answer(f"Quality: {fmt_scale(val)}×")
    await cq.message.edit_text(f"✅ Quality set: **{fmt_scale(val)}×**", reply_markup=quality_keyboard())

@app.on_message(filters.command("cancel") & filters.private)
async def cancel_cmd(client, message: Message):
    if not is_owner(message): return
    if job_state["active"] and cancel_event is not None:
        cancel_event.set()
        await message.reply_text("🛑 Cancel request bhej di gayi hai.")
    else:
        await message.reply_text("😴 Abhi koi job nahi chal rahi.")

@app.on_message(filters.text & filters.private)
async def fallback_text(client, message: Message):
    if not is_owner(message): return
    if not message.text.startswith("/"):
        await message.reply_text("🤖 Video/GIF bhejo, ya commands use karo (/quality, /ping, /cancel).")


# ============================================================
# 5. CORE UPSCALING PIPELINE 
# ============================================================
_models = {}
def get_upsampler(tile: int) -> RealESRGANer:
    if tile not in _models:
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
        _models[tile] = RealESRGANer(
            scale=4, model_path=str(MODEL_PATH), model=model, tile=tile,
            tile_pad=10, pre_pad=0, half=False, device=torch.device("cpu")
        )
    return _models[tile]

class LiveStatus:
    def __init__(self, msg, loop):
        self.msg, self.loop, self.last, self.last_text = msg, loop, 0.0, ""
    def request(self, text: str, force: bool = False):
        job_state["text"] = text
        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self._edit(text, force)))
    async def _edit(self, text: str, force: bool):
        now = time.time()
        if not force and (now - self.last < 4 or text == self.last_text): return
        self.last, self.last_text = now, text
        try: await self.msg.edit_text(text)
        except Exception: pass
    async def edit(self, text: str):
        job_state["text"] = text
        await self._edit(text, True)

def run_pipeline_sync(input_path, output_path, info, scale, status, cancel, is_gif):
    w, h, fps = info["width"], info["height"], info["fps"]
    
    # Compute output dims
    ow, oh = w * scale, h * scale
    while scale > 1.0 and (int(ow + ow%2) * int(oh + oh%2)) > MAX_OUT_PIXELS:
        scale = max(1.0, scale - 0.5)
        ow, oh = w * scale, h * scale
    ow, oh = int(ow + (ow % 2)), int(oh + (oh % 2))
    
    tile = 0 if (ow * oh) <= 2_600_000 else 320
    ups = get_upsampler(tile)
    t_start = time.time()
    dec = enc = None
    stats = {"frames": 0, "avg_spf": 0.0, "seconds": 0.0, "out_w": ow, "out_h": oh, "loops": 1, "scale": scale}

    def prog_txt(d, t, spf):
        eta = (t - d) * spf if t and spf else 0
        return (f"✨ Upscaling — {fmt_scale(scale)}×\n\n🎞 Frames: {d}/{t if t else '?'} "
                f"({int(d*100/t) if t else '…'}%)\n" +
                (f"⚡ Per-frame: {spf:.2f}s\n⏱ ETA: {fmt_time(eta)}\n" if spf > 0 else "⚡ Measuring...\n") +
                f"🕒 Elapsed: {fmt_time(time.time() - t_start)}\n❌ To stop: /cancel")

    try:
        if not is_gif:
            total = info["frames"] or max(1, int(info["duration"] * fps))
            dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(input_path), "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE)
            
            cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", f"{fps:.6f}", "-i", "pipe:0", "-i", str(input_path), "-map", "0:v:0"]
            if info.get("has_audio"): cmd += ["-map", "1:a?"]
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(output_path)]
            enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

            done, sum_t, f_bytes = 0, 0.0, w * h * 3
            while True:
                if cancel.is_set(): raise Exception("Cancelled by user")
                raw = dec.stdout.read(f_bytes)
                if not raw or len(raw) != f_bytes: break
                
                t0 = time.time()
                img = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
                out, _ = ups.enhance(img, outscale=scale)
                enc.stdin.write(out.tobytes())
                
                sum_t += time.time() - t0
                done += 1
                status.request(prog_txt(done, total, sum_t / done))
                if done % 60 == 0: gc.collect()
            
            enc.stdin.close(); dec.wait(); enc.wait()
            stats.update(frames=done, avg_spf=(sum_t / done if done else 0.0))
        else:
            fps_g = fps if fps > 0 else 10.0
            dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(input_path), "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE)
            frames = []
            while len(frames) < MAX_GIF_FRAMES:
                raw = dec.stdout.read(w*h*3)
                if not raw: break
                frames.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy())
            dec.wait()
            
            total = len(frames)
            ups_list, sum_t = [], 0.0
            for i, fr in enumerate(frames, 1):
                if cancel.is_set(): raise Exception("Cancelled")
                t0 = time.time()
                out, _ = ups.enhance(fr, outscale=scale)
                sum_t += time.time() - t0
                ups_list.append(out)
                status.request(prog_txt(i, total, sum_t / i))
            
            frames.clear(); gc.collect()
            loops = min(max(1, math.ceil(GIF_MIN_SEC / (total / fps_g))), max(1, 600 // total))
            
            enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", f"{fps_g:.6f}", "-i", "pipe:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)], stdin=subprocess.PIPE)
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
            except: pass


@app.on_message((filters.video | filters.document | filters.animation) & filters.private)
async def process_video(client, message: Message):
    global cancel_event
    if not is_owner(message): return
    
    async with busy_lock:
        if job_state["active"]:
            await message.reply_text("⏳ Ek video already process ho rahi hai.")
            return
        job_state["active"] = True
        cancel_event = threading.Event()
        
    job_dir = None
    try:
        media = message.video or message.document or message.animation
        filename = media.file_name or f"vid_{message.id}.mp4"
        is_gif = filename.lower().endswith(".gif") or getattr(media, "mime_type", "") == "image/gif"
        
        if not filename.lower().endswith((".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".gif")):
            await message.reply_text("❌ Unsupported file format.")
            return

        status_msg = await message.reply_text("📥 Video Downloading...")
        status = LiveStatus(status_msg, asyncio.get_running_loop())

        job_dir = WORK_DIR / f"job_{message.id}"
        job_dir.mkdir(exist_ok=True)
        in_path = job_dir / filename
        await app.download_media(message, file_name=str(in_path))

        # FFprobe
        probe = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(in_path)], capture_output=True)
        data = json.loads(probe.stdout)
        vid = next(s for s in data["streams"] if s.get("codec_type") == "video")
        
        # Calculate FPS & Frames safely
        fps_str = vid.get("avg_frame_rate", "0/1")
        fps = float(fps_str.split('/')[0]) / float(fps_str.split('/')[1]) if float(fps_str.split('/')[1]) else 30.0
        dur = float(vid.get("duration") or data.get("format", {}).get("duration") or 0)
        frames = int(float(vid.get("nb_frames") or dur * fps))

        info = {"width": int(vid["width"]), "height": int(vid["height"]), "fps": fps, "duration": dur, "frames": frames, "has_audio": any(s.get("codec_type") == "audio" for s in data["streams"])}
        
        if not is_gif and info["frames"] > MAX_FRAMES:
            await status.edit(f"❌ Video limit se badi hai ({info['frames']} frames). Max: {MAX_FRAMES}.")
            return

        await status.edit("✨ Processing shuru ho rahi hai...")
        out_path = OUTPUT_DIR / f"{Path(filename).stem}_upscaled.mp4"
        
        stats = await asyncio.to_thread(run_pipeline_sync, in_path, out_path, info, settings["scale"], status, cancel_event, is_gif)
        
        size_mb = out_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_SEND_MB: raise RuntimeError("File 2GB se badi ban gayi.")
        
        await status.edit(f"📤 Uploading... ({size_mb:.1f} MB)")
        cap = f"✅ Done!\nScale: {fmt_scale(stats['scale'])}×\nTime: {fmt_time(stats['seconds'])}"
        
        if is_gif: await app.send_animation(message.chat.id, str(out_path), caption=cap)
        else: await app.send_video(message.chat.id, str(out_path), caption=cap, supports_streaming=True)
        await status.edit("✅ Processed & Uploaded!")

    except Exception as e:
        log.error("Error: %s", e)
        try: await message.reply_text(f"🛑 Stopped/Failed: {str(e)[:200]}")
        except: pass
    finally:
        if job_dir: subprocess.run(["rm", "-rf", str(job_dir)])
        job_state["active"] = False
        cancel_event = None

# ============================================================
# 6. START APPLICATION ENGINE
# ============================================================
if __name__ == "__main__":
    log.info("🚀 Triggering Pyrogram Engine (app.run)...")
    app.run()
