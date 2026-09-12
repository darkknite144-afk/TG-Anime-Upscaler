import asyncio
import gc
import json
import logging
import math
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pyrogram import Client, filters
from pyrogram.types import (
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# ============================================================
# CONFIG (exact env names - no spaces!)
# ============================================================
API_ID = int(os.getenv("API_ID", "0") or 0)
API_HASH = os.getenv("API_HASH", "") or ""
BOT_TOKEN = os.getenv("BOT_TOKEN", "") or ""
OWNER_CHAT_ID = (os.getenv("OWNER_CHAT_ID", "") or "").strip()

SCALE_OPTIONS = [1.5, 2.0, 3.0, 4.0]
MODEL_PATH = Path("weights/realesr-animevideov3.pth")
WORK_DIR = Path("work")
OUTPUT_DIR = Path("output")

MAX_FRAMES = int(os.getenv("MAX_FRAMES", "3600"))
MAX_GIF_FRAMES = 240
MAX_OUT_PIXELS = 3840 * 2160
GIF_MIN_SEC = 2.0
CPU_THREADS = os.cpu_count() or 4
torch.set_num_threads(CPU_THREADS)
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))

settings = {"scale": 2.0}
job_state = {"active": False, "text": "😴 Idle — koi job nahi."}
cancel_event = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("anime-upscaler")

if not API_ID or not API_HASH or not BOT_TOKEN or not OWNER_CHAT_ID:
    raise RuntimeError(
        "Missing GitHub Secrets. Required: "
        "API_ID, API_HASH, BOT_TOKEN, OWNER_CHAT_ID"
    )

log.info("OWNER_CHAT_ID configured: %r (numeric=%s)",
         OWNER_CHAT_ID, OWNER_CHAT_ID.lstrip("-").isdigit())

WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def fmt_scale(s: float) -> str:
    return f"{s:g}"


def fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def even(x: int) -> int:
    x = int(round(x))
    return x + (x % 2)


def est_sec_per_frame(out_pixels: int) -> float:
    return 4.5e-7 * out_pixels + 0.06


# ============================================================
# MODEL
# ============================================================
_models = {}


def get_upsampler(tile: int) -> RealESRGANer:
    if tile in _models:
        return _models[tile]
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    log.info("Loading Real-ESRGAN AnimeVideo-v3 (tile=%s, threads=%s)...", tile, CPU_THREADS)
    model = SRVGGNetCompact(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_conv=16, upscale=4, act_type="prelu",
    )
    ups = RealESRGANer(
        scale=4,
        model_path=str(MODEL_PATH),
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=torch.device("cpu"),
    )
    _models[tile] = ups
    return ups


def choose_tile(out_pixels: int) -> int:
    return 0 if out_pixels <= 2_600_000 else 320


def compute_out(w: int, h: int, scale: float):
    capped = False
    while scale > 1.0 and even(w * scale) * even(h * scale) > MAX_OUT_PIXELS:
        scale = max(1.0, scale - 0.5)
        capped = True
    return even(w * scale), even(h * scale), scale, capped


# ============================================================
# FFPROBE
# ============================================================
def run_cmd(cmd):
    log.info("CMD: %s", " ".join(map(str, cmd)))
    return subprocess.run(
        cmd, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def ffprobe_json(path: Path):
    result = run_cmd([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ])
    return json.loads(result.stdout)


def get_video_info(path: Path):
    data = ffprobe_json(path)
    video = next(s for s in data["streams"] if s.get("codec_type") == "video")
    audio = next((s for s in data["streams"] if s.get("codec_type") == "audio"), None)
    fps_value = video.get("avg_frame_rate") or "0/1"
    num, den = fps_value.split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    if fps <= 0:
        fps_value = video.get("r_frame_rate") or "30/1"
        num, den = fps_value.split("/")
        fps = float(num) / float(den) if float(den) else 30.0
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    frames = int(float(video.get("nb_frames") or max(0, round(duration * fps))))
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": fps,
        "duration": duration,
        "frames": frames,
        "has_audio": audio is not None,
        "codec": video.get("codec_name", "unknown"),
    }


def safe_stem(name: str):
    stem = Path(name).stem
    return "".join(c for c in stem if c not in '/\\\x00') or "video"


# ============================================================
# LIVE STATUS
# ============================================================
class LiveStatus:
    def __init__(self, msg, loop):
        self.msg = msg
        self.loop = loop
        self.last = 0.0
        self.last_text = ""

    def request(self, text: str, force: bool = False):
        job_state["text"] = text

        def _do():
            asyncio.create_task(self._edit(text, force))

        self.loop.call_soon_threadsafe(_do)

    async def _edit(self, text: str, force: bool):
        now = time.time()
        if not force and (now - self.last < 4 or text == self.last_text):
            return
        self.last = now
        self.last_text = text
        try:
            await self.msg.edit_text(text)
        except Exception:
            pass

    async def edit(self, text: str):
        job_state["text"] = text
        await self._edit(text, True)


class JobCancelled(Exception):
    pass


# ============================================================
# PIPELINE (zero temp files)
# ============================================================
def run_pipeline(input_path: Path, output_path: Path, info: dict,
                 scale: float, status: LiveStatus,
                 cancel: asyncio.Event, is_gif: bool) -> dict:
    w, h, fps = info["width"], info["height"], info["fps"]
    ow, oh, scale, _ = compute_out(w, h, scale)
    tile = choose_tile(ow * oh)
    ups = get_upsampler(tile)
    t_start = time.time()
    dec = enc = None
    stats = {"frames": 0, "avg_spf": 0.0, "seconds": 0.0,
             "out_w": ow, "out_h": oh, "loops": 1, "scale": scale}

    def progress_text(done, total, spf):
        pct = f"{done * 100 // total}%" if total else "…"
        eta = (total - done) * spf if (total and spf) else 0
        lines = [
            f"✨ Upscaling — {fmt_scale(scale)}× (Real-ESRGAN AnimeVideo-v3)",
            "",
            f"🎞 Frames: {done}/{total if total else '?'} ({pct})",
        ]
        if spf > 0:
            lines.append(f"⚡ Per-frame: {spf:.2f}s  ({1 / spf:.2f} frame/sec)")
            lines.append(f"⏱ ETA: {fmt_time(eta)}")
        else:
            lines.append("⚡ Per-frame time measure ho raha hai...")
        lines += [
            f"🕒 Elapsed: {fmt_time(time.time() - t_start)}",
            f"🖼 Output: {ow}×{oh} | 🧠 {CPU_THREADS} threads | tile {tile or 'full'}",
            "",
            "❌ Rokna ho to /cancel",
        ]
        return "\n".join(lines)

    try:
        if not is_gif:
            total = info["frames"] or max(1, int(info["duration"] * fps))
            dec = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-i", str(input_path),
                 "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                stdout=subprocess.PIPE,
            )
            cmd = ["ffmpeg", "-y", "-v", "error",
                   "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{ow}x{oh}", "-r", f"{fps:.6f}", "-i", "pipe:0",
                   "-i", str(input_path), "-map", "0:v:0"]
            if info["has_audio"]:
                cmd += ["-map", "1:a?"]
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
                    "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
                    "-movflags", "+faststart", str(output_path)]
            enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

            frame_bytes = w * h * 3
            done = 0
            sum_t = 0.0
            while True:
                if cancel.is_set():
                    raise JobCancelled("User cancelled")
                raw = dec.stdout.read(frame_bytes)
                if not raw or len(raw) != frame_bytes:
                    break
                t0 = time.time()
                img = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
                out, _ = ups.enhance(img, outscale=scale)
                enc.stdin.write(out.tobytes())
                sum_t += time.time() - t0
                done += 1
                status.request(progress_text(done, total, sum_t / done))
                if done % 60 == 0:
                    gc.collect()
                del img, out
            enc.stdin.close()
            dec.wait()
            enc.wait()
            if enc.returncode != 0:
                raise RuntimeError("FFmpeg encode failed")
            stats.update(frames=done, avg_spf=(sum_t / done if done else 0.0))
        else:
            fps_g = fps if fps > 0 else 10.0
            dec = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-i", str(input_path),
                 "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                stdout=subprocess.PIPE,
            )
            frame_bytes = w * h * 3
            raw_frames = []
            while len(raw_frames) < MAX_GIF_FRAMES:
                raw = dec.stdout.read(frame_bytes)
                if not raw or len(raw) != frame_bytes:
                    break
                raw_frames.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy())
            dec.wait()
            if not raw_frames:
                raise RuntimeError("GIF se frames nahi mile")
            total = len(raw_frames)
            ups_list = []
            sum_t = 0.0
            for i, fr in enumerate(raw_frames, start=1):
                if cancel.is_set():
                    raise JobCancelled("User cancelled")
                t0 = time.time()
                out, _ = ups.enhance(fr, outscale=scale)
                sum_t += time.time() - t0
                ups_list.append(out)
                status.request(progress_text(i, total, sum_t / i))
                del fr, out
            raw_frames.clear()
            gc.collect()
            loops = max(1, math.ceil(GIF_MIN_SEC / (total / fps_g)))
            loops = min(loops, max(1, 600 // total))
            enc = subprocess.Popen(
                ["ffmpeg", "-y", "-v", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{ow}x{oh}", "-r", f"{fps_g:.6f}", "-i", "pipe:0",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)],
                stdin=subprocess.PIPE,
            )
            for _ in range(loops):
                for fr in ups_list:
                    enc.stdin.write(fr.tobytes())
            enc.stdin.close()
            enc.wait()
            if enc.returncode != 0:
                raise RuntimeError("FFmpeg encode failed")
            stats.update(frames=total, avg_spf=(sum_t / total if total else 0.0), loops=loops)
        stats["seconds"] = time.time() - t_start
        return stats
    finally:
        for p in (dec, enc):
            try:
                if p and p.poll() is None:
                    p.kill()
                if p:
                    if p.stdout:
                        p.stdout.close()
                    if p.stdin:
                        p.stdin.close()
            except Exception:
                pass


def reencode_smaller(path: Path, crf: int) -> Path:
    tmp = path.with_name(path.stem + f"_c{crf}.mp4")
    run_cmd(["ffmpeg", "-y", "-v", "error", "-i", str(path),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-c:a", "copy",
             "-movflags", "+faststart", str(tmp)])
    return tmp


# ============================================================
# SMART REPLIES
# ============================================================
def smart_reply(text: str) -> str:
    t = text.lower()
    s = fmt_scale(settings["scale"])
    if any(k in t for k in ["hi", "hello", "hey", "namaste", "hlo", "yo "]):
        return ("🙏 Namaste boss! Main tumhara Anime Video Upscaler hoon.\n\n"
                "🎥 Video ya GIF bhejo — main upscale kar dunga.\n"
                f"🎯 Abhi quality: {s}× (badalne ke liye /quality)\n"
                "⏱ Har job me live progress + per-frame time + ETA dikhta hai.")
    if "kaise ho" in t or "how are you" in t:
        return ("💪 Ekdam badhiya! CPU thanda, RAM khali, aur Real-ESRGAN garam hai 🔥\n"
                "Tum batao — kaunsi video upscale karni hai aaj?")
    if any(k in t for k in ["quality", "scale", "kitni", "resolution", "select"]):
        return (f"🎯 Abhi quality: {s}× hai.\n\n"
                "Badalne ke liye /quality dabao aur button se chuno:\n"
                "1.5× / 2× / 3× / 4×\n\n"
                "Zyada scale = zyada detail, par zyada time.")
    if any(k in t for k in ["time", "eta", "estimate", "kitna waqt", "kitna time"]):
        return ("⏱ Jaise hi video aati hai, frames count karke main **estimate time** bata deta hoon.\n"
                "Phir pehle kuch frames ke baad **asli per-frame speed** se ETA live update hoti hai.\n"
                "Progress message me har frame ka time (sec/frame) dikhta rehta hai.")
    if "gif" in t:
        return ("🎞 GIF support hai boss!\n\n"
                "GIF bhejo → frames upscale hongi → phir **seamless loop video (≥2 sec)** banegi.\n"
                "Ye loop-wala rule sirf GIF par lagta hai, normal videos par nahi.")
    if "audio" in t or "sound" in t:
        return "🎧 Audio bilkul preserved rehta hai — original audio copy hoti hai, re-encode nahi."
    if "model" in t or "kaunsa" in t:
        return ("🧠 Model: **Real-ESRGAN AnimeVideo-v3** (anime ke liye best).\n"
                "CPU par optimized: adaptive tiling + single-pass pipe encoding.")
    if any(k in t for k in ["status", "progress", "kitna hua", "chal raha"]):
        return job_state["text"]
    if any(k in t for k in ["thank", "shukriya", "dhanyavad", "thx"]):
        return "😊 Arre boss, apna kaam hai! Aur video/GIF bhejte raho, upscale karta rahunga."
    if any(k in t for k in ["bye", "alvida", "good night", "gn"]):
        return "👋 Bye boss! Main yahin rahunga — jab bhi video bhejni ho, bhej dena."
    if any(k in t for k in ["help", "madad", "kya kar sakte", "command"]):
        return ("📚 Commands:\n"
                "/quality — upscale scale chuno\n"
                "/status — live job status\n"
                "/settings — bot settings\n"
                "/cancel — chal rahi job roko\n\n"
                "Bas video/GIF bhejo, baaki main sambhal lunga 😎")
    if any(k in t for k in ["love", "pyar", "jaan"]):
        return "😄 Pyar milta rahe boss! Badle me main 4K tak upscale kar deta hoon ❤️"
    variants = [
        (f"🤖 Haan boss, sun raha hoon! Main upscale bot hoon — baatein kam, kaam zyada 😄\n\n"
         f"🎯 Current quality: {s}×\n"
         "🎥 Video ya GIF bhejo → estimate + live progress + per-frame time sab chat me milega.\n"
         "🎛 Quality badalni ho to /quality."),
        (f"📝 Note kar liya! Waise main sirf ek kaam me expert hoon: **anime upscale** 🔥\n"
         f"Abhi setting: {s}× | Model: AnimeVideo-v3\n"
         "Video/GIF bhejo ya /quality se scale chuno."),
        ("🧐 Interesting! Par boss, meri specialty hai video upscaling.\n"
         "🎥 Media bhejo → main frames, estimate time, per-frame speed sab dikhata hoon.\n"
         "/help se poori list dekh lo."),
    ]
    return variants[len(text) % len(variants)]


# ============================================================
# TELEGRAM CLIENT
# ============================================================
app = Client(
    "telegram_anime_upscaler",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    workdir=".",
)
busy_lock = asyncio.Lock()


def owner_only(message: Message) -> bool:
    ok = str(message.chat.id) == OWNER_CHAT_ID
    if not ok:
        log.warning("OWNER MISMATCH | message chat=%s | secret OWNER_CHAT_ID=%r",
                    message.chat.id, OWNER_CHAT_ID)
    return ok


def mismatch_text(message: Message) -> str:
    return ("❌ This bot is private.\n\n"
            f"🔧 DEBUG: tumhara chat ID = `{message.chat.id}`\n"
            f"Secret me OWNER_CHAT_ID = `{OWNER_CHAT_ID or '(empty)'}` set hai.\n\n"
            "Agar dono alag hain → GitHub secret OWNER_CHAT_ID ko upar wale number par set karo, "
            "phir workflow **dobara run** karo (secrets sirf run ke start me load hote hain).")


# ---- DEBUG: har incoming message log hoga ----
@app.on_message(filters.all & filters.private, group=-1)
async def debug_logger(_, message: Message):
    kind = ("text" if message.text else
            "video" if message.video else
            "animation" if message.animation else
            "document" if message.document else "other")
    log.info("INCOMING | chat_id=%s | kind=%s | text=%r",
             message.chat.id, kind, (message.text or "")[:60])


# ============================================================
# COMMANDS
# ============================================================
@app.on_message(filters.command("start") & filters.private)
async def start_handler(_, message: Message):
    log.info("/start received from chat %s", message.chat.id)
    if not owner_only(message):
        await message.reply_text(mismatch_text(message))
        return
    await message.reply_text(
        "🎬 **Anime Video Upscaler Bot**\n\n"
        "Real-ESRGAN AnimeVideo-v3 se video/GIF upscale — GitHub CPU server par,\n"
        "memory/CPU safe tarike se, minimum time me.\n\n"
        "📛 Commands:\n"
        "/quality — upscale quality select karo\n"
        "/status — live job status\n"
        "/settings — bot settings\n"
        "/cancel — job roko\n"
        "/help — sab kuch samjho\n\n"
        "🎥 Video ya GIF bhejo:\n"
        "• Frames detect hote hi **estimate time**\n"
        "• Live **per-frame time + ETA + progress**\n"
        "• Audio preserved | GIF → loop video (≥2s)\n\n"
        f"🎯 Current quality: {fmt_scale(settings['scale'])}×",
        reply_markup=quality_keyboard(),
    )


@app.on_message(filters.command("help") & filters.private)
async def help_handler(_, message: Message):
    if not owner_only(message):
        await message.reply_text(mismatch_text(message))
        return
    await message.reply_text(smart_reply("help"))


@app.on_message(filters.command("quality") & filters.private)
async def quality_handler(_, message: Message):
    if not owner_only(message):
        await message.reply_text(mismatch_text(message))
        return
    await message.reply_text(
        "🎛 Upscale quality chuno:\n"
        f"(abhi: {fmt_scale(settings['scale'])}×)\n\n"
        "Zyada scale = zyada detail + zyada time.",
        reply_markup=quality_keyboard(),
    )


@app.on_callback_query(filters.create(lambda _, __, cq: bool(cq.data) and cq.data.startswith("scale:")))
async def scale_callback(_, cq):
    if str(cq.message.chat.id) != OWNER_CHAT_ID:
        await cq.answer("Private bot!", show_alert=True)
        return
    val = float(cq.data.split(":", 1)[1])
    settings["scale"] = val
    log.info("Scale set to %s by chat %s", val, cq.message.chat.id)
    await cq.answer(f"Quality: {fmt_scale(val)}×")
    try:
        await cq.message.edit_text(
            f"✅ Quality set: **{fmt_scale(val)}×**\n\n"
            "Ab jo video/GIF bhejoge wo isi scale par upscale hogi.",
            reply_markup=quality_keyboard(),
        )
    except Exception:
        pass


@app.on_message(filters.command("status") & filters.private)
async def status_handler(_, message: Message):
    if not owner_only(message):
        await message.reply_text(mismatch_text(message))
        return
    await message.reply_text(job_state["text"])


@app.on_message(filters.command("settings") & filters.private)
async def settings_handler(_, message: Message):
    if not owner_only(message):
        await message.reply_text(mismatch_text(message))
        return
    await message.reply_text(
        "⚙️ Bot settings:\n\n"
        f"🎯 Scale: {fmt_scale(settings['scale'])}×\n"
        f"🧠 Model: Real-ESRGAN AnimeVideo-v3 (CPU)\n"
        f"🧵 Threads: {CPU_THREADS}\n"
        f"🎞 Max frames: {MAX_FRAMES} (GIF: {MAX_GIF_FRAMES})\n"
        f"🖼 Max output: 4K (auto-cap)\n"
        f"🔁 GIF loop video: ≥{GIF_MIN_SEC:g}s\n"
        "💾 Pipeline: zero temp files (pipe decode→upscale→encode)"
    )


@app.on_message(filters.command("cancel") & filters.private)
async def cancel_handler(_, message: Message):
    if not owner_only(message):
        await message.reply_text(mismatch_text(message))
        return
    if job_state["active"] and cancel_event is not None:
        cancel_event.set()
        await message.reply_text("🛑 Cancel request bhej di — current frame ke baad ruk jayega.")
    else:
        await message.reply_text("😴 Abhi koi job chal nahi rahi.")


# ============================================================
# SMART TEXT REPLIES
# ============================================================
@app.on_message(filters.text & filters.private & ~filters.command(
    ["start", "help", "quality", "status", "settings", "cancel"]))
async def text_handler(_, message: Message):
    if not owner_only(message):
        await message.reply_text(mismatch_text(message))
        return
    txt = message.text or ""
    if txt.startswith("/"):
        await message.reply_text("🤔 Aisa koi command nahi hai boss. /help dekh lo.")
        return
    await message.reply_text(smart_reply(txt))


# ============================================================
# VIDEO / GIF HANDLER
# ============================================================
@app.on_message((filters.video | filters.document | filters.animation) & filters.private)
async def video_handler(_, message: Message):
    global cancel_event
    log.info("Media received from chat %s", message.chat.id)
    if not owner_only(message):
        await message.reply_text(mismatch_text(message))
        return
    async with busy_lock:
        if job_state["active"]:
            await message.reply_text("⏳ Ek video already process ho raha hai, thoda wait karo.")
            return
        job_state["active"] = True
        cancel_event = asyncio.Event()
    job_dir = None
    try:
        media = message.video or message.document or message.animation
        mime = getattr(media, "mime_type", "") or ""
        filename = media.file_name or f"video_{message.id}.mp4"
        is_gif = filename.lower().endswith(".gif") or mime == "image/gif"
        allowed = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".gif")
        if not filename.lower().endswith(allowed):
            await message.reply_text(
                "❌ Video ya GIF file bhejo.\n\nSupported:\nMP4 / MKV / MOV / WEBM / AVI / M4V / GIF")
            return
        status_msg = await message.reply_text("📥 Video received.\nDownloading...")
        loop = asyncio.get_running_loop()
        status = LiveStatus(status_msg, loop)

        job_name = f"job_{message.id}_{int(time.time())}"
        job_dir = WORK_DIR / job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / filename
        await app.download_media(message, file_name=str(input_path))

        info = await asyncio.to_thread(get_video_info, input_path)
        scale = settings["scale"]
        ow, oh, scale, capped = compute_out(info["width"], info["height"], scale)
        total = info["frames"] or max(0, round(info["duration"] * info["fps"]))
        if not is_gif and total > MAX_FRAMES:
            await status.edit(
                f"❌ Video bahut lambi hai: ~{total} frames.\n"
                f"Server safety limit: {MAX_FRAMES} frames.\n\n"
                "Chhoti video bhejo ya duration kam karo.")
            return
        est = (total or 1) * est_sec_per_frame(ow * oh)
        cap_note = "\n⚠️ Server safety ke liye scale auto-cap hua." if capped else ""
        await status.edit(
            "🔍 Video detect hua\n\n"
            f"📐 {info['width']}×{info['height']}\n"
            f"🎞 FPS: {info['fps']:.3f}\n"
            f"⏱ Duration: {info['duration']:.2f}s\n"
            f"🖼 Total frames: {total if total else 'counting...'}\n"
            f"🎧 Audio: {'Yes' if info['has_audio'] else 'No'}\n"
            f"🎬 Codec: {info['codec']}\n"
            f"{'🎞 GIF mode: loop video (≥2s) banega' if is_gif else ''}\n\n"
            f"🎯 Quality: {fmt_scale(scale)}× → output {ow}×{oh}\n"
            f"⏳ Estimated time: ~{fmt_time(est)} (±30%){cap_note}\n\n"
            "✨ Processing shuru... pehle frames ke baad asli speed + ETA dikhegi.")

        output_name = f"{safe_stem(filename)}_upscaled.mp4"
        output_path = OUTPUT_DIR / output_name
        stats = await asyncio.to_thread(
            run_pipeline, input_path, output_path, info,
            scale, status, cancel_event, is_gif,
        )
        await status.edit("🎞 Frames finished!\n\nEncoding + audio restore ho raha hai...")
        if not output_path.exists():
            raise RuntimeError("Output file nahi bani.")

        size_mb = output_path.stat().st_size / (1024 * 1024)
        for crf in (23, 27):
            if size_mb <= 48:
                break
            await status.edit(f"📦 Size {size_mb:.1f}MB zyada hai — compress kar raha hoon (crf {crf})...")
            new_path = await asyncio.to_thread(reencode_smaller, output_path, crf)
            output_path.unlink(missing_ok=True)
            new_path = new_path.rename(output_path)
            size_mb = output_path.stat().st_size / (1024 * 1024)

        caption = (
            "✅ Upscale complete!\n\n"
            "🧠 Model: Real-ESRGAN AnimeVideo-v3\n"
            f"🎯 Scale: {fmt_scale(stats['scale'])}× → {stats['out_w']}×{stats['out_h']}\n"
            f"🎞 Frames: {stats['frames']}"
            + (f" (loop ×{stats['loops']})" if is_gif else "") + "\n"
            f"⚡ Avg per-frame: {stats['avg_spf']:.2f}s\n"
            f"🕒 Total time: {fmt_time(stats['seconds'])}\n"
            f"📦 Size: {size_mb:.1f} MB\n"
            f"🎧 Audio: {'preserved' if info['has_audio'] and not is_gif else ('n/a (gif)' if is_gif else 'none')}"
        )
        await status.edit(f"📤 Upload ho raha hai... ({size_mb:.1f} MB)")
        if is_gif:
            await app.send_animation(
                chat_id=message.chat.id, animation=str(output_path), caption=caption)
        else:
            await app.send_video(
                chat_id=message.chat.id, video=str(output_path),
                caption=caption, supports_streaming=True)
        await status.edit("✅ Done!\n\nUpscaled result Telegram par bhej diya. 🎉")
    except JobCancelled:
        try:
            await message.reply_text("🛑 Job cancel kar di gayi.")
        except Exception:
            pass
    except Exception as exc:
        log.exception("Upscaling failed")
        try:
            await message.reply_text(f"❌ Upscaling failed:\n\n{type(exc).__name__}: {exc}")
        except Exception:
            pass
    finally:
        if job_dir:
            import shutil
            shutil.rmtree(job_dir, ignore_errors=True)
        for f in OUTPUT_DIR.glob("*.mp4"):
            try:
                f.unlink()
            except Exception:
                pass
        job_state["active"] = False
        job_state["text"] = "😴 Idle — koi job nahi."
        cancel_event = None


def quality_keyboard() -> InlineKeyboardMarkup:
    rows = []
    for sc in SCALE_OPTIONS:
        mark = "✅ " if abs(settings["scale"] - sc) < 0.01 else ""
        rows.append([InlineKeyboardButton(
            f"{mark}{fmt_scale(sc)}×", callback_data=f"scale:{sc}")])
    return InlineKeyboardMarkup(rows)


# ============================================================
# START
# ============================================================
async def main():
    await app.start()
    log.info("Bot started. Waiting for videos...")
    try:
        await app.send_message(
            OWNER_CHAT_ID,
            "✅ Anime Video Upscaler GitHub Action running!\n\n"
            "🎥 Video/GIF bhejo\n"
            "/quality se scale chuno\n"
            "⏱ Estimate + live per-frame time + ETA\n"
            "🎞 GIF → loop video (≥2s)\n\n"
            f"🎯 Default quality: {fmt_scale(settings['scale'])}×",
            reply_markup=quality_keyboard(),
        )
    except Exception as exc:
        log.warning("Startup message failed: %s", exc)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
