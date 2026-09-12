import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from pyrogram import Client, filters
from pyrogram.types import Message
from realesrgan import RealESRGANer


# ============================================================
# CONFIG
# ============================================================

API_ID = int(os.getenv("API_ID", "0") or "0")
API_HASH = os.getenv("API_HASH", "").strip()
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OWNER_CHAT_ID = os.getenv("OWNER_CHAT_ID", "").strip()

MODEL_PATH = Path("weights/realesr-animevideov3.pth")
WORK_ROOT = Path("work")

SCALE = 2
TILE = 128

SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("anime-upscaler")


# ============================================================
# VALIDATION
# ============================================================

if not API_ID or not API_HASH or not BOT_TOKEN:
    raise RuntimeError(
        "Missing API_ID, API_HASH or BOT_TOKEN GitHub Secret."
    )

if not OWNER_CHAT_ID:
    raise RuntimeError(
        "Missing OWNER_CHAT_ID GitHub Secret."
    )


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


# ============================================================
# REAL-ESRGAN
# ============================================================

def load_upsampler():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Real-ESRGAN model not found: {MODEL_PATH}"
        )

    log.info("Loading Real-ESRGAN AnimeVideo-v3 on CPU...")

    model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=4,
        act_type="prelu",
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path=str(MODEL_PATH),
        model=model,
        tile=TILE,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=torch.device("cpu"),
    )

    log.info("Real-ESRGAN model loaded.")
    return upsampler


upsampler = load_upsampler()


# ============================================================
# SHELL / FFMPEG HELPERS
# ============================================================

def run_cmd(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(map(str, cmd))
            + "\n\n"
            + result.stderr[-4000:]
        )

    return result.stdout


def ffprobe_json(path: Path):
    output = run_cmd([
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        str(path),
    ])
    return json.loads(output)


def get_video_info(path: Path):
    data = ffprobe_json(path)

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )

    if not video_stream:
        raise RuntimeError("No video stream found.")

    audio_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"),
        None,
    )

    width = int(video_stream["width"])
    height = int(video_stream["height"])

    fps_text = (
        video_stream.get("avg_frame_rate")
        or video_stream.get("r_frame_rate")
        or "30/1"
    )

    try:
        num, den = fps_text.split("/")
        fps = float(num) / float(den) if float(den) else 30.0
    except Exception:
        fps = 30.0

    duration = float(
        video_stream.get("duration")
        or data.get("format", {}).get("duration")
        or 0
    )

    try:
        frames = int(video_stream.get("nb_frames") or 0)
    except Exception:
        frames = 0

    if frames <= 0:
        frames = max(1, round(duration * fps))

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "frames": frames,
        "has_audio": audio_stream is not None,
        "video_codec": video_stream.get("codec_name", "unknown"),
    }


def safe_name(name: str):
    return Path(name).name


def make_workdir():
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    path = WORK_ROOT / f"{int(time.time() * 1000)}"
    path.mkdir(parents=True, exist_ok=False)
    return path


# ============================================================
# VIDEO PROCESSING
# ============================================================

def extract_frames(input_video: Path, frames_dir: Path):
    frames_dir.mkdir(parents=True, exist_ok=True)

    run_cmd([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(input_video),
        "-vsync", "0",
        str(frames_dir / "frame_%08d.png"),
    ])


def upscale_frames(
    frames_dir: Path,
    output_frames_dir: Path,
    progress_callback,
):
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.png"))

    if not frame_paths:
        raise RuntimeError("FFmpeg extracted 0 frames.")

    total = len(frame_paths)

    log.info("Upscaling %d frames at %dx...", total, SCALE)

    for index, frame_path in enumerate(frame_paths, start=1):
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)

        if img is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")

        try:
            output, _ = upsampler.enhance(
                img,
                outscale=SCALE,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Real-ESRGAN failed on frame {index}/{total}: {exc}"
            ) from exc

        output_path = output_frames_dir / frame_path.name

        if not cv2.imwrite(str(output_path), output):
            raise RuntimeError(
                f"Could not write frame: {output_path}"
            )

        if progress_callback:
            progress_callback(index, total)


def encode_video(
    upscaled_frames_dir: Path,
    output_video: Path,
    fps: float,
):
    run_cmd([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-framerate", f"{fps:.12g}",
        "-i", str(upscaled_frames_dir / "frame_%08d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_video),
    ])


def merge_original_audio(
    upscaled_video: Path,
    original_video: Path,
    final_output: Path,
):
    run_cmd([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(upscaled_video),
        "-i", str(original_video),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        "-movflags", "+faststart",
        str(final_output),
    ])


def process_video(
    input_video: Path,
    workdir: Path,
    progress_callback,
):
    info = get_video_info(input_video)

    log.info(
        "Input: %dx%d | %.3f FPS | %.2fs | audio=%s | codec=%s",
        info["width"],
        info["height"],
        info["fps"],
        info["duration"],
        info["has_audio"],
        info["video_codec"],
    )

    frames_dir = workdir / "frames"
    upscaled_frames_dir = workdir / "upscaled_frames"
    silent_output = workdir / "upscaled_silent.mp4"
    final_output = workdir / "final_output.mp4"

    extract_frames(input_video, frames_dir)

    upscale_frames(
        frames_dir,
        upscaled_frames_dir,
        progress_callback,
    )

    encode_video(
        upscaled_frames_dir,
        silent_output,
        info["fps"],
    )

    merge_original_audio(
        silent_output,
        input_video,
        final_output,
    )

    if not final_output.exists() or final_output.stat().st_size == 0:
        raise RuntimeError("Final output video was not created.")

    return final_output, info


# ============================================================
# TELEGRAM HELPERS
# ============================================================

async def safe_reply(message: Message, text: str):
    try:
        return await message.reply_text(text)
    except Exception as exc:
        log.warning(
            "Could not reply to chat %s: %s",
            message.chat.id,
            exc,
        )
        return None


# ============================================================
# /START
# IMPORTANT:
# This handler is intentionally NOT owner-restricted.
# It lets us diagnose a wrong OWNER_CHAT_ID instead of silently
# ignoring /start.
# ============================================================

@app.on_message(filters.private & filters.command("start"))
async def start_handler(client: Client, message: Message):
    chat_id = str(message.chat.id)
    username = getattr(message.from_user, "username", None)

    log.info(
        "Received /start from chat_id=%s username=%s",
        chat_id,
        username,
    )

    matches = chat_id == OWNER_CHAT_ID

    await safe_reply(
        message,
        "✅ Anime Video Upscaler is online!\n\n"
        f"🆔 Your Chat ID: `{chat_id}`\n"
        f"🔐 OWNER_CHAT_ID match: "
        f"{'YES ✅' if matches else 'NO ❌'}\n\n"
        "Send a video to upscale it 2×."
    )

    if matches:
        try:
            await client.send_message(
                message.chat.id,
                "🟢 GitHub Action connected successfully.\n"
                "Waiting for your video..."
            )
        except Exception as exc:
            log.warning(
                "Post-/start status failed for chat %s: %s",
                chat_id,
                exc,
            )


# ============================================================
# /ID
# Gives the exact Telegram chat ID for OWNER_CHAT_ID.
# ============================================================

@app.on_message(filters.private & filters.command("id"))
async def id_handler(client: Client, message: Message):
    chat_id = str(message.chat.id)

    await safe_reply(
        message,
        "🆔 Your Telegram Chat ID is:\n"
        f"`{chat_id}`\n\n"
        "Put this exact number in GitHub Secret:\n"
        "`OWNER_CHAT_ID`"
    )

    log.info("Chat ID requested: %s", chat_id)


# ============================================================
# TEXT HANDLER
# ============================================================

@app.on_message(
    filters.private
    & filters.text
    & ~filters.command(["start", "id"])
)
async def text_handler(client: Client, message: Message):
    chat_id = str(message.chat.id)

    if chat_id != OWNER_CHAT_ID:
        await safe_reply(
            message,
            "❌ This bot is restricted to its configured owner."
        )
        log.warning(
            "Unauthorized text message from chat_id=%s",
            chat_id,
        )
        return

    await safe_reply(
        message,
        "👋 Bot is running.\n\n"
        "Send a video file and I will upscale it 2×."
    )


# ============================================================
# VIDEO / DOCUMENT HANDLER
# ============================================================

@app.on_message(
    filters.private & (filters.video | filters.document)
)
async def video_handler(client: Client, message: Message):
    chat_id = str(message.chat.id)

    if chat_id != OWNER_CHAT_ID:
        await safe_reply(
            message,
            "❌ Unauthorized chat.\n"
            "Send `/id` to see your Chat ID."
        )
        log.warning(
            "Unauthorized video attempt from chat_id=%s",
            chat_id,
        )
        return

    if message.video:
        file_name = message.video.file_name or "video.mp4"
    else:
        file_name = message.document.file_name or "video"
        extension = Path(file_name).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            await safe_reply(
                message,
                "❌ Unsupported file type.\n"
                "Please send MP4/MKV/MOV/WEBM/AVI/M4V."
            )
            return

    file_name = safe_name(file_name)

    if not Path(file_name).suffix:
        file_name += ".mp4"

    workdir = make_workdir()
    input_path = workdir / file_name

    await safe_reply(
        message,
        "📥 Video received.\n"
        "Downloading..."
    )

    try:
        log.info(
            "Downloading video: chat=%s file=%s",
            chat_id,
            file_name,
        )

        await message.download(file_name=str(input_path))

        if not input_path.exists():
            raise RuntimeError(
                "Telegram download did not create the input file."
            )

        size_mb = input_path.stat().st_size / (1024 * 1024)

        await safe_reply(
            message,
            f"✅ Download complete: {size_mb:.1f} MB\n"
            "🔍 Reading video information..."
        )

        loop = asyncio.get_running_loop()

        progress_state = {
            "last_time": 0.0,
            "last_percent": -1,
        }

        async def send_progress(index: int, total: int):
            percent = int(index * 100 / total)
            now = time.monotonic()

            # Telegram message throttling.
            # Always allow 100%.
            if (
                percent != 100
                and percent == progress_state["last_percent"]
            ):
                return

            if (
                percent != 100
                and now - progress_state["last_time"] < 5
            ):
                return

            progress_state["last_time"] = now
            progress_state["last_percent"] = percent

            await safe_reply(
                message,
                f"⚙️ Upscaling: {percent}% "
                f"({index}/{total} frames)\n"
                "🤖 Real-ESRGAN AnimeVideo-v3\n"
                f"📈 Scale: {SCALE}×"
            )

        def progress_sync(index: int, total: int):
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    send_progress(index, total)
                )
            )

        await safe_reply(
            message,
            "🚀 Starting Real-ESRGAN AnimeVideo-v3...\n"
            "CPU processing can take a while.\n"
            "Keep the GitHub Action running."
        )

        final_output, info = await asyncio.to_thread(
            process_video,
            input_path,
            workdir,
            progress_sync,
        )

        await safe_reply(
            message,
            "🎬 Upscaling finished.\n"
            "🔊 Restoring original audio..."
        )

        caption = (
            "✅ Upscaling complete!\n\n"
            f"📐 {info['width']}×{info['height']} → "
            f"{info['width'] * SCALE}×{info['height'] * SCALE}\n"
            f"🎞 FPS: {info['fps']:.3f}\n"
            f"🔊 Original audio: "
            f"{'kept' if info['has_audio'] else 'none'}\n"
            f"🤖 Real-ESRGAN AnimeVideo-v3 • {SCALE}×"
        )

        log.info(
            "Sending final video: %s",
            final_output,
        )

        await message.reply_video(
            video=str(final_output),
            caption=caption,
            supports_streaming=True,
        )

        log.info("Video sent successfully.")

    except Exception as exc:
        log.exception("Video processing failed")

        await safe_reply(
            message,
            "❌ Processing failed.\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            "Check the GitHub Actions log for the full traceback."
        )

    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        log.info("Cleaned workdir: %s", workdir)


# ============================================================
# MAIN
# ============================================================

async def main():
    log.info("Starting Telegram Anime Video Upscaler...")

    await app.start()

    me = await app.get_me()

    log.info(
        "Telegram bot connected: @%s (id=%s)",
        me.username,
        me.id,
    )

    log.info(
        "Configured OWNER_CHAT_ID=%s",
        OWNER_CHAT_ID,
    )

    log.info(
        "Bot started. Waiting for videos..."
    )

    # DO NOT send a startup message to OWNER_CHAT_ID here.
    # If the user has not opened /started the bot, Telegram can return
    # PEER_ID_INVALID. The /start handler sends the status after the
    # Telegram peer is known.
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bot stopped by user.")
