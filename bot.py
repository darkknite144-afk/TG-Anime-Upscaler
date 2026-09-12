import asyncio
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import torch
from pyrogram import Client, filters
from pyrogram.types import Message
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


# ============================================================
# CONFIG
# ============================================================

API_ID = int(os.getenv("API_ID", "0") or "0")
API_HASH = os.getenv("API_HASH", "")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
OWNER_CHAT_ID = os.getenv("OWNER_CHAT_ID", "").strip()

# Default upscale
SCALE = 2

# AnimeVideo-v3
MODEL_PATH = Path("weights/realesr-animevideov3.pth")

# Temporary folders
WORK_DIR = Path("work")
OUTPUT_DIR = Path("output")

# Smaller tile = lower RAM usage, but slower
TILE = int(os.getenv("ESRGAN_TILE", "128"))


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("anime-upscaler")


# ============================================================
# CHECK SECRETS
# ============================================================

if not API_ID or not API_HASH or not BOT_TOKEN or not OWNER_CHAT_ID:
    raise RuntimeError(
        "Missing GitHub Secrets. Required: "
        "API_ID, API_HASH, BOT_TOKEN, OWNER_CHAT_ID"
    )


WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# REAL-ESRGAN MODEL
# ============================================================

upsampler = None


def load_model():

    global upsampler

    if upsampler is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
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

        # CPU
        half=False,
        device=torch.device("cpu"),
    )

    log.info("Real-ESRGAN model loaded.")


# ============================================================
# COMMAND HELPER
# ============================================================

def run_cmd(cmd):

    log.info(
        "CMD: %s",
        " ".join(map(str, cmd))
    )

    return subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


# ============================================================
# FFPROBE
# ============================================================

def ffprobe_json(path: Path):

    import json

    result = run_cmd([
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ])

    return json.loads(result.stdout)


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(path: Path):

    data = ffprobe_json(path)

    video = next(
        s for s in data["streams"]
        if s.get("codec_type") == "video"
    )

    audio = next(
        (
            s
            for s in data["streams"]
            if s.get("codec_type") == "audio"
        ),
        None,
    )

    fps_value = video.get("avg_frame_rate") or "0/1"

    num, den = fps_value.split("/")

    fps = (
        float(num) / float(den)
        if float(den)
        else 0.0
    )

    if fps <= 0:

        fps_value = (
            video.get("r_frame_rate")
            or "30/1"
        )

        num, den = fps_value.split("/")

        fps = (
            float(num) / float(den)
            if float(den)
            else 30.0
        )

    duration = float(
        video.get("duration")
        or data.get("format", {}).get("duration")
        or 0
    )

    frames = int(
        float(
            video.get("nb_frames")
            or max(
                0,
                round(duration * fps)
            )
        )
    )

    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": fps,
        "duration": duration,
        "frames": frames,
        "has_audio": audio is not None,
        "codec": video.get(
            "codec_name",
            "unknown"
        ),
    }


# ============================================================
# SAFE FILENAME
# ============================================================

def safe_stem(name: str):

    stem = Path(name).stem

    return "".join(
        c
        for c in stem
        if c not in '/\\\x00'
    ) or "video"


# ============================================================
# EXTRACT VIDEO FRAMES
# ============================================================

def extract_frames(
    input_path: Path,
    frames_dir: Path,
    fps: float,
):

    frames_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    run_cmd([
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),

        "-vsync",
        "0",

        "-q:v",
        "1",

        str(
            frames_dir /
            "frame_%08d.png"
        ),
    ])

    return sorted(
        frames_dir.glob(
            "frame_*.png"
        )
    )


# ============================================================
# UPSCALE FRAMES
# ============================================================

def upscale_frames(
    frames,
    frames_dir: Path,
    update_cb=None,
):

    total = len(frames)

    for index, frame_path in enumerate(
        frames,
        start=1
    ):

        img = cv2.imread(
            str(frame_path),
            cv2.IMREAD_COLOR
        )

        if img is None:
            raise RuntimeError(
                f"Could not read frame: "
                f"{frame_path}"
            )

        # Real-ESRGAN
        output, _ = upsampler.enhance(
            img,
            outscale=SCALE
        )

        out_path = (
            frames_dir /
            f"up_{index:08d}.png"
        )

        ok = cv2.imwrite(
            str(out_path),
            output,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                1,
            ],
        )

        if not ok:
            raise RuntimeError(
                f"Could not write frame: "
                f"{out_path}"
            )

        if update_cb:
            update_cb(
                index,
                total
            )

        del img
        del output


# ============================================================
# ENCODE FINAL VIDEO
# ============================================================

def encode_video(
    frames_dir: Path,
    output_path: Path,
    input_path: Path,
    fps: float,
):

    silent_video = (
        output_path.with_name(
            output_path.stem +
            "_silent.mp4"
        )
    )

    # Frames → video
    run_cmd([
        "ffmpeg",
        "-y",

        "-framerate",
        f"{fps:.12f}",

        "-i",
        str(
            frames_dir /
            "up_%08d.png"
        ),

        "-c:v",
        "libx264",

        "-preset",
        "slow",

        "-crf",
        "16",

        "-pix_fmt",
        "yuv420p",

        str(silent_video),
    ])

    # Processed video + ORIGINAL audio
    run_cmd([
        "ffmpeg",
        "-y",

        "-i",
        str(silent_video),

        "-i",
        str(input_path),

        "-map",
        "0:v:0",

        "-map",
        "1:a?",

        "-c:v",
        "copy",

        "-c:a",
        "copy",

        "-shortest",

        str(output_path),
    ])

    silent_video.unlink(
        missing_ok=True
    )


# ============================================================
# CLEANUP
# ============================================================

def cleanup_dir(path: Path):

    if path.exists():

        shutil.rmtree(
            path,
            ignore_errors=True
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


busy = False
busy_lock = asyncio.Lock()


# ============================================================
# OWNER CHECK
# ============================================================

def owner_only(message: Message):

    return (
        str(message.chat.id)
        == OWNER_CHAT_ID
    )


# ============================================================
# /START
# ============================================================

@app.on_message(
    filters.command("start")
    & filters.private
)
async def start_handler(
    _,
    message: Message
):

    if not owner_only(message):

        await message.reply_text(
            "❌ This bot is private."
        )

        return

    await message.reply_text(
        "✅ Anime Video Upscaler ready.\n\n"

        "MP4/video bhejo.\n\n"

        "Model: Real-ESRGAN AnimeVideo-v3\n"
        "Scale: 2×\n"
        "FPS: automatic\n"
        "Audio: preserved"
    )


# ============================================================
# VIDEO HANDLER
# ============================================================

@app.on_message(
    (
        filters.video
        | filters.document
    )
    & filters.private
)
async def video_handler(
    _,
    message: Message
):

    global busy

    if not owner_only(message):

        await message.reply_text(
            "❌ This bot is private."
        )

        return


    # One video at a time
    async with busy_lock:

        if busy:

            await message.reply_text(
                "⏳ Ek video already "
                "process ho raha hai."
            )

            return

        busy = True


    job_dir = None


    try:

        # ------------------------------------------------
        # GET FILE
        # ------------------------------------------------

        media = (
            message.video
            or message.document
        )

        filename = (
            media.file_name
            or f"video_{message.id}.mp4"
        )


        allowed = (
            ".mp4",
            ".mkv",
            ".mov",
            ".webm",
            ".avi",
            ".m4v",
        )

        if not filename.lower().endswith(
            allowed
        ):

            await message.reply_text(
                "❌ Video file bhejo.\n\n"
                "Supported:\n"
                "MP4 / MKV / MOV / WEBM / AVI / M4V"
            )

            return


        # ------------------------------------------------
        # STATUS
        # ------------------------------------------------

        status = await message.reply_text(
            "📥 Video received.\n"
            "Downloading..."
        )


        # ------------------------------------------------
        # JOB FOLDER
        # ------------------------------------------------

        job_name = (
            f"job_{message.id}_"
            f"{int(time.time())}"
        )

        job_dir = (
            WORK_DIR /
            job_name
        )

        job_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        input_path = (
            job_dir /
            filename
        )


        # ------------------------------------------------
        # DOWNLOAD
        # ------------------------------------------------

        await app.download_media(
            message,
            file_name=str(
                input_path
            ),
        )


        # ------------------------------------------------
        # VIDEO INFO
        # ------------------------------------------------

        info = await asyncio.to_thread(
            get_video_info,
            input_path
        )


        await status.edit_text(

            "🔍 Video detected\n\n"

            f"📐 {info['width']}×"
            f"{info['height']}\n"

            f"🎞 FPS: "
            f"{info['fps']:.3f}\n"

            f"⏱ Duration: "
            f"{info['duration']:.2f}s\n"

            f"🎧 Audio: "
            f"{'Yes' if info['has_audio'] else 'No'}\n"

            f"🎬 Codec: "
            f"{info['codec']}\n\n"

            "✨ Real-ESRGAN "
            "AnimeVideo-v3 2× processing..."
        )


        # ------------------------------------------------
        # LOAD MODEL
        # ------------------------------------------------

        await asyncio.to_thread(
            load_model
        )


        # ------------------------------------------------
        # EXTRACT FRAMES
        # ------------------------------------------------

        frames_dir = (
            job_dir /
            "frames"
        )

        await asyncio.to_thread(
            extract_frames,
            input_path,
            frames_dir,
            info["fps"],
        )


        frames = sorted(
            frames_dir.glob(
                "frame_*.png"
            )
        )


        if not frames:

            raise RuntimeError(
                "No frames were extracted."
            )


        # ------------------------------------------------
        # PROGRESS
        # ------------------------------------------------

        last_percent = -1


        async def send_progress(
            index,
            total
        ):

            nonlocal last_percent

            percent = int(
                index * 100 / total
            )

            if (
                percent
                >= last_percent + 10
                or percent == 100
            ):

                last_percent = percent

                try:

                    await status.edit_text(

                        "✨ Upscaling "
                        "AnimeVideo-v3 2×\n\n"

                        f"Progress: "
                        f"{percent}%\n"

                        f"Frames: "
                        f"{index}/{total}\n\n"

                        "⏳ GitHub CPU "
                        "processing..."
                    )

                except Exception:

                    pass


        loop = (
            asyncio.get_running_loop()
        )


        def progress_sync(
            index,
            total
        ):

            loop.call_soon_threadsafe(

                lambda:
                asyncio.create_task(
                    send_progress(
                        index,
                        total
                    )
                )
            )


        # ------------------------------------------------
        # UPSCALE
        # ------------------------------------------------

        await asyncio.to_thread(

            upscale_frames,

            frames,
            frames_dir,

            progress_sync,
        )


        # ------------------------------------------------
        # OUTPUT
        # ------------------------------------------------

        output_name = (
            f"{safe_stem(filename)}"
            "_upscaled.mp4"
        )

        output_path = (
            OUTPUT_DIR /
            output_name
        )


        await status.edit_text(
            "🎞 Frames finished.\n\n"
            "Encoding video + "
            "restoring original audio..."
        )


        # ------------------------------------------------
        # ENCODE
        # ------------------------------------------------

        await asyncio.to_thread(

            encode_video,

            frames_dir,
            output_path,
            input_path,
            info["fps"],
        )


        # ------------------------------------------------
        # SEND
        # ------------------------------------------------

        size_mb = (
            output_path.stat().st_size
            / (1024 * 1024)
        )


        await status.edit_text(

            "📤 Uploading result "
            "to Telegram...\n\n"

            f"Size: {size_mb:.1f} MB"
        )


        caption = (

            "✅ Upscale complete\n\n"

            "Model: "
            "Real-ESRGAN AnimeVideo-v3\n"

            "Scale: 2×\n"

            f"FPS: {info['fps']:.3f}\n"

            "Audio: "
            f"{'preserved' if info['has_audio'] else 'none'}"
        )


        await app.send_video(

            chat_id=message.chat.id,

            video=str(output_path),

            caption=caption,

            supports_streaming=True,
        )


        await status.edit_text(

            "✅ Done!\n\n"
            "Upscaled video "
            "Telegram par bhej diya."
        )


    except Exception as exc:

        log.exception(
            "Upscaling failed"
        )

        try:

            await message.reply_text(

                "❌ Upscaling failed:\n\n"

                f"{type(exc).__name__}: "
                f"{exc}"
            )

        except Exception:

            pass


    finally:

        if job_dir:

            cleanup_dir(
                job_dir
            )

        busy = False


# ============================================================
# START BOT
# ============================================================

async def main():

    await app.start()

    log.info(
        "Bot started. "
        "Waiting for videos..."
    )


    try:

        await app.send_message(

            OWNER_CHAT_ID,

            "✅ Anime Video Upscaler "
            "GitHub Action is running!\n\n"

            "Video bhejo.\n\n"

            "Model: Real-ESRGAN "
            "AnimeVideo-v3\n"

            "Scale: 2×"
        )

    except Exception as exc:

        log.warning(
            "Startup message failed: %s",
            exc
        )


    # Keep GitHub Action alive
    await asyncio.Event().wait()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())
