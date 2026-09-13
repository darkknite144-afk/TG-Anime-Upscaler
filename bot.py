#!/usr/bin/env python3
"""
Smart Anime/Game Upscaler v9
- Secrets EXACTLY waise hi (koi change nahi) — sirf self-healing add hua
- 🔁 Session watchdog: updates rukhe to khud reconnect
- 📚 Archive auto-retry + forward-se-channel-set
- 🧠 Governor v2 + 💾 RAM-minimal pipeline + 🎌/🎮 models + 🖼 photo + ️ preview
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
from pyrogram import Client, filters
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

# ================= CONFIG (secrets untouched) =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("anime-upscaler")

API_ID = int(os.getenv("API_ID", "0") or 0)
API_HASH = (os.getenv("API_HASH", "") or "").strip()
BOT_TOKEN = (os.getenv("BOT_TOKEN", "") or "").strip()
OWNER_CHAT_ID = (os.getenv("OWNER_CHAT_ID", "") or "").strip()

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
    "game":   {"file": "realesr-general-x4v3.pth",  "arch": "srvgg", "label": "🎮 Game (Fast)"},
    "gamehq": {"file": "RealESRGAN_x4plus.pth",     "arch": "rrdb",  "label": "🎮 Game (HQ, slow)"},
}
PRESETS = {"fast": {"crf": "23", "preset": "veryfast"},
           "balanced": {"crf": "19", "preset": "veryfast"},
           "best": {"crf": "16", "preset": "slow"}}

settings = {"scale": 2.0, "preset": "balanced", "audio": "keep", "model": "anime"}
job_state = {"active": False}
current_job: Optional[Dict[str, Any]] = None
cancel_event: Optional[threading.Event] = None
LAST_UPDATE = {"ts": time.time()}

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

def bar(pct: float, n: int = 8) -> str:
    f = int(n * min(100, max(0, pct)) / 100)
    return "▰" * f + "▱" * (n - f)

def owner_ok(m) -> bool:
    cid = getattr(getattr(m, "chat", None), "id", None)
    ok = str(cid) == OWNER_CHAT_ID
    if not ok:
        log.warning("🚫 OWNER MISMATCH | aaya=%s | expected=%s", cid, OWNER_CHAT_ID)
    return ok

# ================= GOVERNOR v2 =================
CONFIGS = [(1, 4), (2, 2), (2, 3), (3, 1), (4, 1)]
DOWNGRADE = {(4, 1): (3, 1), (3, 1): (2, 2), (2, 3): (2, 2), (2, 2): (1, 2),
             (1, 4): (1, 2), (1, 2): (1, 1), (1, 1): (1, 1)}
PROBE, EXPLOIT = 6, 12

class Governor:
    def __init__(self, out_px: int):
        self.fp = out_px * 512 / 1e9
        self.ema = {c: 0.0 for c in CONFIGS}
        self.cnt = {c: 0 for c in CONFIGS}
        start = (2, 2) if self._ram_ok(2) else ((1, 2) if self._ram_ok(1) else (1, 1))
        self.current = start
        self.probe_left, self.exploit = PROBE, 0
        self.load_ema = load1(); self.ram_ema = mem_avail_gb()
        self.press = 0; self.idle = 0; self.safe = False
        self.lock = threading.Lock()
        self.apply()
        log.info("🧠 Governor v2 %s | fp %.2fGB/fr | RAM %.1fGB | load %.1f",
                 self.current, self.fp, self.ram_ema, self.load_ema)

    def _ram_ok(self, w: int) -> bool:
        return w * self.fp <= max(1.0, mem_avail_gb() * 0.7)

    def apply(self):
        torch.set_num_threads(self.current[1])

    def thr(self, c) -> float:
        return c[0] / self.ema[c] if self.ema[c] else 0.0

    def on_frame(self, cfg, dt: float, done: int):
        with self.lock:
            c = tuple(cfg)
            if c in self.ema:
                self.ema[c] = dt if self.cnt[c] == 0 else self.ema[c] * 0.7 + dt * 0.3
                self.cnt[c] += 1
            self.load_ema = self.load_ema * 0.8 + load1() * 0.2
            self.ram_ema = self.ram_ema * 0.8 + mem_avail_gb() * 0.2
            if done % 20 == 0:
                gc.collect()
            if self.ram_ema < 1.2 or self.load_ema > CPU_THREADS * 1.5:
                self.press += 1; self.idle = 0
                if self.press >= 2:
                    self.press = 0; self.safe = True
                    nxt = DOWNGRADE.get(self.current, self.current)
                    if nxt != self.current:
                        self.current = nxt; self.apply()
                        log.warning("🛡 DOWNGRADE -> %s (RAM %.1fGB, load %.1f)", nxt, self.ram_ema, self.load_ema)
                    return
            else:
                self.press = 0
                if self.safe and self.ram_ema > 3.0 and self.load_ema < CPU_THREADS * 0.9:
                    self.safe = False
                    log.info("🛡 safe-mode OFF (RAM %.1fGB)", self.ram_ema)
            if not self.safe and self.load_ema < CPU_THREADS * 0.55 and self.ram_ema > 4.0:
                self.idle += 1
                if self.idle >= 4:
                    self.idle = 0; self._probe(up=True); return
            else:
                self.idle = 0
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
            if target is None:
                self.probe_left = PROBE; return
        else:
            cand = [c for c in CONFIGS if self.cnt[c] < 3 and self._ram_ok(c[0])]
            if not cand:
                cand = [c for c in CONFIGS if self._ram_ok(c[0])]
            if not cand: return
            target = cand[0]
        if target != self.current:
            self.current = target; self.apply()
            log.info("🤖 AI probe -> %s%s", target, " (UPGRADE)" if up else "")
        self.probe_left = PROBE

    def _pick_best(self):
        tested = [c for c in CONFIGS if self.cnt[c] >= 3 and self._ram_ok(c[0])]
        if not tested: return
        b = max(tested, key=self.thr)
        if b != self.current:
            self.current = b; self.apply()
            log.info("🤖 AI best -> %s (%.2f f/s)", b, self.thr(b))

    def status(self, spf: float) -> str:
        return (f"🤖 {self.current[0]}W×{self.current[1]}T | {spf:.2f}s/fr | "
                f"{self.thr(self.current):.2f} f/s | 🛡 RAM {self.ram_ema:.1f}GB | "
                f"load {self.load_ema:.1f}" + (" | SAFE" if self.safe else ""))

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

# ================= PANEL =================
_panel: Optional[Message] = None

def panel_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(MODELS[settings["model"]]["label"], callback_data="b:mmenu"),
         InlineKeyboardButton(f"🎯 {fmt_scale(settings['scale'])}×", callback_data="b:qmenu"),
         InlineKeyboardButton(f"⚡ {settings['preset'].title()}", callback_data="b:pmenu"),
         InlineKeyboardButton(f"🔊 {settings['audio'].title()}", callback_data="b:amenu")],
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
    lines = [f"🎛 **Upscaler v9** • 🧠 {CPU_THREADS} cores • 🛡 {mem_avail_gb():.1f}GB free • load {load1():.1f}",
             f"Model: {MODELS[settings['model']]['label']} • {fmt_scale(settings['scale'])}× • "
             f"{settings['preset'].title()} • 🔊 {settings['audio'].title()}", ""]
    if job_state.get("active") and current_job:
        j = current_job
        if j.get("total"):
            pct = j["done"] * 100 / j["total"]
            lines += [f"{j.get('stage', '🎨')} **{j['filename'][:22]}** — {bar(pct)} {pct:.0f}%",
                      f"   {j['done']}/{j['total']} fr • ⏱ ETA {fmt_time(j.get('eta', 0))}",
                      "   " + j.get("ai", "")]
        else:
            lines += [f"{j.get('stage', '📥')} **{j.get('filename', '')[:22]}**"]
    else:
        lines += ["😴 Idle — video/GIF/photo bhejo. Model button se 🎌/🎮 chuno."]
    return "\n".join(lines)

async def ensure_panel(cid: int) -> Message:
    global _panel
    if _panel is None:
        _panel = await app.send_message(cid, panel_text(), reply_markup=panel_kb())
    return _panel

async def refresh_panel():
    global _panel
    try:
        if _panel:
            await _panel.edit_text(panel_text(), reply_markup=panel_kb())
    except Exception:
        pass

# ================= DEBUG + WATCHDOG (self-healing) =================
@app.on_message(filters.all, group=-2)
async def debug_incoming(client, message):
    LAST_UPDATE["ts"] = time.time()
    log.info("📥 INCOMING | chat=%s | text=%r", message.chat.id, (message.text or "")[:60])

async def _session_watchdog():
    """Updates ruk gaye ho to session khud reconnect karo (ghost-session fix)."""
    while True:
        await asyncio.sleep(90)
        if job_state.get("active"):
            LAST_UPDATE["ts"] = time.time()
            continue
        gap = time.time() - LAST_UPDATE["ts"]
        if gap > 240:
            log.warning("🔁 %d sec se koi update nahi — session reconnect...", int(gap))
            try:
                await app.stop()
                await asyncio.sleep(3)
                await app.start()
                LAST_UPDATE["ts"] = time.time()
                log.info("🔁 Reconnect OK — updates dobara listen ho rahi hain")
            except Exception as e:
                log.error("🔁 Reconnect fail: %s", e)

async def _archive_retry_loop():
    for i in range(1, 11):
        await asyncio.sleep(60)
        if archive is None or archive.channel_id:
            return
        log.info("📚 Archive retry #%d ...", i)
        await archive.load()
    if archive and not archive.channel_id:
        log.error("❌ Archive 10 retries ke baad bhi fail — channel ki post bot ko forward karo")

# ================= CALLBACKS =================
@app.on_callback_query(filters.regex(r"^b:"))
async def btn(client, cq):
    global _panel
    if not owner_ok(cq.message):
        await cq.answer("Private bot!", show_alert=True); return
    parts = cq.data[2:].split(":"); a = parts[0]; v = parts[1] if len(parts) > 1 else ""
    kb = panel_kb()
    if a == "mmenu": kb = model_kb()
    elif a == "qmenu": kb = q_kb()
    elif a == "pmenu": kb = p_kb()
    elif a == "amenu": kb = a_kb()
    elif a == "m":
        settings["model"] = v
        if archive: archive.state["model"] = v
        await cq.answer(f"Model: {MODELS[v]['label']}")
    elif a == "q":
        settings["scale"] = float(v)
        if archive: archive.state["scale"] = float(v)
        await cq.answer(f"Scale {v}×")
    elif a == "p":
        settings["preset"] = v
        if archive: archive.state["preset"] = v
        await cq.answer(f"Preset {v}")
    elif a == "a":
        settings["audio"] = v
        if archive: archive.state["audio"] = v
        await cq.answer(f"Audio {v}")
    elif a == "stop":
        if cancel_event: cancel_event.set()
        await cq.answer("⛔ Stop request")
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
    try:
        if _panel and cq.message.id == _panel.id:
            await _panel.edit_text(panel_text(), reply_markup=kb)
        else:
            await cq.message.edit_text(panel_text(), reply_markup=kb)
            _panel = cq.message
    except Exception:
        pass
    await cq.answer()

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

    def upscale_one(img, cfg):
        t0 = time.time()
        out, _ = ups.enhance(img, outscale=settings["scale"])
        dt = time.time() - t0
        with stats_lock:
            first = (stats["done"] == 0)
            stats["done"] += 1; stats["sum"] += dt
            job["done"] = stats["done"]
            job["spf"] = stats["sum"] / stats["done"]
            job["eta"] = (total - job["done"]) * job["spf"] / max(1, gov.current[0])
            job["ai"] = gov.status(job["spf"])
        if first:
            try: cv2.imwrite(str(prev_dir / "prev_out.png"), out)
            except Exception: pass
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
            if frames_raw:
                try: cv2.imwrite(str(prev_dir / "prev_in.png"), frames_raw[0])
                except Exception: pass
            outs = []
            for fr in frames_raw:
                if cancel.is_set(): raise RuntimeError("Cancelled")
                if time.time() - t_start > MAX_JOB_SEC: raise RuntimeError("Job time-limit cross")
                outs.append(upscale_one(fr, gov.current))
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
                for arr in outs:
                    enc.stdin.write(arr.tobytes())
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

async def _refresh_loop():
    hb = 0
    while True:
        await refresh_panel()
        hb += 1
        if hb % 24 == 0 and job_state.get("active") and current_job:
            log.info("💓 heartbeat | %s fr | RAM %.1fGB | load %.1f",
                     current_job.get("done", 0), mem_avail_gb(), load1())
        await asyncio.sleep(2.5)

@app.on_message((filters.video | filters.document | filters.animation) & filters.private)
async def media_handler(client, message: Message):
    global cancel_event, current_job
    if not owner_ok(message): return
    async with busy_lock:
        if job_state.get("active"):
            await message.reply_text("⏳ Ek job chal rahi hai."); return
        job_state["active"] = True
        cancel_event = threading.Event()
    job_dir = None; out_path = None
    refresher = asyncio.create_task(_refresh_loop())
    try:
        media = message.video or message.document or message.animation
        filename = getattr(media, "file_name", None) or f"video_{message.id}.mp4"
        mime = getattr(media, "mime_type", "") or ""
        is_gif = filename.lower().endswith(".gif") or mime == "image/gif"
        if not filename.lower().endswith((".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".gif")):
            await message.reply_text("❌ Sirf video/GIF bhejo."); return
        await ensure_panel(message.chat.id)
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
            await message.reply_text(f"❌ Video bahut lambi: {info['frames']} frames (max {MAX_FRAMES}).")
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
        current_job.update({"ow": ow, "oh": oh, "stage": "🎨", "ai": gov.status(0.0)})
        out_path = OUTPUT_DIR / f"{Path(filename).stem}_up_{message.id}.mp4"
        t0 = time.time()

        await asyncio.to_thread(run_pipeline, current_job, in_path, out_path, info,
                                ups, gov, cancel_event, is_gif, job_dir)
        current_job["stage"] = "⬆️"
        size_mb = out_path.stat().st_size / 1048576
        if size_mb > MAX_SEND_MB:
            raise RuntimeError(f"Output {size_mb:.0f}MB > 2GB limit")

        cap = (f"✅ **{filename}**\n{MODELS[model_key]['label']} • {fmt_scale(scale)}× → {ow}×{oh}\n"
               f"🎞 {current_job.get('frames_done', current_job['done'])} fr"
               + (f" (loop ×{current_job.get('loops', 1)})" if is_gif else "") +
               f" • ⚡ {current_job['spf']:.2f}s/fr • 🕒 {fmt_time(time.time() - t0)}\n"
               f"📦 {size_mb:.1f}MB" + (" ⚠️ 4K-cap" if capped else ""))

        def ul_cb(cur, tot, *a):
            current_job["stage"] = f"⬆️ {cur/1048576:.0f}/{tot/1048576:.0f}MB"

        comp = await asyncio.to_thread(make_compare, job_dir)
        if comp:
            try:
                await app.send_photo(message.chat.id, str(comp), caption="⬅️ Before | ➡️ After")
            except Exception: pass

        if is_gif:
            await send_with_retry(lambda: app.send_animation(
                message.chat.id, str(out_path), caption=cap, progress=ul_cb), "animation")
        else:
            await send_with_retry(lambda: app.send_video(
                message.chat.id, str(out_path), caption=cap,
                supports_streaming=True, progress=ul_cb), "video")
        if archive:
            if not archive.channel_id:
                await archive.load()
            await archive.archive_video(out_path, f"{filename} | {MODELS[model_key]['label']} | "
                                                  f"{fmt_scale(scale)}× | {fmt_time(time.time() - t0)}")
            archive.record_job(filename, scale, time.time() - t0, True,
                               extra={"scale": settings["scale"], "preset": settings["preset"],
                                      "audio": settings["audio"], "model": model_key})
            await archive.save_state()
        current_job["stage"] = "✅"
    except Exception as e:
        log.exception("Job failed")
        try: await message.reply_text(f"❌ Failed: {str(e)[:250]}")
        except Exception: pass
        if current_job: current_job["stage"] = "❌"
    finally:
        refresher.cancel()
        if job_dir: shutil.rmtree(job_dir, ignore_errors=True)
        if out_path and out_path.exists():
            try: out_path.unlink()
            except Exception: pass
        job_state["active"] = False
        current_job = None
        cancel_event = None
        gc.collect()
        await refresh_panel()

# ================= PHOTO =================
@app.on_message(filters.photo & filters.private)
async def photo_handler(client, message: Message):
    if not owner_ok(message): return
    async with busy_lock:
        if job_state.get("active"):
            await message.reply_text("⏳ Job chal rahi hai, photo baad me bhejo."); return
        job_state["active"] = True
    try:
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
        await send_with_retry(lambda: app.send_photo(
            message.chat.id, str(outp),
            caption=f"✅ Photo upscale {MODELS[model_key]['label']} • "
                    f"{fmt_scale(settings['scale'])}× • {w}×{h} → {out.shape[1]}×{out.shape[0]} • {dt:.1f}s"), "photo")
    except Exception as e:
        log.exception("Photo fail")
        try: await message.reply_text(f"❌ Photo fail: {str(e)[:200]}")
        except Exception: pass
    finally:
        job_state["active"] = False
        for f in WORK_DIR.glob(f"photo_{message.id}*"):
            try: f.unlink()
            except Exception: pass

# ================= COMMANDS / TEXT =================
@app.on_message(filters.command("stats") & filters.private)
async def stats_cmd(client, message: Message):
    if not owner_ok(message): return
    if not archive:
        await message.reply_text("ℹ️ Archive channel set nahi hai."); return
    st = archive.state
    hist = st.get("history", [])
    tot_t = sum(x.get("t", 0) for x in hist)
    lines = [f"📊 **Stats**\n✅ Jobs done: {st.get('jobs_done', 0)}",
             f"🕒 Total process time: {fmt_time(tot_t)}"]
    for x in hist[-3:]:
        lines.append(f"• {x.get('f', '?')[:24]} | {x.get('s', 0)}× | {fmt_time(x.get('t', 0))}")
    await message.reply_text("\n".join(lines))

@app.on_message(filters.command("cancel") & filters.private)
async def cancel_cmd(client, message: Message):
    if not owner_ok(message): return
    if cancel_event: cancel_event.set()
    await message.reply_text("🛑 Cancel request bhej di.")

@app.on_message(filters.forwarded & filters.private)
async def forward_id_handler(client, message: Message):
    """Channel post forward karo → archive turant usi channel par set ho jayega (secret change nahi)."""
    if not owner_ok(message): return
    src = getattr(message, "forward_from_chat", None)
    if src is not None and getattr(src, "id", None):
        if archive:
            archive.candidates = [src.id]
            archive.channel_id = src.id
            archive.state_msg_id = None
            asyncio.create_task(archive.save_state())
            log.info("📌 Archive channel forward se set: %s", src.id)
        await message.reply_text(f"📌 Channel ID set ho gayi: `{src.id}` — archive ab isi par chalega.")

@app.on_message(filters.text & filters.private & ~filters.command(["start", "stats", "cancel"]))
async def text_handler(client, message: Message):
    if not owner_ok(message): return
    t = (message.text or "").lower()
    if any(k in t for k in ["game", "free fire", "pubg", "bgmi"]):
        await message.reply_text("🎮 Game videos: panel me model button → Game (Fast) ya Game (HQ).")
    elif any(k in t for k in ["ram", "cpu", "load"]):
        await message.reply_text(f"🛡 RAM free: {mem_avail_gb():.1f}GB • load: {load1():.1f} • cores: {CPU_THREADS}")
    else:
        await message.reply_text("🤖 v9: video/GIF/photo bhejo. Panel se model 🎌/🎮, scale, preset, audio. "
                                 "/stats dekho, /cancel se roko. Channel post forward karo → archive set.")

@app.on_message(filters.command("start") & filters.private)
async def start_handler(client, message: Message):
    global _panel
    if not owner_ok(message): return
    try:
        if _panel: await _panel.delete()
    except Exception: pass
    _panel = None
    await ensure_panel(message.chat.id)
    await message.reply_text(f"ℹ️ Tumhara chat ID: `{message.chat.id}` (OWNER_CHAT_ID secret verify karne ke liye)")

# ================= BOOT =================
def notify_owner_startup():
    try:
        r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          json={"chat_id": OWNER_CHAT_ID,
                                "text": "✅ Upscaler v9 online!\n🔁 Self-healing session + 📚 archive auto-retry.\nPanel se control karo."},
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
                log.info("✅ Archive channel WRITE test OK")
            except Exception as e:
                log.error("❌ Archive channel WRITE FAIL: %s", e)
        else:
            log.warning("⚠️ Archive connect nahi hua — auto-retry shuru (har 60s, 10 baar)")
            asyncio.create_task(_archive_retry_loop())
    log.info("🚀 v9 ready (cores=%s)", CPU_THREADS)

async def _main():
    try:
        await app.start()
        log.info("🔌 Client started — ab boot...")
        await _boot()
        asyncio.create_task(_session_watchdog())
        await asyncio.Event().wait()
    finally:
        try: await app.stop()
        except Exception: pass

if __name__ == "__main__":
    try:
        r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook?drop_pending_updates=True", timeout=15)
        log.info("🧹 Webhook: %s", r.text[:120])
    except Exception as e:
        log.warning("Webhook delete fail: %s", e)
    asyncio.run(_main())
