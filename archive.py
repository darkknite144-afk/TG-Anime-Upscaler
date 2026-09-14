import json, logging, time, asyncio
from typing import Any, Dict, Optional, List

try:
    from pyrogram.errors import FloodWait
except Exception:
    FloodWait = Exception  # fallback agar pyrogram import na ho

log = logging.getLogger("anime-upscaler.archive")

MARKER = "⚙️ UPSCALER-STATE v1"

# ===== Anti-flood knobs =====
STATE_MIN_EDIT_INTERVAL = 8.0     # seconds — do edits ke beech minimum gap
STATE_MAX_BODY_BYTES    = 3800    # ~ Telegram caption limit (4096) ke andar
SAVE_RETRY_ON_FLOOD     = True


def normalize_channel_id(raw) -> List[Any]:
    """Supports both string usernames (@username) and numeric IDs."""
    raw = (raw or "").strip()
    if not raw:
        return []

    # Agar username hai
    if not raw.replace("-", "").isdigit():
        if not raw.startswith("@") and "t.me" not in raw:
            raw = "@" + raw
        return [raw]

    # Agar purely numeric ID hai
    try:
        v = int(raw)
    except ValueError:
        return [raw]

    cands = [v]
    if v > 0:
        cands.append(-(1000000000000 + v))
    elif -1000000000000 < v < 0:
        cands.append(-(1000000000000 + abs(v)))
    return list(dict.fromkeys(cands))


class ChannelArchive:
    def __init__(self, app, channel_id: str):
        self.app = app
        self.candidates = normalize_channel_id(channel_id)
        self.channel_id: Optional[Any] = None
        if not self.candidates:
            log.warning("ARCHIVE_CHANNEL_ID set nahi/galat — archive OFF")

        self.state_msg_id: Optional[int] = None

        self.state: Dict[str, Any] = {
            "scale": 2.0, "preset": "balanced", "audio": "keep",
            "model": "anime", "jobs_done": 0, "history": []
        }

        # ===== Flood-safety trackers =====
        self._last_body: str = ""              # last text that was successfully saved
        self._last_edit_ts: float = 0.0        # time of last successful edit
        self._dirty: bool = False              # state changed but not yet persisted
        self._save_lock = asyncio.Lock()       # serialize saves
        self._flood_until: float = 0.0         # sleep until this ts if flood hit

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------
    async def load(self):
        for cid in self.candidates:
            try:
                chat = await self.app.get_chat(cid)
                self.channel_id = chat.id
                found = chat.pinned_message

                if found and found.text and found.text.startswith(MARKER):
                    self.state_msg_id = found.id
                    try:
                        self.state.update(json.loads(found.text.split("\n", 1)[1]))
                        # cache loaded body so pehla save same-text skip kar sake
                        self._last_body = found.text
                        self._last_edit_ts = time.time()
                    except Exception:
                        pass

                log.info("📚 Archive ready | channel=%s | state_msg=%s | jobs_done=%s",
                         self.channel_id, self.state_msg_id, self.state.get("jobs_done", 0))
                return

            except Exception as e:
                log.warning("⚠️ Channel id %s kaam nahi kari: %s", cid, e)

        log.error("❌ Archive FAIL — channel post bot ko forward karo, wo sahi ID dega")

    # ------------------------------------------------------------------
    # SERIALIZE — trim history if body gets too big
    # ------------------------------------------------------------------
    def _serialize(self) -> str:
        body = MARKER + "\n" + json.dumps(self.state, ensure_ascii=False, separators=(",", ":"))
        if len(body.encode("utf-8")) <= STATE_MAX_BODY_BYTES:
            return body

        # Body bahut badi — history trim karke retry
        trimmed = dict(self.state)
        h = list(trimmed.get("history", []))
        while h and len(body.encode("utf-8")) > STATE_MAX_BODY_BYTES:
            h = h[1:]
            trimmed["history"] = h
            body = MARKER + "\n" + json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
        self.state["history"] = h
        return body

    # ------------------------------------------------------------------
    # SAVE — rate-limited, same-text-skip, flood-aware
    # ------------------------------------------------------------------
    async def save_state(self, force: bool = False):
        """Save state to Telegram.

        - Skips if body identical to last saved body (avoids MESSAGE_NOT_MODIFIED).
        - Rate-limits edits to STATE_MIN_EDIT_INTERVAL seconds.
        - Handles FloodWait gracefully by deferring.
        - `force=True` bypasses rate-limit (for final saves).
        """
        if not self.channel_id:
            return

        async with self._save_lock:
            now = time.time()

            # Flood cooldown active? Skip quietly
            if now < self._flood_until:
                self._dirty = True
                return

            body = self._serialize()

            # Same text = skip (prevents MESSAGE_NOT_MODIFIED flood)
            if body == self._last_body:
                self._dirty = False
                return

            # Rate limit (unless forced)
            if not force and (now - self._last_edit_ts) < STATE_MIN_EDIT_INTERVAL:
                self._dirty = True
                return

            try:
                if self.state_msg_id:
                    await self.app.edit_message_text(
                        self.channel_id, self.state_msg_id, body
                    )
                else:
                    m = await self.app.send_message(self.channel_id, body)
                    self.state_msg_id = m.id
                    try:
                        await self.app.pin_chat_message(
                            self.channel_id, m.id, disable_notification=True
                        )
                    except Exception:
                        pass

                # Success
                self._last_body = body
                self._last_edit_ts = time.time()
                self._dirty = False

            except FloodWait as fw:
                wait_s = getattr(fw, "value", 30)
                self._flood_until = time.time() + wait_s + 2
                self._dirty = True
                log.warning("⏳ archive FloodWait %ss — cooldown set", wait_s)

            except Exception as e:
                err = str(e).lower()

                # These are benign — silently ignore
                if ("not modified" in err
                        or "message_id_invalid" in err
                        or "message to edit not found" in err
                        or "message is not modified" in err):
                    self._last_body = body
                    self._last_edit_ts = time.time()
                    self._dirty = False
                    return

                # Message deleted → forget msg_id, next save will re-create
                if "deleted" in err or "message_id_invalid" in err:
                    self.state_msg_id = None
                    self._last_body = ""
                    self._dirty = True
                    return

                log.warning("save_state fail: %s", str(e)[:180])
                self._dirty = True

    # ------------------------------------------------------------------
    # OPTIONAL: periodic flush (call from your main loop occasionally)
    # ------------------------------------------------------------------
    async def flush_if_dirty(self):
        """Call this from a slow background loop to persist deferred saves."""
        if self._dirty:
            await self.save_state(force=False)

    # ------------------------------------------------------------------
    # ARCHIVE VIDEO
    # ------------------------------------------------------------------
    async def archive_video(self, path, caption: str):
        if not self.channel_id:
            return
        try:
            await self.app.send_video(
                self.channel_id, str(path),
                caption="🗄 **ARCHIVE** | " + caption[:900],
                supports_streaming=True
            )
        except FloodWait as fw:
            wait_s = getattr(fw, "value", 30)
            log.warning("⏳ archive_video FloodWait %ss", wait_s)
            await asyncio.sleep(min(wait_s, 300))
        except Exception as e:
            log.warning("archive_video fail: %s", str(e)[:180])

    # ------------------------------------------------------------------
    # RECORD JOB
    # ------------------------------------------------------------------
    def record_job(self, filename, scale, seconds, ok=True, extra=None):
        if extra:
            for k in ("scale", "preset", "audio", "model", "core", "colorize"):
                if extra.get(k) is not None:
                    self.state[k] = extra[k]

        self.state["jobs_done"] = int(self.state.get("jobs_done", 0)) + (1 if ok else 0)
        h = self.state.setdefault("history", [])
        h.append({
            "f": filename[:40],
            "s": scale,
            "t": int(seconds),
            "ok": ok,
            "ts": int(time.time()),
        })
        self.state["history"] = h[-15:]
        self._dirty = True
