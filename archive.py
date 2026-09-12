import json, logging, time
from typing import Any, Dict, Optional
log = logging.getLogger("anime-upscaler.archive")
MARKER = "⚙️ UPSCALER-STATE v1"

class ChannelArchive:
    def __init__(self, app, channel_id: str):
        self.app = app; self.channel_id = None
        raw = (channel_id or "").strip()
        if raw:
            try: self.channel_id = int(raw)
            except ValueError: log.error("ARCHIVE_CHANNEL_ID numeric nahi: %r", raw)
        else: log.warning("ARCHIVE_CHANNEL_ID set nahi — archive OFF")
        self.state_msg_id = None
        self.state = {"scale": 2.0, "preset": "balanced", "audio": "keep",
                      "model": "anime", "jobs_done": 0, "history": []}
    async def load(self):
        if not self.channel_id: return
        try:
            async for msg in self.app.get_chat_history(self.channel_id, limit=100):
                if msg.text and msg.text.startswith(MARKER):
                    self.state_msg_id = msg.id
                    try: self.state.update(json.loads(msg.text.split("\n", 1)[1]))
                    except Exception: pass
                    break
            log.info("📚 Archive ready | jobs_done=%s", self.state.get("jobs_done", 0))
        except Exception as e:
            log.error("❌ Archive load FAIL: %s (bot ko channel admin banao!)", e)
    async def save_state(self):
        if not self.channel_id: return
        body = MARKER + "\n" + json.dumps(self.state, ensure_ascii=False)
        try:
            if self.state_msg_id:
                await self.app.edit_message_text(self.channel_id, self.state_msg_id, body)
            else:
                m = await self.app.send_message(self.channel_id, body)
                self.state_msg_id = m.id
                try: await self.app.pin_chat_message(self.channel_id, m.id, disable_notification=True)
                except Exception: pass
        except Exception as e: log.error("save_state FAIL: %s", e)
    async def archive_video(self, path, caption: str):
        if not self.channel_id: return
        try: await self.app.send_video(self.channel_id, str(path), caption="🗄 **ARCHIVE** | " + caption, supports_streaming=True)
        except Exception as e: log.error("archive video FAIL: %s", e)
    def record_job(self, filename, scale, seconds, ok=True, extra=None):
        if extra:
            for k in ("scale", "preset", "audio", "model"):
                if extra.get(k) is not None: self.state[k] = extra[k]
        self.state["jobs_done"] = int(self.state.get("jobs_done", 0)) + (1 if ok else 0)
        h = self.state.setdefault("history", [])
        h.append({"f": filename[:40], "s": scale, "t": int(seconds), "ok": ok, "ts": int(time.time())})
        self.state["history"] = h[-15:]
