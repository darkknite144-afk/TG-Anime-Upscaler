import json, logging, time
from typing import Any, Dict, Optional, List

log = logging.getLogger("anime-upscaler.archive")
MARKER = "⚙️ UPSCALER-STATE v1"

def normalize_channel_id(raw) -> List[Any]:
    """Supports both string usernames (@username) and numeric IDs."""
    raw = (raw or "").strip()
    if not raw: return []
    
    # Agar username hai (characters hain)
    if not raw.replace("-", "").isdigit():
        if not raw.startswith("@") and "t.me" not in raw:
            raw = "@" + raw
        return [raw]
        
    # Agar purely numeric ID hai
    try: v = int(raw)
    except ValueError: return [raw]
    
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
        self.state: Dict[str, Any] = {"scale": 2.0, "preset": "balanced", "audio": "keep",
                                      "model": "anime", "jobs_done": 0, "history": []}

    async def load(self):
        for cid in self.candidates:
            try:
                found = None
                async for msg in self.app.get_chat_history(cid, limit=100):
                    if msg.text and msg.text.startswith(MARKER):
                        found = msg; break
                self.channel_id = cid
                if found:
                    self.state_msg_id = found.id
                    try: self.state.update(json.loads(found.text.split("\n", 1)[1]))
                    except Exception: pass
                log.info("📚 Archive ready | channel=%s | state_msg=%s | jobs_done=%s",
                         cid, self.state_msg_id, self.state.get("jobs_done", 0))
                return
            except Exception as e:
                log.warning("⚠️ Channel id %s kaam nahi kari: %s", cid, e)
        log.error("❌ Archive FAIL — channel post bot ko forward karo, wo sahi ID dega")

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
