"""Voices on disk: one folder per voice under ROOT/voices/, its state in voice.json.

voice.json is the only state; the server can restart at any time and pick up
where it was (a job that was running when it stopped is marked failed).
"""
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

_ID = re.compile(r"^[0-9a-f]{12}$")


class VoiceDeleted(KeyError):
    """The voice does not exist (any more)."""


class VoiceStore:
    def __init__(self, root):
        self.root = Path(root) / "voices"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def dir(self, vid: str) -> Path:
        if not _ID.match(vid or ""):
            raise VoiceDeleted(vid)
        d = self.root / vid
        if not (d / "voice.json").exists():
            raise VoiceDeleted(vid)
        return d

    def create(self, name: str, lang: str) -> dict:
        vid = uuid.uuid4().hex[:12]
        d = self.root / vid
        d.mkdir()
        v = {"id": vid, "name": name, "lang": lang, "created": time.time(),
             "status": "uploaded", "error": None, "sample_sec": None,
             "ref_text": None, "previews": [], "recordings": [],
             "train": {"status": "none", "error": None, "log": [], "adapter": None, "speech_min": None}}
        self._write(d, v)
        return v

    def get(self, vid: str) -> dict:
        return json.loads((self.dir(vid) / "voice.json").read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        out = []
        for p in self.root.glob("*/voice.json"):
            out.append(json.loads(p.read_text(encoding="utf-8")))
        return sorted(out, key=lambda v: v["created"], reverse=True)

    def update(self, vid: str, fn) -> dict:
        """Apply fn(voice) under the lock and save; returns the new voice."""
        with self._lock:
            d = self.dir(vid)
            v = json.loads((d / "voice.json").read_text(encoding="utf-8"))
            fn(v)
            self._write(d, v)
            return v

    def delete(self, vid: str) -> None:
        with self._lock:
            shutil.rmtree(self.dir(vid))

    def fail_interrupted(self) -> None:
        """Jobs that were running when the server stopped will never finish."""
        busy = {"queued", "cleaning", "transcribing"}
        for v in self.list():
            def fix(v):
                if v["status"] in busy:
                    v["status"], v["error"] = "failed", "the server stopped during processing; upload again"
                if v["train"]["status"] in {"queued", "preparing", "training"}:
                    v["train"]["status"], v["train"]["error"] = "failed", "the server stopped during training"
                for p in v["previews"]:
                    if p["status"] in {"queued", "running"}:
                        p["status"] = "failed"
            self.update(v["id"], fix)

    @staticmethod
    def _write(d: Path, v: dict) -> None:
        tmp = d / "voice.json.tmp"
        tmp.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(d / "voice.json")
