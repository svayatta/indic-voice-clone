"""A second training run on a voice starts clean, and a stray KeyError in a job is a failure.

Run 1 leaves transcripts and SNAC tokens under `work/`. Both are resumable
caches keyed by chunk file name, and run 2 writes new chunks with the same
names. Run 2 must not read them. A job that raises a plain KeyError must
show "failed" instead of leaving the status on "preparing".
"""
import pytest

pytest.importorskip("fastapi")

from voiceclone.studio.engine import Engine  # noqa: E402
from voiceclone.studio.jobs import Studio  # noqa: E402


class StubEngine:
    min_train_minutes = 8.0
    capabilities = set()

    def __init__(self, error=None):
        self.error = error

    def train(self, sample_wav, recordings, work, lang, log=print, clean_opts=None, opts=None):
        if self.error:
            raise self.error
        return work / "adapter"


def ready_voice(studio):
    vid = studio.store.create("v", "kn")["id"]

    def mark(v):
        v["status"] = "ready"
        v["recordings"] = [{"file": "a.wav", "sec": 60}]
    studio.store.update(vid, mark)
    return vid


def run_queue(studio):
    while not studio.q.empty():
        fn, *args = studio.q.get()
        studio.run_job(fn, *args)


def test_a_plain_keyerror_fails_the_training_and_the_next_run_is_accepted(tmp_path):
    engine = StubEngine(KeyError("R00_0003.wav"))
    studio = Studio(tmp_path, engine, start=False)
    vid = ready_voice(studio)
    studio.train(vid)
    run_queue(studio)
    t = studio.store.get(vid)["train"]
    assert t["status"] == "failed" and "R00_0003.wav" in t["error"]
    engine.error = None
    studio.train(vid)   # raises ValueError("training is already running") if the status is stuck


def test_a_voice_deleted_while_queued_is_ignored_silently(tmp_path):
    studio = Studio(tmp_path, StubEngine(), start=False)
    vid = ready_voice(studio)
    studio.train(vid)
    studio.store.delete(vid)
    run_queue(studio)   # must not raise
    assert studio.store.list() == []


def test_second_run_does_not_see_data_of_the_first(tmp_path, monkeypatch):
    import voiceclone.align, voiceclone.asr, voiceclone.audio, voiceclone.dataset
    import voiceclone.segment, voiceclone.speaker, voiceclone.train

    seen = []   # per run: stale files that a step found when it started

    def vad_segment(audio, sr, d, prefix=""):
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{prefix}_0000.wav").write_bytes(b"new audio")
        return "segmented"

    def transcribe_dir(d, lang, log, asr):
        seen.append(("transcripts", (d / "transcripts.jsonl").exists()))
        (d / "transcripts.jsonl").write_text("{}\n")

    def build_clips(chunk_dirs, out, *a, **k):
        seen.append(("snac", (out / "snac_tokens.json").exists()))
        out.mkdir(parents=True, exist_ok=True)
        (out / "snac_tokens.json").write_text("{}")
        (out / "metadata.jsonl").write_text('{"duration": 600}\n')
        return "ok"

    class Enc:
        def __init__(self, *a): pass
        def embed_file(self, p): return None

    monkeypatch.setattr(voiceclone.segment, "vad_segment", vad_segment)
    monkeypatch.setattr(voiceclone.asr, "transcribe_dir", transcribe_dir)
    monkeypatch.setattr(voiceclone.dataset, "build_clips", build_clips)
    monkeypatch.setattr(voiceclone.audio, "load_wav", lambda *a: None)
    monkeypatch.setattr(voiceclone.align, "Aligner", lambda: None)
    monkeypatch.setattr(voiceclone.speaker, "SpeakerEncoder", Enc)
    monkeypatch.setattr(voiceclone.speaker, "centroid", lambda x: None)
    monkeypatch.setattr(voiceclone.train, "finetune", lambda *a, **k: None)

    engine = Engine()
    monkeypatch.setattr(engine, "clean", lambda src, dst, opts=None: dst)
    monkeypatch.setattr(engine, "asr", lambda: None)
    work = tmp_path / "work"
    (work / "adapter").mkdir(parents=True)
    (work / "adapter" / "keep.txt").write_text("x")

    for _ in range(2):
        engine.train(tmp_path / "s.wav", [tmp_path / "r.wav"], work, "kn", log=lambda *_: None)

    assert seen == [("transcripts", False), ("snac", False)] * 2
    assert (work / "adapter" / "keep.txt").exists()   # only the data caches are removed
