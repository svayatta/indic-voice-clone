"""Every take the software generates must carry the watermark, and the check
must not accuse real recordings. These need torch and AudioSeal; CI runs them
in their own job (`watermark` in .github/workflows/ci.yml)."""
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("audioseal")
import torch  # noqa: E402

from voiceclone.watermark import Watermark  # noqa: E402

SR = 24000


def voice_like(seconds=4.0, seed=0) -> np.ndarray:
    """A gliding harmonic tone with syllable-rate loudness, close enough to
    speech for the detector, without putting a recording in the repository."""
    t = np.arange(int(seconds * SR)) / SR
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 25)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) ** 2
    x = 0.3 * x / np.abs(x).max() + 0.003 * np.random.default_rng(seed).normal(size=len(t))
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def wm():
    return Watermark()


def test_unmarked_audio_is_not_claimed(wm):
    # a false "made with pratidhvani" would accuse a genuine recording
    assert not wm.detect(voice_like(), SR).marked


def test_marked_audio_is_recognised_as_ours(wm):
    d = wm.detect(wm.embed(voice_like(), SR), SR)
    assert d.ours, d


def test_mark_is_quiet(wm):
    # the mark must not change how the clone sounds; AudioSeal sits near 26 dB below the voice
    x = voice_like()
    y = wm.embed(x, SR)
    assert 10 * np.log10((x ** 2).sum() / ((y - x) ** 2).sum()) > 20


def test_mark_survives_a_cropped_quieter_excerpt(wm):
    y = wm.embed(voice_like(6.0), SR)
    assert wm.detect(0.4 * y[SR: 4 * SR], SR).ours


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_mark_survives_mp3(wm, tmp_path):
    # clones get shared as MP3s; a mark that compression strips is no use
    import soundfile as sf

    from voiceclone.audio import load16
    from voiceclone.watermark import SR as WM_SR

    sf.write(tmp_path / "a.wav", wm.embed(voice_like(), SR), SR, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp_path / "a.wav", "-b:a", "32k", tmp_path / "a.mp3"], check=True)
    assert wm.detect(load16(tmp_path / "a.mp3"), WM_SR).ours


def test_every_generated_take_is_marked(wm, monkeypatch):
    """Synthesizer.generate is the one way audio comes out of the model: the
    studio, `synth`, `long-form` and the benchmark all call it. Whatever the
    codec decodes must leave it marked."""
    from voiceclone.svara.synth import Synthesizer
    from voiceclone.svara.tokens import EOS

    class Model:
        def generate(self, ids, **k):
            return torch.cat([ids, torch.tensor([[11] * 70 + [EOS]])], dim=1)

    class Codec:
        def decode(self, tokens):
            return voice_like()

    s = Synthesizer.__new__(Synthesizer)
    s.device, s.tok, s.model, s.codec, s.watermark, s._torch = "cpu", None, Model(), Codec(), wm, torch
    monkeypatch.setattr("voiceclone.svara.synth.build_prompt", lambda *a, **k: [1, 2, 3])
    wav, _ = s.generate("ನಮಸ್ಕಾರ", None, None)
    assert wm.detect(wav, SR).ours


@pytest.mark.parametrize("sentence_sec", [3.0, 0.5])
def test_studio_preview_of_several_sentences_is_marked(wm, tmp_path, sentence_sec):
    """The file the user plays is the join of the takes with unmarked pauses
    between them. The detector reports the share of the file that is marked,
    so the join must stay ours, also for very short sentences."""
    import soundfile as sf

    from voiceclone.studio.jobs import Studio

    class Engine:
        def synthesize(self, text, ref_wav, ref_text, lora=None, seed=0, params=None):
            return wm.embed(voice_like(sentence_sec, seed=len(text)), SR), 0   # as Synthesizer.generate returns it

    studio = Studio(tmp_path, Engine(), start=False)
    vid = studio.store.create("v", "kn")["id"]
    studio.store.update(vid, lambda v: v.update(status="ready", ref_text="x"))
    pid = studio.preview(vid, "ಒಂದು. ಎರಡು? ಮೂರು! ನಾಲ್ಕು।")[0]["id"]
    while not studio.q.empty():
        fn, *args = studio.q.get()
        fn(*args)
    wav, sr = sf.read(studio.store.dir(vid) / "previews" / f"{pid}.wav", dtype="float32")
    assert len(wav) > 4 * sentence_sec * SR   # four takes and the pauses
    d = wm.detect(wav, sr)
    assert d.ours, d
