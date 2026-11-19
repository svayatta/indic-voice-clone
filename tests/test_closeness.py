"""The closeness report must say *in what way* a clone differs from the speaker.

Each test changes one thing about a recording and checks that the report
blames that dimension and no other. A report that called every degraded copy
"far" on every dimension would pass a single-score test, but would not tell
anyone what to fix.
"""
import json

import numpy as np
import pytest

from voiceclone.audio import ASR_SR, write_wav
from voiceclone.benchmark import LeakError
from voiceclone.closeness import band, check_leak, closeness, identity_metric, long_form_pairs, write_report

SR = ASR_SR


def speech(seed=0, f0=120.0, words=6, tempo=1.0):
    """Speech-like test signal: words of 2-4 voiced syllables with a moving
    pitch and changing formants (vowels), separated by pauses. The same seed
    gives the same words and timing, so `f0` changes only the pitch and
    `tempo` only the pace (durations are divided by it)."""
    rng = np.random.default_rng(seed)
    vowels = [(700, 1200), (300, 2300), (500, 900), (400, 2000)]
    out = [np.zeros(int(0.3 * SR))]
    phase = 0.0
    for w in range(words):
        for s in range(rng.integers(2, 5)):
            n = int(rng.uniform(0.15, 0.3) / tempo * SR)
            t = np.arange(n) / SR
            # declination over the utterance, plus a rise-fall inside each syllable
            f = f0 * (1 - 0.15 * w / words) * (1 + 0.08 * np.sin(np.pi * t / t[-1]) + rng.uniform(-0.05, 0.05))
            ph = phase + 2 * np.pi * np.cumsum(f) / SR
            phase = ph[-1]
            f1, f2 = vowels[rng.integers(len(vowels))]
            y = np.zeros(n)
            for k in range(1, 30):
                fk = k * f.mean()
                if fk > 4000:
                    break
                amp = (np.exp(-((fk - f1) / 150) ** 2) + 0.6 * np.exp(-((fk - f2) / 200) ** 2) + 0.05) / k ** 0.5
                y += amp * np.sin(k * ph)
            out.append(y * np.hanning(n) ** 0.3)
        out.append(np.zeros(int(rng.uniform(0.2, 0.45) / tempo * SR)))
    y = np.concatenate(out)
    y = 0.3 * y / np.abs(y).max() + 1e-4 * rng.standard_normal(len(y))
    return y.astype(np.float32)


@pytest.fixture(scope="module")
def original():
    return speech(0)


def bands(rep):
    return {d: v["band"] for d, v in rep["dimensions"].items()}


def test_a_recording_is_close_to_itself_on_every_dimension(original):
    rep = closeness([(original, original)], same_words=True)
    measured = {d: b for d, b in bands(rep).items() if b is not None}
    assert set(measured) == {"timbre", "pitch", "rhythm", "energy"}   # identity needs an encoder
    assert all(b == "close" for b in measured.values()), bands(rep)
    m = rep["dimensions"]["timbre"]["metrics"]["timbre_distance_db"]
    assert m["value"] == pytest.approx(0, abs=1e-6)


def test_a_higher_voice_fails_pitch_but_keeps_its_rhythm(original):
    clone = speech(0, f0=120.0 * 2 ** (4 / 12))   # same words and timing, 4 semitones higher
    rep = closeness([(clone, original)], same_words=True)
    pitch = rep["dimensions"]["pitch"]["metrics"]
    assert pitch["pitch_median_st"]["value"] == pytest.approx(4, abs=0.6)   # it measures the shift itself
    assert rep["dimensions"]["pitch"]["band"] == "far"
    assert rep["dimensions"]["rhythm"]["band"] == "close"


def test_slower_speech_fails_rhythm_but_keeps_its_pitch(original):
    clone = speech(0, tempo=0.65)   # same words and pitch, 1.54x longer
    rep = closeness([(clone, original)], same_words=True)
    rhythm = rep["dimensions"]["rhythm"]["metrics"]
    assert rhythm["speech_duration_ratio"]["value"] == pytest.approx(1 / 0.65, rel=0.1)
    assert rep["dimensions"]["rhythm"]["band"] == "far"
    assert rep["dimensions"]["pitch"]["band"] == "close"


def test_alignment_metrics_are_not_computed_for_different_words(original):
    # frame-by-frame comparison of two different sentences measures the
    # sentences, not the voice, so those metrics must be absent, not guessed
    other = speech(seed=7)
    rep = closeness([(other, original)], same_words=False)
    assert rep["dimensions"]["timbre"]["band"] is None
    assert rep["dimensions"]["pitch"]["metrics"]["pitch_contour_corr"]["value"] is None
    assert rep["dimensions"]["rhythm"]["metrics"]["speech_duration_ratio"]["value"] is None
    # the same speaker's pitch level and range are still comparable
    assert rep["dimensions"]["pitch"]["band"] == "close"


def test_identity_is_judged_against_the_speakers_own_similarity():
    # on muffled tape two real recordings of him may only reach 0.58; a clone
    # at 0.55 is then as close as the recordings get, although 0.55 alone
    # looks mediocre
    assert band("identity", identity_metric(0.55, baseline=0.58)[0]) == "close"
    assert band("identity_absolute", identity_metric(0.55, baseline=None)[0]) == "noticeably different"
    # another speaker stays far whichever way it is judged
    assert band("identity", identity_metric(0.25, baseline=0.58)[0]) == "far"
    assert band("identity_absolute", identity_metric(0.25, baseline=None)[0]) == "far"


def test_identity_uses_the_encoder_and_the_baseline(original):
    class Encoder:   # stands in for ECAPA: the voiceprint is the pitch level
        def embed(self, y):
            import librosa

            f0 = np.nanmedian(librosa.pyin(y, fmin=60, fmax=400, sr=SR, hop_length=160)[0])
            v = np.array([np.cos(f0 / 100), np.sin(f0 / 100)])
            return v / np.linalg.norm(v)

    rep = closeness([(speech(3), original)], same_words=False, encoder=Encoder(), baseline=speech(5))
    ident = rep["dimensions"]["identity"]
    assert ident["band"] == "close"
    assert ident["metrics"]["identity"]["baseline"] == pytest.approx(1, abs=0.01)


def clips(path, files):
    path.mkdir(parents=True)
    (path / "metadata.jsonl").write_text("".join(json.dumps({"file": f, "text": "x"}) + "\n" for f in files))
    return path


def lora(path, clips_dir):
    path.mkdir(parents=True)
    (path / "train_log.json").write_text(json.dumps({"clips": str(clips_dir), "params": {}}))
    return path


def test_a_clone_of_a_recording_it_trained_on_is_refused(tmp_path):
    # a clone always sounds closer to audio it learned from, so that score flatters it
    adapter = lora(tmp_path / "lora", clips(tmp_path / "c", ["K06_0001.wav", "K14_0003.wav"]))
    with pytest.raises(LeakError, match="K06"):
        check_leak(["K06"], adapter)
    assert check_leak(["K06"], adapter, allow=True) == {"checked": True, "leaked": ["K06"], "allowed": True}
    assert check_leak(["K20"], adapter) == {"checked": True, "leaked": [], "allowed": False}
    assert check_leak(["K06"], None) == {"checked": False, "leaked": [], "allowed": False}


def test_long_form_folder_pairs_each_clone_chunk_with_the_original_words(tmp_path, original):
    # long-form places chunk i at script row i; the original's words for it
    # are the row's start-end span, not whatever is at the same place in cloned.wav
    a, b = original[: 2 * SR], original[2 * SR:]
    write_wav(tmp_path / "original.wav", original, SR)
    (tmp_path / "chunks").mkdir()
    write_wav(tmp_path / "chunks" / "000.wav", a, SR)
    write_wav(tmp_path / "chunks" / "002.wav", b, SR)
    end = len(original) / SR
    (tmp_path / "script.tsv").write_text(f"0.00\t2.00\tone\n2.00\t2.00\t\n2.00\t{end:.2f}\ttwo\n")
    pairs = long_form_pairs(tmp_path)
    assert len(pairs) == 2   # the empty row has no chunk
    for clone, orig in pairs:
        assert len(clone) == pytest.approx(len(orig), abs=SR // 100)
    rep = closeness(pairs, same_words=True)
    assert all(b == "close" for b in bands(rep).values() if b is not None)


def test_report_files_name_every_dimension_and_embed_the_audio(tmp_path, original):
    write_wav(tmp_path / "o.wav", original, SR)
    rep = closeness([(original, original)], same_words=True)
    write_report(rep, tmp_path / "out", audio=[("original", tmp_path / "o.wav")])
    data = json.loads((tmp_path / "out" / "closeness.json").read_text())
    assert set(data["dimensions"]) == {"identity", "timbre", "pitch", "rhythm", "energy"}
    html = (tmp_path / "out" / "closeness.html").read_text()
    assert "data:audio/wav;base64," in html and "Pitch" in html and "not measured" in html
