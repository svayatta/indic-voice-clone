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
from voiceclone.cli import main
from voiceclone.closeness import band, check_leak, closeness, identity_metric, long_form_pairs, write_report

SR = ASR_SR


def speech(seed=0, f0=120.0, words=6, tempo=1.0, formants=1.0):
    """Speech-like test signal: words of 2-4 voiced syllables with a moving
    pitch and changing formants (vowels), separated by pauses. The same seed
    gives the same words and timing, so `f0` changes only the pitch, `tempo`
    only the pace (durations are divided by it) and `formants` only the
    vocal tract (every formant frequency is multiplied by it)."""
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
            f1, f2 = (formants * x for x in vowels[rng.integers(len(vowels))])
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


def test_a_higher_voice_fails_pitch_and_nothing_else(original):
    clone = speech(0, f0=120.0 * 2 ** (4 / 12))   # same words, timing and vocal tract, 4 semitones higher
    rep = closeness([(clone, original)], same_words=True)
    pitch = rep["dimensions"]["pitch"]["metrics"]
    assert pitch["pitch_median_st"]["value"] == pytest.approx(4, abs=0.6)   # it measures the shift itself
    assert bands(rep) == {"identity": None, "timbre": None, "pitch": "far", "rhythm": "close", "energy": "close",
                          "naturalness": None}
    # harmonics 4 semitones apart sample the same vocal tract at other frequencies; scored anyway, the
    # pitch change read as a timbre distance of about 20 dB ("far"), sending the next experiment after
    # the voice instead of the pitch. Timbre is left out, and the report says why.
    assert any("timbre" in s and "pitch" in s for s in rep["skipped"])


def test_another_vocal_tract_fails_timbre_and_nothing_else(original):
    clone = speech(0, formants=1.3)   # same words, timing and pitch, formants 30% higher
    rep = closeness([(clone, original)], same_words=True)
    assert bands(rep) == {"identity": None, "timbre": "far", "pitch": "close", "rhythm": "close", "energy": "close",
                          "naturalness": None}


def test_slower_speech_fails_rhythm_but_keeps_its_pitch(original):
    clone = speech(0, tempo=0.65)   # same words and pitch, 1.54x longer
    rep = closeness([(clone, original)], same_words=True)
    rhythm = rep["dimensions"]["rhythm"]["metrics"]
    assert rhythm["speech_duration_ratio"]["value"] == pytest.approx(1 / 0.65, rel=0.1)
    assert bands(rep) == {"identity": None, "timbre": "close", "pitch": "close", "rhythm": "far", "energy": "close",
                          "naturalness": None}


def test_alignment_metrics_are_not_computed_for_different_words(original):
    # frame-by-frame comparison of two different sentences measures the
    # sentences, not the voice, so those metrics must be absent, not guessed
    other = speech(seed=7)
    rep = closeness([(other, original)], same_words=False)
    assert rep["dimensions"]["pitch"]["metrics"]["pitch_contour_corr"]["value"] is None
    # pause share depends on the sentence, so different words leave rhythm unscored, not red
    assert rep["dimensions"]["rhythm"]["metrics"]["pause_fraction_diff"]["value"] is None
    # the same speaker's pitch level, range and loudness spread are still comparable
    assert bands(rep) == {"identity": None, "timbre": None, "pitch": "close", "rhythm": None, "energy": "close",
                          "naturalness": None}


def test_a_monotone_clone_is_far_on_pitch_range(original):
    # a range of 0.0 is the worst range there is, not a missing one
    from voiceclone.closeness import unaligned_metrics

    flat = {"pitch_median": 10.0, "pitch_range": 0.0, "pause_fraction": 0.2, "dynamic_range": 10.0}
    lively = {**flat, "pitch_range": 4.0}

    def range_band(clone, orig):
        return band("pitch_range_ratio", unaligned_metrics(clone, orig, same_words=True)["pitch_range_ratio"])

    assert range_band(flat, lively) == "far"     # monotone clone of a lively original
    assert range_band(lively, flat) == "far"     # lively clone of a monotone original
    assert range_band(flat, flat) == "close"     # both flat: the same, not missing


def test_a_quiet_chunk_is_measured_with_its_own_speech_mask(original):
    # one speech mask over loud and quiet chunks glued together counts the quiet
    # chunk as silence, so its 4-semitone error would vanish from the pitch level
    quiet_orig = 0.003 * original / np.abs(original).max()
    quiet_clone = 0.003 * speech(0, f0=120.0 * 2 ** (4 / 12))
    rep = closeness([(original, original), (quiet_clone, quiet_orig)], same_words=True)
    level = rep["dimensions"]["pitch"]["metrics"]["pitch_median_st"]["value"]
    assert level == pytest.approx(2, abs=0.5)   # half the speaking time is 4 semitones off
    assert [c["pair"] for c in rep["contours"]] == [0, 1]   # the figure shows every pair
    # each caption gives his speaking time, not the length of the file with its pauses
    for c, m in zip(rep["contours"], rep["per_pair"]):
        assert c["seconds"] == round(m["speech_s_original"], 1) < len(original) / SR


def test_pairs_count_by_speaking_time_not_by_number(original):
    # a one-second chunk must not move the result as much as a long one
    short = original[: int(1.4 * SR)]
    rep = closeness([(original, original), (speech(0, formants=1.3)[: len(short)], short)], same_words=True)
    t = rep["dimensions"]["timbre"]["metrics"]["timbre_distance_db"]["value"]
    far = closeness([(speech(0, formants=1.3)[: len(short)], short)], same_words=True)
    assert 0 < t < 0.5 * far["dimensions"]["timbre"]["metrics"]["timbre_distance_db"]["value"]


def test_his_own_spread_counts_as_close():
    # two recordings of him 2 semitones apart: a clone inside that spread is as
    # close as he is to himself, not "noticeably different"
    assert band("pitch_median_st", 1.8) == "noticeably different"
    assert band("pitch_median_st", 1.8, baseline=2.0) == "close"
    assert band("pitch_median_st", 2.5, baseline=2.0) == "noticeably different"
    assert band("pitch_range_ratio", 0.7, baseline=1 / 1.5) == "close"
    # a tight baseline never makes the cutoffs stricter than the defaults
    assert band("pitch_median_st", 0.9, baseline=0.1) == "close"


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

    rep = closeness([(speech(3), original)], same_words=False, encoder=Encoder(), baseline=speech(5, f0=150))
    ident = rep["dimensions"]["identity"]["metrics"]
    # the ratio is shown against his own 1.0, and the raw cosines on their own row, never mixed in one cell
    assert ident["identity"]["baseline"] == 1.0
    base_cos = ident["identity_cosine"]["baseline"]
    assert 0.5 <= base_cos < 0.98
    assert ident["identity"]["value"] == pytest.approx(ident["identity_cosine"]["value"] / base_cos)


def test_a_baseline_that_is_the_same_take_or_another_speaker_is_refused(original):
    # the same take (cosine ~1) makes every clone look poor; a cosine of 0.2 makes a
    # mediocre clone look as close as recordings get. Either way the band would lie.
    baseline = speech(5)

    class Encoder:   # the baseline embeds as `vec`; clone and original as (1, 0)
        def __init__(self, vec):
            self.vec = vec

        def embed(self, y):
            return self.vec if y is baseline else np.array([1.0, 0.0])

    with pytest.raises(ValueError, match="same take"):
        closeness([(original, original)], False, encoder=Encoder(np.array([1.0, 0.0])), baseline=baseline)
    with pytest.raises(ValueError, match="another speaker"):
        closeness([(original, original)], False, encoder=Encoder(np.array([0.2, np.sqrt(0.96)])), baseline=baseline)
    # the same holds for the second model: ECAPA accepting the baseline must not let the ResNet
    # divide by a cosine of 0.2, which would turn a clone at 0.3 into a ratio of 1.5, "close"
    ok = Encoder(np.array([0.8, 0.6]))
    with pytest.raises(ValueError, match="ResNet.*another speaker"):
        closeness([(original, original)], False, encoder=ok, encoder2=Encoder(np.array([0.2, np.sqrt(0.96)])),
                  baseline=baseline)
    with pytest.raises(ValueError, match="ResNet.*same take"):
        closeness([(original, original)], False, encoder=ok, encoder2=Encoder(np.array([1.0, 0.0])),
                  baseline=baseline)
    closeness([(original, original)], False, encoder=ok, encoder2=ok, baseline=baseline)   # a sound baseline passes


class ByLength:
    """Stands in for a speaker model: each recording (told apart by its length, since the
    pairs are concatenated before embedding) has a fixed voiceprint at `cos` to (1, 0)."""

    def __init__(self, cos_by_length):
        self.cos = cos_by_length

    def embed(self, y):
        c = self.cos[len(y)]
        return np.array([c, np.sqrt(1 - c * c)])


def test_a_second_model_that_disagrees_drags_identity_down(original):
    # ECAPA can be fooled: a clone that matches what ECAPA attends to scores as high as his own
    # recordings. If the ResNet hears another voice, identity must not be reported close on
    # ECAPA's word alone, or one model's blind spot becomes the report's.
    clone, baseline = speech(3), speech(5)
    lengths = {len(original): 1.0, len(clone): 0.8, len(baseline): 0.8}
    assert len(lengths) == 3
    ecapa = ByLength(lengths)                                  # clone as close as his own other recording
    resnet = ByLength({**lengths, len(clone): 0.3})           # clone far below his own other recording
    alone = closeness([(clone, original)], False, encoder=ecapa, baseline=baseline)
    assert alone["dimensions"]["identity"]["band"] == "close"
    both = closeness([(clone, original)], False, encoder=ecapa, baseline=baseline, encoder2=resnet)
    ident = both["dimensions"]["identity"]
    assert ident["metrics"]["identity"]["band"] == "close"    # ECAPA's own row still says what it saw
    assert ident["metrics"]["identity_resnet"]["value"] == pytest.approx(0.3 / 0.8)
    assert ident["metrics"]["identity_resnet"]["band"] == "far"
    assert ident["band"] == "far"
    # without another recording of him the ResNet's scale is uncalibrated: shown, never banded on a guess
    nobase = closeness([(clone, original)], False, encoder=ecapa, encoder2=resnet)
    row = nobase["dimensions"]["identity"]["metrics"]["identity_resnet"]
    assert row["value"] == pytest.approx(0.3) and row["band"] is None


class Mos:
    """Stands in for UTMOS: a fixed predicted MOS per recording."""

    def __init__(self, scores):
        self.scores = scores

    def predict(self, y):
        return next(s for a, s in self.scores if a is y)


def test_naturalness_judges_the_gap_to_his_recording_not_the_score(original):
    # archival tape scores about 2 on UTMOS however natural he sounds. Judged on the score,
    # every clone of a tape would be "far"; judged on the gap, a clone that sounds as natural
    # as his tape is close, and one far below a clean recording is far.
    clone = speech(3)

    def gap(clone_mos, orig_mos, baseline=None):
        scores = [(clone, clone_mos), (original, orig_mos)] + ([(baseline[0], baseline[1])] if baseline else [])
        rep = closeness([(clone, original)], False, mos=Mos(scores), baseline=baseline[0] if baseline else None)
        return rep["dimensions"]["naturalness"]

    tape = gap(2.0, 2.1)
    assert tape["band"] == "close"
    assert tape["metrics"]["naturalness_gap"]["value"] == pytest.approx(0.1)
    assert tape["metrics"]["mos_clone"]["value"] == 2.0   # the scores themselves are shown, not banded
    assert gap(2.0, 3.9)["band"] == "far"                 # the same 2.0 below a clean recording
    assert gap(4.2, 2.1)["band"] == "close"               # cleaner than the tape is not less natural
    # his own recordings already 0.6 apart: a clone 0.5 below this one is within his spread
    assert gap(2.0, 2.5)["band"] == "noticeably different"
    assert gap(2.0, 2.5, baseline=(speech(5), 3.1))["band"] == "close"


def test_a_missing_extra_reports_not_measured_instead_of_failing(tmp_path, monkeypatch, capsys, original):
    # the core install has no torch or speechbrain; the command must still report the other
    # dimensions and say which rows it could not measure, not crash or leave them out silently
    import sys

    for mod in ("torch", "speechbrain", "voiceclone.utmos"):
        monkeypatch.setitem(sys.modules, mod, None)
    write_wav(tmp_path / "o.wav", original, SR)
    main(["closeness", "--clone", str(tmp_path / "o.wav"), "--original", str(tmp_path / "o.wav"),
          "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "identity not measured" in out and "naturalness not measured" in out
    assert "Naturalness: not measured" in out and "Identity: not measured" in out
    data = json.loads((tmp_path / "out" / "closeness.json").read_text())
    assert data["dimensions"]["naturalness"]["band"] is None
    assert data["dimensions"]["identity"]["metrics"]["identity_resnet"]["value"] is None
    assert data["dimensions"]["pitch"]["band"] == "close"   # the rest is still measured


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
    # unknown training data: refused, or with --allow-leak reported as unchecked (as `benchmark` does)
    with pytest.raises(LeakError):
        check_leak(["K06"], tmp_path)
    assert check_leak(["K06"], tmp_path, allow=True)["checked"] is False


def test_the_leak_check_cannot_be_dodged_by_the_file_name(tmp_path):
    # long-form writes original.wav; its stem names no recording, so a check
    # against it would always pass. The CLI must ask which recording it is.
    adapter = lora(tmp_path / "lora", clips(tmp_path / "c", ["K06_0001.wav"]))
    wav = tmp_path / "original.wav"
    write_wav(wav, speech(0)[:SR], SR)
    with pytest.raises(SystemExit, match="original-source"):
        main(["closeness", "--clone", str(wav), "--original", str(wav), "--lora", str(adapter), "--no-identity",
              "--out", str(tmp_path / "o")])
    # a file name in --original-source is read the way benchmark reads clip names
    with pytest.raises(SystemExit, match="K06"):
        main(["closeness", "--clone", str(wav), "--original", str(wav), "--lora", str(adapter), "--no-identity",
              "--original-source", "K06.wav", "--out", str(tmp_path / "o")])


def test_long_form_cannot_be_mixed_with_a_single_pair(tmp_path):
    with pytest.raises(SystemExit, match="one mode"):
        main(["closeness", "--long-form", str(tmp_path), "--clone", "a.wav", "--out", str(tmp_path / "o")])


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
    assert set(data["dimensions"]) == {"identity", "timbre", "pitch", "rhythm", "energy", "naturalness"}
    html = (tmp_path / "out" / "closeness.html").read_text()
    assert "data:audio/wav;base64," in html and "Pitch" in html and "not measured" in html
