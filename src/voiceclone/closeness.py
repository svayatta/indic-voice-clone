"""How close is a clone to the original recording, and in what way?

`benchmark` and `gate` answer "is run B better than run A" with one
similarity number. This answers "where does this clone fall short of him",
one dimension at a time, so the next change can target it:

  identity  ECAPA voiceprint cosine, as a fraction of what two real
            recordings of him reach (or absolute, without a second recording)
  timbre    spectral-envelope distance on the same words (MCD-style)
  pitch     median level and range (any words); intonation contour (same words)
  rhythm    speaking time for the same words; share of time spent in pauses
  energy    loud-to-quiet spread (any words); stress contour (same words)

Metrics that compare frame by frame need the same words in both recordings
(`long-form` writes such pairs). For different words they are left out, not
guessed: they would measure the sentences, not the voice.

Each metric gets a band: close, noticeably different or far. Thresholds are
named constants below, each with its reason. A dimension takes the worst
band of its metrics. There is deliberately no overall score: one number
hides which dimension to fix.
"""
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voiceclone.audio import ASR_SR

SR = ASR_SR
HOP = 160                       # 10 ms frames
FRAME = 400                     # 25 ms level window
FMIN, FMAX = 60.0, 400.0        # pitch search range, Hz (adult speech)
SPEECH_ABOVE_FLOOR_DB = 12.0    # speech = frame level above the 10th-percentile level + this, as in segment.py,
SPEECH_BELOW_PEAK_DB = 30.0     # and within this of loud speech (95th percentile): fainter frames are tails,
                                # reverb or processing artefacts, which would otherwise count on clean recordings
MAX_DTW_CELLS = 25_000_000      # frames(clone) x frames(original); about 50 s x 50 s. Longer pairs are skipped
MIN_FRAMES_FOR_CORR = 20        # fewer aligned frames than this give no contour correlation

# librosa's MFCCs are a DCT of 10*log10(mel power), i.e. of 20/ln(10) * ln|X|.
# Classic MCD is 10/ln(10) * sqrt(2 * sum(dc^2)) over natural-log cepstra, which
# for these coefficients is |dc| / sqrt(2). The DCT normalisation still differs
# from SPTK mel-cepstra, so the numbers are MCD-like, not comparable with papers.
TIMBRE_SCALE = 1 / np.sqrt(2)

CLOSE, DIFFERENT, FAR = "close", "noticeably different", "far"
_ORDER = {CLOSE: 0, DIFFERENT: 1, FAR: 2}


@dataclass(frozen=True)
class Threshold:
    kind: str        # "high": higher is closer; "low": lower is closer; "ratio": closer to 1 is closer
    close: float     # at or better than this: close
    different: float  # at or better than this: noticeably different; worse: far
    label: str
    unit: str
    why: str


THRESHOLDS = {
    "identity": Threshold(
        "high", 0.90, 0.75, "voice identity, as a fraction of his own", "",
        "ECAPA cosine of clone vs original, divided by the cosine of another real recording of him vs the "
        "original. On tape even his own recordings disagree, so 0.9 of that is as close as recordings get."),
    "identity_absolute": Threshold(
        "high", 0.65, 0.50, "voice identity (ECAPA cosine)", "",
        "No second recording to compare with. His own clips reach about 0.75 against his voiceprint and the "
        "best adapter so far about 0.70 (docs/experiments/log.md); below 0.5 is usually another speaker."),
    "timbre_distance_db": Threshold(
        "low", 6.0, 9.0, "timbre distance (MCD-like)", "dB",
        "Spectral-envelope distance on the same words after time alignment. Provisional thresholds: typical "
        "TTS systems report 4-8 dB MCD; to be calibrated on real clone/original pairs."),
    "pitch_median_st": Threshold(
        "low", 1.0, 3.0, "pitch level difference", "semitones",
        "Listeners start to hear a different pitch level at about 1 semitone; 3 semitones is another register."),
    "pitch_range_ratio": Threshold(
        "ratio", 1.25, 1.6, "pitch range, clone / his", "x",
        "Range is the 5th-95th percentile in semitones. A ratio far below 1 is monotone, far above is "
        "exaggerated intonation."),
    "pitch_contour_corr": Threshold(
        "high", 0.7, 0.4, "intonation contour correlation", "",
        "Same words, time-aligned: does the pitch rise and fall in the same places? Intonation carries much "
        "of an accent."),
    "speech_duration_ratio": Threshold(
        "ratio", 1.15, 1.4, "speaking time for the same words, clone / his", "x",
        "Tempo changes of about 15% are noticeable; 40% sounds like another speaker's pace."),
    "pause_fraction_diff": Threshold(
        "low", 0.05, 0.15, "difference in share of time spent in pauses", "",
        "Pauses between phrases are part of a speaker's rhythm; computed between the first and last speech."),
    "dynamic_range_diff_db": Threshold(
        "low", 3.0, 6.0, "difference in loud-to-quiet spread", "dB",
        "Spread is the 10th-95th percentile frame level during speech. 3 dB is about the smallest audible "
        "level difference; flat speech sounds read, not spoken."),
    "energy_contour_corr": Threshold(
        "high", 0.6, 0.3, "stress contour correlation", "",
        "Same words, time-aligned: does loudness rise and fall on the same syllables?"),
}

DIMENSIONS = {
    "identity": ("Identity", ["identity"]),
    "timbre": ("Timbre", ["timbre_distance_db"]),
    "pitch": ("Pitch", ["pitch_median_st", "pitch_range_ratio", "pitch_contour_corr"]),
    "rhythm": ("Rhythm", ["speech_duration_ratio", "pause_fraction_diff"]),
    "energy": ("Energy", ["dynamic_range_diff_db", "energy_contour_corr"]),
}


def band(key: str, value) -> str | None:
    """Band of a metric value under THRESHOLDS[key]; None when not measured."""
    if value is None or not np.isfinite(value):
        return None
    t = THRESHOLDS[key]
    if t.kind == "high":
        return CLOSE if value >= t.close else DIFFERENT if value >= t.different else FAR
    x = (max(value, 1 / value) if value > 0 else np.inf) if t.kind == "ratio" else value
    return CLOSE if x <= t.close else DIFFERENT if x <= t.different else FAR


def identity_metric(sim: float, baseline: float | None) -> tuple[float, str]:
    """(value, threshold key): relative to his own real-vs-real similarity when there is one."""
    if baseline is not None and baseline > 0:
        return sim / baseline, "identity"
    return sim, "identity_absolute"


def _worst(bands):
    bands = [b for b in bands if b is not None]
    return max(bands, key=_ORDER.get) if bands else None


# ---------- measurements ----------
def features(y: np.ndarray) -> dict:
    """Per-frame level (dB), speech mask, pitch (semitones re 55 Hz, NaN when unvoiced) and MFCCs."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0] + 1e-8)
    speech = db > max(np.percentile(db, 10) + SPEECH_ABOVE_FLOOR_DB, np.percentile(db, 95) - SPEECH_BELOW_PEAK_DB)
    f0, voiced, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=SR, frame_length=1024, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_fft=512, hop_length=HOP, n_mels=40)
    n = min(len(db), len(f0), mfcc.shape[1])
    st = np.where(voiced[:n] & speech[:n], 12 * np.log2(np.where(voiced[:n], f0[:n], FMIN) / 55.0), np.nan)
    return {"db": db[:n], "speech": speech[:n], "st": st, "mfcc": mfcc[:, :n]}


def stats(f: dict) -> dict:
    """Whole-recording statistics that need no alignment."""
    st = f["st"][np.isfinite(f["st"])]
    idx = np.flatnonzero(f["speech"])
    span = f["speech"][idx[0]: idx[-1] + 1] if len(idx) else np.array([], bool)
    level = f["db"][f["speech"]]
    return {
        "pitch_median": float(np.median(st)) if len(st) else None,
        "pitch_range": float(np.percentile(st, 95) - np.percentile(st, 5)) if len(st) > 1 else None,
        "pause_fraction": float(1 - span.mean()) if len(span) else None,
        "dynamic_range": float(np.percentile(level, 95) - np.percentile(level, 10)) if len(level) > 1 else None,
    }


def _absdiff(a, b):
    return None if a is None or b is None else abs(a - b)


def unaligned_metrics(sc: dict, so: dict) -> dict:
    return {
        "pitch_median_st": _absdiff(sc["pitch_median"], so["pitch_median"]),
        "pitch_range_ratio": (sc["pitch_range"] / so["pitch_range"]
                              if sc["pitch_range"] and so["pitch_range"] else None),
        "pause_fraction_diff": _absdiff(sc["pause_fraction"], so["pause_fraction"]),
        "dynamic_range_diff_db": _absdiff(sc["dynamic_range"], so["dynamic_range"]),
    }


def _speech_span(f):
    idx = np.flatnonzero(f["speech"])
    if not len(idx):
        return f
    s = slice(idx[0], idx[-1] + 1)
    return {k: (v[:, s] if v.ndim == 2 else v[s]) for k, v in f.items()}


def _corr(a, b):
    if len(a) < MIN_FRAMES_FOR_CORR or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def aligned_metrics(fc: dict, fo: dict) -> dict:
    """Same-words metrics after DTW on MFCC 1-12; {"skipped": reason} when the pair is too long."""
    import librosa

    fc, fo = _speech_span(fc), _speech_span(fo)
    nc, no = fc["mfcc"].shape[1], fo["mfcc"].shape[1]
    if nc * no > MAX_DTW_CELLS:
        return {"skipped": f"{nc / 100:.0f} s x {no / 100:.0f} s is too long to align; use long-form chunk pairs"}
    X, Y = fc["mfcc"][1:13], fo["mfcc"][1:13]
    _, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean")
    i, j = wp[::-1, 0], wp[::-1, 1]
    both = fc["speech"][i] & fo["speech"][j]
    voiced = np.isfinite(fc["st"][i]) & np.isfinite(fo["st"][j])
    dist = np.linalg.norm(X[:, i[both]] - Y[:, j[both]], axis=0)
    return {
        "timbre_distance_db": float(dist.mean() * TIMBRE_SCALE) if len(dist) else None,
        "pitch_contour_corr": _corr(fc["st"][i][voiced], fo["st"][j][voiced]),
        "energy_contour_corr": _corr(fc["db"][i][both], fo["db"][j][both]),
        "speech_s_clone": float(fc["speech"].sum() * HOP / SR),
        "speech_s_original": float(fo["speech"].sum() * HOP / SR),
        "contour": (fc["st"][i].tolist(), fo["st"][j].tolist()),
    }


def _mean(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return float(np.mean(v)) if v else None


def _downsample(xs, n=400):
    xs = list(xs)
    step = max(1, len(xs) // n)
    return [None if not np.isfinite(x) else round(float(x), 2) for x in xs[::step]]


# ---------- report ----------
def closeness(pairs, same_words: bool, encoder=None, baseline: np.ndarray | None = None) -> dict:
    """Report for (clone, original) waveform pairs at 16 kHz.

    same_words: each pair says the same words (enables the aligned metrics).
    encoder: anything with .embed(y16) -> unit vector (SpeakerEncoder); None skips identity.
    baseline: another real recording of the speaker (different words), so each
    unaligned metric and identity can be shown next to what his own recordings reach."""
    clone = np.concatenate([np.asarray(c, np.float32) for c, _ in pairs])
    orig = np.concatenate([np.asarray(o, np.float32) for _, o in pairs])
    fo = features(orig)
    values = unaligned_metrics(stats(features(clone)), stats(fo))
    base_values = unaligned_metrics(stats(features(baseline)), stats(fo)) if baseline is not None else {}

    per_pair, skipped = [], []
    if same_words:
        for n, (c, o) in enumerate(pairs):
            m = aligned_metrics(features(c), features(o))
            if "skipped" in m:
                skipped.append(f"pair {n}: {m['skipped']}")
            per_pair.append(m)
        done = [m for m in per_pair if "skipped" not in m]
        for k in ("timbre_distance_db", "pitch_contour_corr", "energy_contour_corr"):
            values[k] = _mean(done, k)
        so = sum(m["speech_s_original"] for m in done)
        values["speech_duration_ratio"] = sum(m["speech_s_clone"] for m in done) / so if so else None
    first = next((m for m in per_pair if "contour" in m), None)
    contour = ({"aligned": True, "clone": _downsample(first["contour"][0]), "original": _downsample(first["contour"][1])}
               if first else {"aligned": False, "clone": _downsample(features(pairs[0][0])["st"]),
                              "original": _downsample(features(pairs[0][1])["st"])})

    keys = {k: k for k in THRESHOLDS}
    ident_base = None
    if encoder is not None:
        eo = encoder.embed(orig)
        sim = float(np.dot(encoder.embed(clone), eo))
        ident_base = float(np.dot(encoder.embed(np.asarray(baseline, np.float32)), eo)) if baseline is not None else None
        values["identity"], keys["identity"] = identity_metric(sim, ident_base)
        values["identity_cosine"] = sim
    base_values["identity"] = ident_base

    dims = {}
    for d, (label, metrics) in DIMENSIONS.items():
        rows = {}
        for m in metrics:
            t = THRESHOLDS[keys[m]]
            v = values.get(m)
            rows[m] = {"value": v, "baseline": base_values.get(m), "band": band(keys[m], v), "label": t.label,
                       "unit": t.unit, "why": t.why,
                       "threshold": {"kind": t.kind, "close": t.close, "different": t.different}}
        dims[d] = {"label": label, "band": _worst(r["band"] for r in rows.values()), "metrics": rows}
    return {"same_words": same_words, "pairs": len(pairs), "identity_cosine": values.get("identity_cosine"),
            "dimensions": dims, "skipped": skipped, "contour": contour, "leak": None,
            "per_pair": [{k: v for k, v in m.items() if k != "contour"} for m in per_pair]}


def check_leak(sources, lora, train_clips=None, allow: bool = False) -> dict:
    """Refuse an original recording the adapter trained on: a clone sounds closer to audio it learned from."""
    from voiceclone.benchmark import LeakError, training_sources

    if lora is None and not train_clips:
        return {"checked": False, "leaked": [], "allowed": False}
    leaked = sorted(set(sources) & set(training_sources(lora, train_clips)))
    if leaked and not allow:
        raise LeakError(f"the adapter trained on {', '.join(leaked)}, so the clone would look closer than it is; "
                        "compare against a held-out recording, or pass --allow-leak to report it anyway")
    return {"checked": True, "leaked": leaked, "allowed": bool(leaked) and allow}


def long_form_pairs(folder) -> list[tuple[np.ndarray, np.ndarray]]:
    """(clone chunk, original words) pairs from a `long-form` output folder: script.tsv row i is
    chunks/NNN.wav, and its start-end span of original.wav is what he said there."""
    from voiceclone.audio import load16

    d = Path(folder)
    orig = load16(d / "original.wav")
    pairs = []
    for i, line in enumerate((d / "script.tsv").read_text(encoding="utf-8").splitlines()):
        start, end = (float(x) for x in line.split("\t")[:2])
        chunk = d / "chunks" / f"{i:03d}.wav"
        if chunk.exists() and end > start:
            pairs.append((load16(chunk), orig[int(start * SR): int(end * SR)]))
    if not pairs:
        raise ValueError(f"{d}: no chunk in chunks/ matches a script.tsv row; is it a `long-form` output folder?")
    return pairs


def _fmt(v, unit=""):
    if v is None:
        return "not measured"
    return f"{v:.2f}" + (f" {unit}" if unit and unit != "x" else "x" if unit == "x" else "")


def format_report(rep: dict) -> str:
    lines = [f"Closeness of the clone to the original ({rep['pairs']} pair(s), "
             f"{'same words' if rep['same_words'] else 'different words: aligned metrics not measured'})"]
    for d in rep["dimensions"].values():
        lines.append(f"{d['label']}: {d['band'] or 'not measured'}")
        for m in d["metrics"].values():
            base = f" (his own recordings: {_fmt(m['baseline'], m['unit'])})" if m["baseline"] is not None else ""
            lines.append(f"  {m['label']}: {_fmt(m['value'], m['unit'])}{base}"
                         + (f" -> {m['band']}" if m["band"] else ""))
    lines += [f"skipped {s}" for s in rep["skipped"]]
    if rep.get("leak") and rep["leak"].get("allowed"):
        lines.append(f"WARNING: the adapter trained on {rep['leak']['leaked']}; these scores flatter the clone")
    return "\n".join(lines)


def _svg(contour) -> str:
    w, h = 720, 200
    pts = [v for v in contour["clone"] + contour["original"] if v is not None]
    if not pts:
        return "<p>No voiced speech to plot.</p>"
    lo, hi = min(pts) - 1, max(pts) + 1

    def poly(vals, colour):
        n = max(len(vals) - 1, 1)
        segs, cur = [], []
        for k, v in enumerate(vals):
            if v is None:
                if cur:
                    segs.append(cur); cur = []
                continue
            cur.append(f"{k / n * w:.1f},{h - (v - lo) / (hi - lo) * h:.1f}")
        segs.append(cur)
        return "".join(f'<polyline fill="none" stroke="{colour}" stroke-width="1.5" points="{" ".join(s)}"/>'
                       for s in segs if len(s) > 1)

    note = "time-aligned on the same words" if contour["aligned"] else "each over its own duration"
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="pitch contours">'
            f'{poly(contour["original"], "#1f6feb")}{poly(contour["clone"], "#d1242f")}</svg>'
            f'<p class="note">Pitch in semitones, {note}. <span style="color:#1f6feb">Original</span>, '
            f'<span style="color:#d1242f">clone</span>.</p>')


def write_report(rep: dict, out_dir, audio=(), max_embed_bytes: int = 25_000_000) -> Path:
    """closeness.json and a self-contained closeness.html (audio embedded) in out_dir."""
    from voiceclone.listen import _data_uri

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "closeness.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    e = html.escape
    players = []
    for label, path in audio:
        p = Path(path)
        if p.stat().st_size <= max_embed_bytes:
            players.append(f'<p>{e(label)}<br><audio controls src="{_data_uri(p)}"></audio></p>')
        else:
            players.append(f"<p>{e(label)}: {e(p.name)} is too large to embed.</p>")
    rows = []
    for d in rep["dimensions"].values():
        rows.append(f'<tr class="dim"><th colspan="4">{e(d["label"])}</th>'
                    f'<th class="{e((d["band"] or "none").split()[0])}">{e(d["band"] or "not measured")}</th></tr>')
        for m in d["metrics"].values():
            rows.append(f'<tr><td>{e(m["label"])}</td><td>{e(_fmt(m["value"], m["unit"]))}</td>'
                        f'<td>{e(_fmt(m["baseline"], m["unit"]) if m["baseline"] is not None else "")}</td>'
                        f'<td class="why">{e(m["why"])}</td><td>{e(m["band"] or "not measured")}</td></tr>')
    words = "same words" if rep["same_words"] else "different words, so time-aligned metrics are not measured"
    warn = ""
    if rep.get("leak") and rep["leak"].get("allowed"):
        warn = f'<p class="far">The adapter trained on {e(", ".join(rep["leak"]["leaked"]))}: these scores flatter the clone.</p>'
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Closeness report</title>
<style>
body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 980px; margin: 2em auto; padding: 0 1em; color: #1f2328; }}
table {{ border-collapse: collapse; width: 100%; }} td, th {{ border-bottom: 1px solid #d0d7de; padding: 6px 8px; text-align: left; vertical-align: top; }}
tr.dim th {{ background: #f6f8fa; }} td.why {{ color: #57606a; font-size: 13px; }} .note {{ color: #57606a; font-size: 13px; }}
.close {{ color: #1a7f37; }} .noticeably {{ color: #9a6700; }} .far {{ color: #d1242f; }} .none {{ color: #57606a; }}
</style></head><body>
<h1>How close is the clone?</h1>
<p>{rep["pairs"]} pair(s), {e(words)}. Each dimension takes the worst band of its metrics; there is no overall score.</p>
{warn}{"".join(players)}
<table><tr><th>Metric</th><th>Clone vs original</th><th>His own recordings</th><th>What it means</th><th>Band</th></tr>
{"".join(rows)}</table>
<h2>Pitch</h2>{_svg(rep["contour"])}
{"".join(f'<p class="note">Skipped {e(s)}</p>' for s in rep["skipped"])}
</body></html>
"""
    (out / "closeness.html").write_text(page, encoding="utf-8")
    return out
