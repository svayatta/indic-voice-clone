"""`voiceclone` command line: one subcommand per pipeline step.

Every path is an argument; defaults follow the layout in the README
(data/, models/, outputs/ under the current directory).
"""
import argparse
import sys
from pathlib import Path

from voiceclone.svara.synth import DEFAULT_MODEL


def _log(*a):
    print(*a, flush=True)


# ---------- data preparation ----------
def cmd_restore(a):
    from voiceclone.audio import ffmpeg_decode, integrated_loudness, restore

    src = Path(a.src)
    if a.decode_first:
        src = ffmpeg_decode(src, Path(a.out).with_name(Path(a.out).stem + "_48k.wav"), 48000)
    restore(src, a.out)
    _log(f"wrote {a.out}  {integrated_loudness(a.out)}")


def cmd_parse_transcript(a):
    from voiceclone.jsonl import write_json
    from voiceclone.transcript import non_monotonic, parse_timed_transcript

    utts, bad = parse_timed_transcript(Path(a.src).read_text(encoding="utf-8"))
    for line in bad:
        _log("UNPARSED:", line[:80])
    if non_monotonic(utts):
        _log("non-monotonic starts at idx:", non_monotonic(utts))
    write_json(a.out, utts)
    from collections import Counter

    _log(f"{len(utts)} utterances, speakers={dict(Counter(u['speaker'] for u in utts))}")


def cmd_align(a):
    import json

    from voiceclone.align import Aligner, align_recording
    from voiceclone.jsonl import write_json

    utts = json.load(open(a.utterances, encoding="utf-8"))
    write_json(a.out, align_recording(Aligner(), a.audio, utts, a.emissions_cache, log=_log))
    _log("wrote", a.out)


def cmd_segment(a):
    import json

    from voiceclone.audio import load_wav
    from voiceclone.segment import AlignedSegmentParams, segment_aligned

    wav = load_wav(a.audio, a.sr)
    p = AlignedSegmentParams(min_sps=a.min_sps, min_mean=a.min_mean)
    _log(segment_aligned(json.load(open(a.aligned, encoding="utf-8")), wav, a.sr, a.out, a.speaker, p))


def cmd_vad(a):
    from voiceclone.audio import load_wav
    from voiceclone.segment import VadParams, vad_segment

    p = VadParams(min_dur=a.min_dur, max_dur=a.max_dur, ideal=a.ideal)
    _log(vad_segment(load_wav(a.audio, a.sr), a.sr, a.out, a.prefix, p))


def cmd_transcribe(a):
    from voiceclone.asr import transcribe_dir

    _log("wrote", transcribe_dir(a.vad, a.lang, a.limit, log=_log))


def cmd_build_clips(a):
    from voiceclone.align import Aligner
    from voiceclone.dataset import ClipFilter, build_clips
    from voiceclone.speaker import SpeakerEncoder

    enc = SpeakerEncoder(a.ecapa_dir)
    cent = enc.centroid_from_clips(a.enroll, a.n_enroll)
    lo, hi = a.min_f0, a.max_f0
    if a.f0_from_enroll:
        from voiceclone.dataset import f0_range
        from voiceclone.jsonl import read_jsonl

        lo, hi = f0_range([Path(a.enroll) / c["file"] for c in read_jsonl(Path(a.enroll) / "metadata.jsonl")])
        _log(f"pitch window from the enrollment clips: {lo:.0f}-{hi:.0f} Hz")
    f = ClipFilter(min_sim=a.min_sim, min_f0=lo, max_f0=hi)
    _log(build_clips(a.vad, a.out, enc, Aligner(), cent, f, log=_log))


def cmd_rank(a):
    from voiceclone.jsonl import write_json
    from voiceclone.rank import rank_recordings
    from voiceclone.speaker import SpeakerEncoder

    enc = SpeakerEncoder(a.ecapa_dir)
    write_json(a.out, rank_recordings(a.paths, enc, enc.centroid_from_clips(a.enroll, a.n_enroll), log=_log))


def cmd_prepare(a):
    from voiceclone.prepare import prepare_recording

    prepare_recording(a.src, a.tag, a.data_dir, a.lang, a.enhance_python, a.enhance_wet, log=_log)


def cmd_import_vault(a):
    from voiceclone.vault import import_vault

    _log(import_vault(a.index, a.vault_dir, a.out, a.min_dnsmos, a.min_spk, log=_log))


def cmd_audacity(a):
    from voiceclone.enhance.audacity import audacity_denoise

    chunks_out = a.chunks_out or (str(Path(a.data_dir) / f"vad_{a.tag}_aud") if a.reuse_chunks else None)
    audacity_denoise(a.src, Path(a.data_dir) / "work" / a.tag, a.app, a.reuse_chunks, chunks_out,
                     force_restart=a.force_restart, log=_log)


# ---------- model ----------
def _gen_params(a):
    from voiceclone.svara.synth import GenParams

    return GenParams(temperature=a.temperature, top_p=a.top_p, repetition_penalty=a.rep,
                     max_sec=a.max_sec, min_sps=a.min_sps, max_sps=a.max_sps)


def _sentences(a) -> list[str]:
    from voiceclone.jsonl import read_lines

    s = list(a.text or [])
    if a.text_file:
        s += read_lines(a.text_file)
    return s


def cmd_synth(a):
    from voiceclone.svara.synth import Synthesizer, synthesize_to_dir

    sentences = _sentences(a)
    if not sentences:
        sys.exit("give --text or --text-file")
    ref_text = a.ref_text or (Path(a.ref_text_file).read_text(encoding="utf-8").strip() if a.ref_text_file else None)
    synth = Synthesizer(a.model, a.lora)
    synthesize_to_dir(synth, sentences, a.ref, ref_text, a.out, a.seed, _gen_params(a), log=_log)


def cmd_detect(a):
    import json

    from voiceclone.audio import load16
    from voiceclone.watermark import SR, Watermark, verdict

    descriptions = {"ours": "made with pratidhvani", "other": "AudioSeal watermark from another source",
                    "none": "no watermark found"}
    wm, results = Watermark(), []
    for f in a.files:
        d = wm.detect(load16(f), SR)
        results.append({"file": str(f), "verdict": verdict(d), "probability": d.probability,
                        "bit_errors": d.bit_errors, "ours": d.ours})
    if a.json:
        print(json.dumps(results, ensure_ascii=False))
    else:
        for result in results:
            print(f"{result['file']}: {descriptions[result['verdict']]} (detector {result['probability']:.2f}, "
                  f"{16 - result['bit_errors']}/16 tag bits)")
    if not a.json and any(result["verdict"] == "none" for result in results):
        print("no watermark doesn't prove a recording is genuine: the code can be changed, and heavy noise can hide it")
    if not all(result["ours"] for result in results):
        raise SystemExit(1)


def cmd_train(a):
    from voiceclone.train import TrainParams, finetune

    p = TrainParams(epochs=a.epochs, lr=a.lr, rank=a.rank, max_ref_sec=a.max_ref_sec, max_tgt_sec=a.max_tgt_sec,
                    grad_accum=a.grad_accum, holdout=a.holdout, max_steps=a.max_steps, seed=a.seed,
                    init_adapter=a.init_adapter, save_every=a.save_every)
    finetune(a.clips, a.out, a.model, p, log=_log)


def cmd_long_form(a):
    from voiceclone.longform import long_form
    from voiceclone.svara.synth import Synthesizer

    ref_text = Path(a.ref_text_file).read_text(encoding="utf-8").strip()
    long_form(Synthesizer(a.model, a.lora), a.chunks, a.source, a.ref, ref_text, a.out, a.minutes, a.seed, log=_log)


# ---------- evaluation ----------
def cmd_eval_similarity(a):
    from voiceclone.evaluate import similarity_report
    from voiceclone.jsonl import write_json
    from voiceclone.speaker import SpeakerEncoder

    write_json(a.report, similarity_report(SpeakerEncoder(a.ecapa_dir), a.clips, a.samples, a.file,
                                           a.n_enroll, log=_log))


def cmd_eval_intelligibility(a):
    from voiceclone.align import Aligner
    from voiceclone.evaluate import intelligibility_report
    from voiceclone.jsonl import write_json

    write_json(a.report, intelligibility_report(Aligner(), a.clips, _maybe_lines(a.sentences), a.samples,
                                                a.holdout, a.n_real, log=_log))


def _maybe_lines(path):
    from voiceclone.jsonl import read_lines

    return read_lines(path) if path else None


def cmd_trim(a):
    from voiceclone.align import Aligner
    from voiceclone.postprocess import trim_to_text

    _log("wrote", trim_to_text(Aligner(), a.samples, _maybe_lines(a.sentences), a.out, a.tail, log=_log))


def cmd_best_of(a):
    from voiceclone.align import Aligner
    from voiceclone.evaluate import best_of
    from voiceclone.jsonl import write_json
    from voiceclone.speaker import SpeakerEncoder

    enc = SpeakerEncoder(a.ecapa_dir)
    cent = enc.centroid_from_clips(a.clips, a.n_enroll, shuffle_seed=0)
    rep = best_of(enc, Aligner(), cent, a.takes, a.out, _maybe_lines(a.sentences), a.pick, a.max_cer, log=_log)
    write_json(Path(a.out) / "selection.json", rep)


# ---------- comparing runs ----------
def cmd_benchmark(a):
    from voiceclone.benchmark import LeakError, run_benchmark

    try:
        run_benchmark(a.spec, a.out, a.lora, a.model, a.allow_leak, a.train_clips, a.leaderboard, a.ecapa_dir,
                      log=_log)
    except LeakError as e:
        sys.exit(f"refusing to score: {e}")


def cmd_compare(a):
    import json

    from voiceclone.compare import compare_runs, format_report, listening_results, load_run
    from voiceclone.jsonl import write_json

    ra, rb = load_run(a.run_a), load_run(a.run_b)
    metrics = compare_runs(ra, rb, a.n_boot)
    listening = ratings = None
    if a.answers:
        if not a.key:
            sys.exit("--answers needs --key (the .key.json written next to the listening page)")
        key = json.load(open(a.key, encoding="utf-8"))
        if key.get("mode") == "rate":
            from voiceclone.rating import format_ratings, rating_results

            ratings = rating_results(key, [_answers_file(f) for f in a.answers], a.run_a, a.run_b, a.n_boot)
        else:
            listening = listening_results(key, [json.load(open(f, encoding="utf-8")) for f in a.answers],
                                          a.run_a, a.run_b)
    _log(format_report(ra, rb, metrics, listening))
    if ratings:
        _log(format_ratings(ratings))
    if a.report:
        write_json(a.report, {"a": a.run_a, "b": a.run_b, "metrics": metrics, "listening": listening,
                              "ratings": ratings})


def _answers_file(path):
    """A downloaded answers JSON, or a text file holding one answer code."""
    import json

    from voiceclone.rating import CODE_PREFIX, decode_answers

    text = Path(path).read_text(encoding="utf-8").strip()
    code = text.split()[-1] if text else ""
    return decode_answers(code, Path(path).stem) if code.startswith(CODE_PREFIX) else json.loads(text)


def cmd_listening_status(a):
    """Score `/listening CODE` comments for the `listening-test` status (see .github/workflows/listening-test.yml)."""
    import json

    from voiceclone.jsonl import read_jsonl, write_json
    from voiceclone.rating import decode_answers, format_ratings, listening_status, rating_results

    key = json.load(open(a.key, encoding="utf-8"))
    allowed = {x.strip().lower() for x in a.listeners.split(",") if x.strip()}
    files, ignored = [], []
    for c in read_jsonl(a.comments):   # oldest first, so a listener's last code wins
        login, words = c["login"], c["body"].split()
        if login.lower() not in allowed or len(words) < 2 or words[0] != "/listening":
            continue
        try:
            files.append(decode_answers(words[1], login))
        except ValueError as e:
            ignored.append(f"{login}: {e}")
    res = rating_results(key, files, n_boot=a.n_boot)
    status = listening_status(res, a.dimension, a.min_listeners)
    report = (f"### Listening test: {status['state']}\n\n{status['description']}\n\n```\n{format_ratings(res)}\n```\n"
              + "".join(f"\nIgnored a comment from {x}" for x in ignored))
    write_json(a.out, {**status, "report": report, "ratings": res})
    _log(report)


def cmd_listen_page(a):
    from voiceclone.listen import write_listening_page

    if a.rate:
        from voiceclone.rating import write_rating_page

        from voiceclone.rating import ANCHOR_EVERY

        every = ANCHOR_EVERY if a.check_every is None else a.check_every
        write_rating_page(a.run_a, a.run_b, a.out, a.max_sentences, every, a.seed, log=_log)
        return
    write_listening_page(a.run_a, a.run_b, a.out, 6 if a.check_every is None else a.check_every, a.seed, log=_log)


def cmd_closeness(a):
    from voiceclone.audio import load16
    from voiceclone.benchmark import LeakError, source_of
    from voiceclone.closeness import GENERATED_NAMES, check_leak, closeness, format_report, long_form_pairs, write_report

    if a.long_form and (a.clone or a.original or a.same_words):
        sys.exit("--long-form DIR replaces --clone, --original and --same-words; pass one mode")
    if a.long_form:
        pairs, same_words = long_form_pairs(a.long_form), True
        audio = [("His line, then the clone's (ab.wav)", Path(a.long_form) / "ab.wav")]
        audio = [x for x in audio if x[1].exists()]
    elif a.clone and a.original:
        pairs, same_words = [(load16(a.clone), load16(a.original))], a.same_words
        audio = [("Clone", a.clone), ("Original", a.original)]
    else:
        sys.exit("pass --clone and --original, or --long-form DIR")
    # name the recording the original words come from the way `benchmark` names clips (K06_0012.wav -> K06)
    source = source_of(a.original_source) if a.original_source else source_of(a.original) if a.original else None
    if (a.lora or a.train_clips) and source in (None, *GENERATED_NAMES):
        sys.exit("cannot tell which recording the original comes from"
                 + (f" ({a.original} is a file the tools write)" if source else "")
                 + "; pass --original-source with its recording tag (e.g. K06) so the leak check compares it")
    try:
        leak = check_leak([source] if source else [], a.lora, a.train_clips, a.allow_leak)
    except LeakError as e:
        sys.exit(str(e))
    encoder = encoder2 = mos = None
    if not a.no_identity:
        try:
            from voiceclone.speaker import RESNET_ID, SpeakerEncoder

            encoder = SpeakerEncoder(a.ecapa_dir)
            encoder2 = SpeakerEncoder(a.resnet_dir, source=RESNET_ID)
        except ImportError:
            _log("identity not measured: install the closeness extra (.[closeness]) or pass --no-identity")
    if not a.no_naturalness:
        try:
            from voiceclone.utmos import MosPredictor

            mos = MosPredictor(a.utmos_dir)
        except ImportError:
            _log("naturalness not measured: install the closeness extra (.[closeness]) or pass --no-naturalness")
    try:
        rep = closeness(pairs, same_words, encoder=encoder, baseline=load16(a.baseline) if a.baseline else None,
                        encoder2=encoder2, mos=mos)
    except ValueError as e:
        sys.exit(str(e))
    rep["leak"] = leak
    write_report(rep, a.out, audio)
    _log(format_report(rep))
    _log(f"wrote {Path(a.out) / 'closeness.html'} and closeness.json")


def cmd_run_recipe(a):
    from voiceclone.recipe import run_recipe

    _log("wrote", run_recipe(a.recipe, a.data, a.work, a.spec, log=_log))


def cmd_gate(a):
    from voiceclone.compare import compare_runs, load_run
    from voiceclone.gate import evaluate_gate, format_markdown

    base, cand = load_run(a.baseline), load_run(a.candidate)
    metrics = compare_runs(base, cand, a.n_boot)
    result = evaluate_gate(base, cand, metrics, a.require_improvement)
    md = format_markdown(base, cand, metrics, result)
    _log(md)
    if a.report:
        Path(a.report).write_text(md + "\n", encoding="utf-8")
    sys.exit(0 if result["passed"] else 1)


# ---------- studio ----------
def cmd_serve(a):
    import uvicorn

    from voiceclone.studio.app import create_app
    from voiceclone.studio.engine import Engine
    from voiceclone.studio.jobs import Studio

    engine = Engine(a.model, a.ecapa_dir, a.enhance_python, a.min_train_minutes)
    _log(f"voice studio on http://{a.host}:{a.port}  (voices in {Path(a.root).resolve()})")
    uvicorn.run(create_app(Studio(a.root, engine)), host=a.host, port=a.port, log_level="warning")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="voiceclone", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(fn=fn)
        return p

    def ecapa(p):
        p.add_argument("--ecapa-dir", default="models/ecapa", help="where the ECAPA encoder is cached")

    # data preparation
    p = add("restore", cmd_restore, "Light ffmpeg restoration to mono 24 kHz (keeps vowel quality)")
    p.add_argument("src"); p.add_argument("--out", default="data/work/full_24k_clean.wav")
    p.add_argument("--decode-first", action="store_true", help="decode to 48 kHz WAV first (MP3 input)")

    p = add("parse-transcript", cmd_parse_transcript, "Timed transcript [SS:MM:00] [Speaker] : text -> JSON")
    p.add_argument("src"); p.add_argument("--out", default="data/work/utterances.json")

    p = add("align", cmd_align, "Force-align a full transcript to a full recording")
    p.add_argument("--audio", default="data/work/full_24k_clean.wav")
    p.add_argument("--utterances", default="data/work/utterances.json")
    p.add_argument("--out", default="data/work/aligned.json")
    p.add_argument("--emissions-cache", default="data/work/emissions.npz")

    p = add("segment", cmd_segment, "Cut aligned target-speaker speech into training clips")
    p.add_argument("--speaker", required=True, help="speaker label in the transcript to keep")
    p.add_argument("--aligned", default="data/work/aligned.json")
    p.add_argument("--audio", default="data/work/full_24k_clean.wav")
    p.add_argument("--out", default="data/clips"); p.add_argument("--sr", type=int, default=24000)
    p.add_argument("--min-sps", type=float, default=2.0); p.add_argument("--min-mean", type=float, default=-1.8)

    p = add("vad", cmd_vad, "Cut a recording into 3-10 s chunks at natural pauses (no transcript)")
    p.add_argument("--audio", default="data/work/full_24k_clean.wav")
    p.add_argument("--out", default="data/vad"); p.add_argument("--prefix", default="seg")
    p.add_argument("--sr", type=int, default=24000)
    p.add_argument("--min-dur", type=float, default=3.0); p.add_argument("--max-dur", type=float, default=10.0)
    p.add_argument("--ideal", type=float, default=8.0)

    p = add("transcribe", cmd_transcribe, "Verbatim transcription of chunks with IndicConformer (resumable)")
    p.add_argument("--vad", default="data/vad"); p.add_argument("--lang", default="kn")
    p.add_argument("--limit", type=int, default=0)

    p = add("build-clips", cmd_build_clips, "Keep chunks that are the speaker and whose transcript fits")
    p.add_argument("--vad", action="append", required=True, help="transcribed chunk folder (repeatable)")
    p.add_argument("--enroll", required=True, help="clip folder (metadata.jsonl) of the target speaker")
    p.add_argument("--n-enroll", type=int, default=40)
    p.add_argument("--out", default="data/clips_v2"); p.add_argument("--min-sim", type=float, default=0.5)
    p.add_argument("--min-f0", type=float, default=None, help="drop chunks whose median pitch is below this (Hz)")
    p.add_argument("--max-f0", type=float, default=None, help="drop chunks whose median pitch is above this (Hz)")
    p.add_argument("--f0-from-enroll", action="store_true",
                   help="pitch window = 5th-95th percentile of the enrollment clips' median pitch")
    ecapa(p)

    p = add("rank", cmd_rank, "Rank recordings by speaker match, SNR, bandwidth and speech density")
    p.add_argument("paths", nargs="+"); p.add_argument("--enroll", required=True)
    p.add_argument("--n-enroll", type=int, default=40)
    p.add_argument("--out", default="outputs/recording_rank.json"); ecapa(p)

    p = add("prepare", cmd_prepare, "Decode, restore, segment, (denoise), transcribe one extra recording")
    p.add_argument("src"); p.add_argument("tag"); p.add_argument("--lang", default="kn")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--enhance-python", default=None, help="python of an env with clearvoice (numpy<2)")
    p.add_argument("--enhance-wet", type=float, default=1.0, help="enhanced share when denoising (0.9 keeps some warmth)")

    p = add("audacity-denoise", cmd_audacity, "Remove hiss, mains hum and clicks with Audacity 3.x")
    p.add_argument("src"); p.add_argument("tag"); p.add_argument("--data-dir", default="data")
    p.add_argument("--app", default=str(Path.home() / "Applications/Audacity 3.7.app"))
    p.add_argument("--reuse-chunks", default=None, help="re-cut chunks at this folder's times, keep transcripts")
    p.add_argument("--chunks-out", default=None)
    p.add_argument("--force-restart", action="store_true",
                   help="quit an already-open Audacity first (its unsaved changes are lost)")

    p = add("import-vault", cmd_import_vault, "Import Vani-Harness cloning-vault segments as chunks")
    p.add_argument("index", help="the vault's metadata.jsonl or vault_admitted_segments.jsonl")
    p.add_argument("--vault-dir", required=True, help="folder holding the vault WAVs (matched by file name)")
    p.add_argument("--out", required=True, help="new chunk folder; next run transcribe, then build-clips")
    p.add_argument("--min-dnsmos", type=float, default=None, help="also require DNSMOS OVRL >= this")
    p.add_argument("--min-spk", type=float, default=None, help="also require the vault's speaker cosine >= this")

    # model
    def gen(p, min_sps=2.0):
        p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face id or local folder")
        p.add_argument("--lora", default=None, help="PEFT adapter folder")
        p.add_argument("--temperature", type=float, default=0.6); p.add_argument("--top-p", type=float, default=0.95)
        p.add_argument("--rep", type=float, default=1.1)
        p.add_argument("--max-sec", type=float, default=20.0, help="hard cap per sentence")
        p.add_argument("--min-sps", type=float, default=min_sps, help="cap at syllables/min_sps + 1.5 s")
        p.add_argument("--max-sps", type=float, default=6.0, help="floor at syllables/max_sps s")
        p.add_argument("--seed", type=int, default=0)

    p = add("detect", cmd_detect, "Check recordings for the watermark that every generated take carries")
    p.add_argument("files", nargs="+", help="audio files (WAV, MP3, M4A, ... anything librosa reads)")
    p.add_argument("--json", action="store_true", help="print machine-readable detection results")

    p = add("synth", cmd_synth, "Speak sentences in the reference speaker's voice")
    p.add_argument("--ref", required=True, help="reference clip of the speaker (4-8 s works well)")
    p.add_argument("--ref-text", default=None); p.add_argument("--ref-text-file", default=None)
    p.add_argument("--text", action="append"); p.add_argument("--text-file", default=None)
    p.add_argument("--out", default="outputs/synth"); gen(p)

    p = add("train", cmd_train, "LoRA fine-tune on the speaker's clips (reference-swap)")
    p.add_argument("--clips", required=True); p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-ref-sec", type=float, default=4.5); p.add_argument("--max-tgt-sec", type=float, default=10.5)
    p.add_argument("--grad-accum", type=int, default=4); p.add_argument("--holdout", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-adapter", default=None, help="continue training this saved LoRA")
    p.add_argument("--save-every", type=int, default=50,
                   help="keep a checkpoint-NNNNN/ adapter every N steps (0 = final only)")

    p = add("long-form", cmd_long_form, "Re-voice minutes of a real recording with the same words")
    p.add_argument("--lora", required=True); p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--chunks", required=True, help="folder with segments.jsonl + transcripts.jsonl")
    p.add_argument("--source", required=True, help="the full cleaned recording the chunks came from")
    p.add_argument("--ref", required=True); p.add_argument("--ref-text-file", required=True)
    p.add_argument("--minutes", type=float, default=5.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)

    # evaluation
    p = add("eval-similarity", cmd_eval_similarity, "Voiceprint similarity of samples to the speaker")
    p.add_argument("--clips", required=True); p.add_argument("--samples", action="append", default=[])
    p.add_argument("--file", action="append", default=[]); p.add_argument("--n-enroll", type=int, default=40)
    p.add_argument("--report", default="outputs/similarity_report.json"); ecapa(p)

    p = add("eval-intelligibility", cmd_eval_intelligibility, "Character error of samples vs real clips")
    p.add_argument("--clips", required=True); p.add_argument("--samples", action="append", default=[])
    p.add_argument("--sentences", default=None, help="only needed for sample folders without log.json")
    p.add_argument("--holdout", default=None, help="holdout.json written by train")
    p.add_argument("--n-real", type=int, default=10)
    p.add_argument("--report", default="outputs/intelligibility_report.json")

    p = add("trim", cmd_trim, "Cut each sample just after the last word of its sentence")
    p.add_argument("--samples", required=True); p.add_argument("--out", default=None)
    p.add_argument("--sentences", default=None, help="only needed if the samples have no log.json")
    p.add_argument("--tail", type=float, default=0.3)

    p = add("best-of", cmd_best_of, "Pick the best of several takes per sentence")
    p.add_argument("--takes", action="append", required=True); p.add_argument("--out", required=True)
    p.add_argument("--clips", required=True, help="speaker clips for the voiceprint")
    p.add_argument("--n-enroll", type=int, default=100); p.add_argument("--max-cer", type=float, default=0.85)
    p.add_argument("--pick", choices=["gated", "voice"], default="gated")
    p.add_argument("--sentences", default=None, help="only needed for take folders without log.json")
    ecapa(p)

    # comparing runs
    p = add("benchmark", cmd_benchmark, "Score one adapter on a frozen benchmark spec")
    p.add_argument("--spec", required=True, help="benchmark spec JSON (see examples/bench_spec.example.json)")
    p.add_argument("--lora", default=None, help="PEFT adapter folder (omit for zero-shot)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", required=True, help="run folder, e.g. outputs/bench/RUN")
    p.add_argument("--leaderboard", default=None, help="results.jsonl to append to (default: next to --out)")
    p.add_argument("--allow-leak", action="store_true",
                   help="score even if the adapter trained on a benchmark recording (the row is flagged)")
    p.add_argument("--train-clips", action="append", default=None,
                   help="clip folder(s) the adapter trained on, if its train_log.json cannot say")
    ecapa(p)

    p = add("closeness", cmd_closeness,
            "How close is a clone to the original recording? Identity, timbre, pitch, rhythm, energy and "
            "naturalness, each banded")
    p.add_argument("--clone", default=None, help="the cloned audio")
    p.add_argument("--original", default=None, help="the real recording to compare with (held out from training)")
    p.add_argument("--same-words", action="store_true",
                   help="clone and original say the same words: also measure timbre, contours and pace")
    p.add_argument("--long-form", default=None, help="a `long-form` output folder: compares each chunk with his words")
    p.add_argument("--baseline", default=None,
                   help="another real recording of him, so each number is shown next to what his own recordings reach")
    p.add_argument("--no-identity", action="store_true",
                   help="skip the ECAPA and ResNet voiceprints (no speechbrain needed)")
    p.add_argument("--no-naturalness", action="store_true", help="skip the UTMOS naturalness predictor (no torch needed)")
    p.add_argument("--resnet-dir", default="models/spk-resnet", help="where the second speaker model is cached")
    p.add_argument("--utmos-dir", default="models/utmos", help="where the UTMOS weights (411 MB) are cached")
    p.add_argument("--lora", default=None, help="adapter that made the clone: refuses an original it trained on")
    p.add_argument("--train-clips", action="append", default=None,
                   help="clip folder(s) the adapter trained on, if its train_log.json cannot say")
    p.add_argument("--original-source", default=None,
                   help="recording name of the original for the leak check (default: from its file name)")
    p.add_argument("--allow-leak", action="store_true", help="report anyway when the adapter trained on the original")
    p.add_argument("--out", required=True, help="folder for closeness.html and closeness.json")
    ecapa(p)

    p = add("compare", cmd_compare, "Is benchmark run B better than run A? (paired takes, bootstrap interval)")
    p.add_argument("run_a"); p.add_argument("run_b")
    p.add_argument("--answers", action="append", default=[], help="listening-page answers JSON (repeatable)")
    p.add_argument("--key", default=None, help="the listening page's .key.json")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--report", default=None, help="also write the comparison as JSON")

    p = add("listen-page", cmd_listen_page, "Blind A/B listening page for two benchmark runs")
    p.add_argument("run_a"); p.add_argument("run_b")
    p.add_argument("--out", required=True, help="HTML file; its .key.json is written next to it")
    p.add_argument("--check-every", type=int, default=None,
                   help="one hidden real-clip check per N trials (0 = none); default 6, or 5 with --rate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rate", action="store_true",
                   help="rating mode: rate each take 1-5 against a real recording, with hidden real-vs-real anchors")
    p.add_argument("--max-sentences", type=int, default=8, help="rating mode: sentences per run (one take each)")

    p = add("listening-status", cmd_listening_status, "Score /listening answer codes for a pull request's status")
    p.add_argument("--key", required=True, help="the rating page's .key.json")
    p.add_argument("--comments", required=True, help="JSONL of {login, body}, oldest first")
    p.add_argument("--listeners", required=True, help="comma-separated GitHub logins whose codes count")
    p.add_argument("--dimension", default="accent", choices=["voice", "accent", "natural", "pacing"])
    p.add_argument("--min-listeners", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out", required=True, help="status JSON: state, description, report")

    # quality gate
    p = add("run-recipe", cmd_run_recipe, "Run a recipe (ci/recipe.json) end to end and benchmark its adapter")
    p.add_argument("--recipe", default="ci/recipe.json")
    p.add_argument("--data", required=True, help="the shared, read-only data folder ({data} in the recipe)")
    p.add_argument("--work", required=True, help="new folder for this run ({work} in the recipe)")
    p.add_argument("--spec", required=True, help="benchmark spec JSON")

    p = add("gate", cmd_gate, "Pass or fail a candidate benchmark run against main's (exit code 1 = fail)")
    p.add_argument("baseline"); p.add_argument("candidate")
    p.add_argument("--require-improvement", action="store_true",
                   help="fail unless at least one metric is better with confidence (experiments)")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--report", default=None, help="also write the verdict as Markdown")

    # studio
    p = add("serve", cmd_serve, "Voice studio: upload a sample, clean it, hear and train the clone in a browser")
    p.add_argument("--root", default="studio", help="where voices are kept")
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 keeps it on this computer")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--enhance-python", default=None,
                   help="python of an env with clearvoice: offers MossFormer2 enhancement on the page")
    p.add_argument("--min-train-minutes", type=float, default=8.0)
    ecapa(p)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
