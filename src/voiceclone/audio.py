"""Audio loading, normalisation and ffmpeg-based decoding/restoration."""
import subprocess
from pathlib import Path

import numpy as np

TTS_SR = 24000   # SNAC codec rate used by the TTS model
ASR_SR = 16000   # aligner, ASR and speaker encoder rate

# Deliberately mild restoration: aggressive denoising shifts formants, which is
# where the accent lives.
#  adeclip: repair clipped peaks; highpass 60 Hz: hum/rumble;
#  afftdn: mild FFT denoise (8 dB) with noise floor at -42 dB.
RESTORE_FILTER = "adeclip,highpass=f=60,afftdn=nf=-42:nr=8:tn=1"


def load_wav(path, sr: int = TTS_SR) -> np.ndarray:
    """Read a WAV with soundfile, downmix to mono, resample to `sr`."""
    import librosa
    import soundfile as sf

    wav, file_sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return wav


def load16(path) -> np.ndarray:
    """Any audio file, mono at 16 kHz, float32. Formats librosa can't open
    (AAC in .m4a, for example) are decoded with ffmpeg."""
    import librosa

    try:
        return librosa.load(str(path), sr=ASR_SR, mono=True)[0].astype(np.float32)
    except Exception:
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(ASR_SR),
                            "-f", "f32le", "-"], capture_output=True, check=True)
        return np.frombuffer(r.stdout, dtype=np.float32).copy()


def peak_normalize(x: np.ndarray, dbfs: float = -3.0) -> np.ndarray:
    return x * (10 ** (dbfs / 20) / (np.abs(x).max() or 1.0))


def write_wav(path, x: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(str(path), x, sr, subtype="PCM_16")


def ffmpeg_decode(src, dst, sr: int = 48000) -> Path:
    """Decode any audio file to mono 16-bit WAV at `sr`."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(sr),
                    "-c:a", "pcm_s16le", str(dst)], check=True)
    return Path(dst)


def restore(src, dst, sr: int = TTS_SR, chain: str = RESTORE_FILTER) -> Path:
    """Light restoration (see RESTORE_FILTER, or an ffmpeg filter `chain`), output mono at the TTS rate."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    af = f"{chain},aresample={sr}" if chain else f"aresample={sr}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af", af,
                    "-ac", "1", "-c:a", "pcm_s16le", str(dst)], check=True)
    return Path(dst)


def integrated_loudness(path) -> str:
    """EBU R128 integrated loudness line from ffmpeg, for logging."""
    r = subprocess.run(["ffmpeg", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    lines = [l.strip() for l in r.stderr.splitlines() if l.strip().startswith("I:")]
    return lines[-1] if lines else ""
