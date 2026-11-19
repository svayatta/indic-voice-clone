"""Inaudible watermark on every take the synthesizer makes.

AudioSeal (Meta, MIT licence, weights included) hides a 16-bit message in
the audio. It survives MP3, AAC and Opus compression, telephone-band
resampling, cropping and volume changes. We embed a fixed tag, so the
detector can tell our audio from other AudioSeal users'.

It marks what this software generates. It cannot stop someone who edits the
code, or who runs a trained adapter with other software, from making unmarked
audio. Not finding the mark doesn't prove a recording is genuine.
"""
from dataclasses import dataclass

import numpy as np

SR = 16000            # AudioSeal works at 16 kHz; other rates are resampled to it and back
TAG = 0xA51C          # 16-bit message meaning "made with pratidhvani"
MIN_SAMPLES = 1600    # 0.1 s at 16 kHz; anything shorter is left as it is
DETECT_THRESHOLD = 0.5
MAX_BIT_ERRORS = 2


def tag_bits(tag: int = TAG):
    import torch

    return torch.tensor([[(tag >> (15 - i)) & 1 for i in range(16)]])


@dataclass
class Detection:
    probability: float    # share of the audio the detector finds marked, 0..1
    bit_errors: int       # how many of the 16 message bits differ from our tag

    @property
    def marked(self) -> bool:
        return self.probability >= DETECT_THRESHOLD

    @property
    def ours(self) -> bool:
        return self.marked and self.bit_errors <= MAX_BIT_ERRORS


class Watermark:
    """Loads the models on first use and runs them on the CPU: a few
    seconds of speech take well under a second there, and it keeps the GPU
    free for generation."""

    def __init__(self):
        self._gen = self._det = None

    def _resample(self, x, sr_from: int, sr_to: int):
        import torchaudio.functional as AF

        return x if sr_from == sr_to else AF.resample(x, sr_from, sr_to)

    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        import torch

        if len(wav) * SR < MIN_SAMPLES * sr:
            return wav
        if self._gen is None:
            from audioseal import AudioSeal

            self._gen = AudioSeal.load_generator("audioseal_wm_16bits").eval()
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).view(1, 1, -1)
        with torch.inference_mode():
            mark = self._gen.get_watermark(self._resample(x, sr, SR), message=tag_bits())
            mark = self._resample(mark, SR, sr)[..., : x.shape[-1]]
        return np.clip((x + mark).view(-1).numpy(), -1.0, 1.0)

    def detect(self, wav: np.ndarray, sr: int) -> Detection:
        import torch

        if self._det is None:
            from audioseal import AudioSeal

            self._det = AudioSeal.load_detector("audioseal_detector_16bits").eval()
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).view(1, 1, -1)
        with torch.inference_mode():
            p, msg = self._det.detect_watermark(self._resample(x, sr, SR))
        errors = int((msg.round().int().cpu() != tag_bits()).sum())
        return Detection(round(float(p), 3), errors)
