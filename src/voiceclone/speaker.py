"""Speaker voiceprints with SpeechBrain ECAPA-TDNN, or its ResNet-TDNN as a second opinion (both Apache 2.0)."""
import random
from pathlib import Path

import numpy as np

ECAPA_ID = "speechbrain/spkrec-ecapa-voxceleb"
RESNET_ID = "speechbrain/spkrec-resnet-voxceleb"   # another architecture: a check on ECAPA's own biases


class SpeakerEncoder:
    def __init__(self, savedir="models/ecapa", device: str = "cpu", source: str = ECAPA_ID):
        from speechbrain.inference.speaker import EncoderClassifier

        self.enc = EncoderClassifier.from_hparams(source=source, savedir=str(savedir),
                                                  run_opts={"device": device})

    def embed(self, y16: np.ndarray) -> np.ndarray:
        """L2-normalised embedding of a 16 kHz mono waveform."""
        import torch

        with torch.no_grad():
            e = self.enc.encode_batch(torch.from_numpy(np.ascontiguousarray(y16)).unsqueeze(0)).squeeze()
        return torch.nn.functional.normalize(e, dim=0).numpy()

    def embed_file(self, path) -> np.ndarray:
        from voiceclone.audio import load16

        return self.embed(load16(path))

    def centroid_from_clips(self, clips_dir, n: int = 40, shuffle_seed: int | None = None) -> np.ndarray:
        """Unit-norm mean embedding of up to `n` clips listed in clips_dir/metadata.jsonl."""
        from voiceclone.jsonl import read_jsonl

        clips = read_jsonl(Path(clips_dir) / "metadata.jsonl")
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(clips)
        return centroid([self.embed_file(Path(clips_dir) / c["file"]) for c in clips[:n]])


def centroid(embeddings) -> np.ndarray:
    c = np.mean(np.stack(list(embeddings)), 0)
    return c / np.linalg.norm(c)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
