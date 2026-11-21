"""The detect command must be useful to people and automation without loading AudioSeal."""
import json
from argparse import Namespace

import pytest

from voiceclone.cli import cmd_detect
from voiceclone.watermark import Detection, verdict


@pytest.mark.parametrize(("detection", "expected"), [
    (Detection(1.0, 0), "ours"),
    (Detection(1.0, 8), "other"),
    (Detection(0.0, 8), "none"),
])
def test_verdict_does_not_need_audioseal(detection, expected):
    assert verdict(detection) == expected


def run_detect(monkeypatch, capsys, detections, json_output=False):
    class FakeWatermark:
        def __init__(self):
            self.detections = iter(detections)

        def detect(self, wav, sr):
            return next(self.detections)

    monkeypatch.setattr("voiceclone.audio.load16", lambda path: path)
    monkeypatch.setattr("voiceclone.watermark.Watermark", FakeWatermark)
    args = Namespace(files=[f"clip-{i}.wav" for i in range(len(detections))], json=json_output)
    cmd_detect(args)
    return capsys.readouterr().out


def test_detect_exits_zero_only_when_every_file_is_ours(monkeypatch, capsys):
    out = run_detect(monkeypatch, capsys, [Detection(1.0, 0), Detection(0.8, 1)])
    assert out.count("made with pratidhvani") == 2


@pytest.mark.parametrize("not_ours", [Detection(0.9, 8), Detection(0.1, 8)])
def test_detect_exits_one_when_any_file_is_not_ours(monkeypatch, capsys, not_ours):
    with pytest.raises(SystemExit) as exc:
        run_detect(monkeypatch, capsys, [Detection(1.0, 0), not_ours])
    assert exc.value.code == 1


def test_detect_json_has_machine_readable_results_and_exit_code(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        run_detect(monkeypatch, capsys, [Detection(0.25, 7)], json_output=True)
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out) == [{
        "file": "clip-0.wav",
        "verdict": "none",
        "probability": 0.25,
        "bit_errors": 7,
        "ours": False,
    }]


def test_detect_keeps_missing_watermark_warning_for_mixed_input(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run_detect(monkeypatch, capsys, [Detection(1.0, 0), Detection(0.1, 8)])
    assert "doesn't prove a recording is genuine" in capsys.readouterr().out
