"""Listener ratings decide accent claims (CONTRIBUTING rule 9), so the tests
check what would let a wrong claim through: a page that gives away which
run a take came from, a listener who can't recognise the speaker counting
anyway, codes for an older page counting, someone outside the allowlist
voting, and a status that turns green before three listeners agree."""
import json
import re
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from voiceclone.cli import main
from voiceclone.rating import (PAGE, decode_answers, encode_answers, listening_status, rating_results,
                               write_rating_page)

from test_listen import bench_run


@pytest.fixture
def page(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    for i in range(3):
        sf.write(real / f"r{i}.wav", np.zeros(800, np.float32), 16000)
    sentences = [f"ವಾಕ್ಯ {i}" for i in range(6)]
    a, b = bench_run(tmp_path / "run_a", sentences, real), bench_run(tmp_path / "run_b", sentences, real)
    html, key = write_rating_page(a, b, tmp_path / "rate.html", max_sentences=5, anchor_every=5, log=lambda *x: None)
    return a, b, html, json.loads(key.read_text(encoding="utf-8"))


def listener(key, accent_a, accent_b, anchor_voice, name):
    """Ratings that give run A `accent_a`, run B `accent_b` and the real anchors `anchor_voice`."""
    out = {}
    for item, t in key["items"].items():
        if t["kind"] == "anchor":
            out[item] = {"voice": anchor_voice, "accent": 5, "natural": 5, "pacing": 5}
        else:
            acc = accent_a if t["run"] == "A" else accent_b
            out[item] = {"voice": 4, "accent": acc, "natural": 4, "pacing": 4}
    return decode_answers(encode_answers(key["page_id"], out), name)


def test_the_page_hides_which_run_each_take_came_from(page, tmp_path):
    a, b, html_path, key = page
    html = html_path.read_text(encoding="utf-8")
    for leak in ("run_a", "run_b", ".wav", str(tmp_path), '"A"', '"B"', "anchor"):
        assert leak not in html
    assert not re.search(r"""src=["']https?:""", html)              # works offline
    takes = [t for t in key["items"].values() if t["kind"] == "take"]
    anchors = [t for t in key["items"].values() if t["kind"] == "anchor"]
    # every sentence is rated in both runs, so ratings compare take by take
    assert sorted(t["text"] for t in takes if t["run"] == "A") == sorted(t["text"] for t in takes if t["run"] == "B")
    assert len(takes) == 10 and len(anchors) == 2
    assert all(t["ref"] != t["path"] for t in anchors)               # real vs another real, never itself
    assert html.count("data:audio/wav;base64,") == 2 * len(key["items"])


def test_a_code_carries_ratings_only_and_rejects_anything_else():
    code = encode_answers("p1", {"r00": {"voice": 5, "accent": 3, "natural": 4, "pacing": 2}})
    assert decode_answers(code, "x") == {"page_id": "p1", "listener": "x", "answers": [
        {"item": "r00", "voice": 5, "accent": 3, "natural": 4, "pacing": 2}]}
    with pytest.raises(ValueError):
        decode_answers("hello", "x")
    with pytest.raises(ValueError):
        decode_answers(encode_answers("p1", {"r00": {"voice": 9}}), "x")
    with pytest.raises(ValueError):
        decode_answers("vc1." + "A" * 9000, "x")


@pytest.mark.skipif(not shutil.which("node"), reason="needs node to run the page's script")
def test_the_code_the_page_shows_decodes_in_python(tmp_path):
    # the page compresses in the browser and the workflow decompresses in Python;
    # run the page's own code() function, not a copy of it
    fn = re.search(r"async function code\(\) \{.*?\n\}", PAGE, re.S).group(0)
    script = (f'const PAGE_ID = "p9"; const ITEMS = [{{id: "r00"}}, {{id: "r01"}}];'
              f'const DIMS = [["voice"], ["accent"], ["natural"], ["pacing"]];'
              f'const ratings = {{r00: {{voice: 5, accent: 2, natural: 4, pacing: 3}}}};'
              f'{fn}\ncode().then(c => console.log(c));')
    (tmp_path / "c.js").write_text(script)
    out = subprocess.run(["node", str(tmp_path / "c.js")], capture_output=True, text=True, check=True).stdout.strip()
    assert decode_answers(out, "x")["answers"] == [{"item": "r00", "voice": 5, "accent": 2, "natural": 4, "pacing": 3}]


def test_ratings_count_only_listeners_who_recognise_the_speaker(page):
    a, b, _, key = page
    good = [listener(key, 2, 5, 5, f"l{i}") for i in range(3)]
    deaf = listener(key, 5, 1, 2, "rates the real speaker 2")
    res = rating_results(key, good + [deaf], a, b, n_boot=500)
    assert res["qualified"] == 3
    acc = res["dimensions"]["accent"]
    assert acc["mean_a"] == 2 and acc["mean_b"] == 5 and acc["mean_real"] == 5
    assert acc["verdict"] == "B better" and acc["pairs"] == 15
    # named in the other order, the same answers favour the other run
    assert rating_results(key, good, b, a, n_boot=500)["dimensions"]["accent"]["verdict"] == "A better"


def test_a_listeners_last_code_for_this_page_is_the_one_that_counts(page):
    a, b, _, key = page
    first, second = listener(key, 5, 1, 5, "l0"), listener(key, 1, 5, 5, "l0")
    old_page = {**listener(key, 5, 1, 5, "l1"), "page_id": "an older commit's page"}
    res = rating_results(key, [first, second, old_page], a, b, n_boot=500)
    assert res["qualified"] == 1 and res["dimensions"]["accent"]["mean_b"] == 5


def test_the_status_waits_for_three_listeners_and_needs_a_real_improvement(page):
    a, b, _, key = page
    better = [listener(key, 2, 5, 5, f"l{i}") for i in range(3)]
    assert listening_status(rating_results(key, better[:2], n_boot=500))["state"] == "pending"
    assert listening_status(rating_results(key, better, n_boot=500))["state"] == "success"
    same = [listener(key, 4, 4, 5, f"l{i}") for i in range(3)]
    assert listening_status(rating_results(key, same, n_boot=500))["state"] == "failure"


def test_only_allowlisted_listeners_codes_count(page, tmp_path):
    a, b, _, key = page
    key_path = tmp_path / "rate.key.json"
    comments = []
    for i, who in enumerate(["ann", "bo", "cy", "mallory", "mallory2"]):
        code = encode_answers(key["page_id"], {k: {"voice": 5, "accent": 5 if v["run"] == "B" else 2,
                                                   "natural": 4, "pacing": 4}
                                              for k, v in key["items"].items()})
        comments.append({"login": who, "body": f"/listening {code}"})
    comments.append({"login": "ann", "body": "looks good"})
    (tmp_path / "c.jsonl").write_text("".join(json.dumps(c) + "\n" for c in comments))
    out = tmp_path / "status.json"
    main(["listening-status", "--key", str(key_path), "--comments", str(tmp_path / "c.jsonl"),
          "--listeners", "Ann, bo", "--n-boot", "200", "--out", str(out)])
    status = json.loads(out.read_text())
    assert status["state"] == "pending" and status["ratings"]["qualified"] == 2   # cy and mallory don't count
