"""Listener ratings: how close is each clone take to the speaker, and where does it fall short?

`listen-page` asks "which of two takes sounds more like him". The rating page
asks each listener to rate one take at a time against a real recording of the
speaker, 1-5, on voice, accent, naturalness and pacing, with a free-text
"what sounds wrong". Takes from run A and run B are mixed in random order, so
a listener can't tell which run a take came from.

Hidden anchors: some items are a *real* recording of the speaker rated against
another real recording. They give the ceiling (what "sounds like him" scores
when it is him), and they test the listener: one who rates the speaker's own
voice below ANCHOR_MIN on voice is not listening, or can't hear the speaker in
these recordings, so their ratings are left out.

The page shows a short answer code: ratings only, no free text (free text may
quote or describe the speaker, so it goes back to the owner in the downloaded
file, never into a public comment). The code means nothing without the
.key.json, which stays with the owner. `compare --key KEY --answers ...` and
`listening-status` accept the downloaded files or the codes.
"""
import base64
import binascii
import hashlib
import json
import random
import zlib
from pathlib import Path

import numpy as np

from voiceclone.compare import bootstrap_ci, load_run, verdict
from voiceclone.jsonl import write_json
from voiceclone.listen import _data_uri

DIMENSIONS = ("voice", "accent", "natural", "pacing")
DIMENSION_TEXT = {"voice": "voice", "accent": "accent", "natural": "naturalness", "pacing": "pacing"}
ANCHOR_MIN = 4.0          # a listener whose mean voice rating of real-vs-real anchors is lower is left out
ANCHOR_EVERY = 5          # one hidden anchor per this many rated takes
MIN_LISTENERS = 3         # CONTRIBUTING rule 9: at least three qualified native listeners
MAX_CODE_CHARS = 8000     # a code is a few hundred characters; anything far longer is not one
CODE_PREFIX = "vc1."


# ---------- the page ----------
def build_rating_trials(a: dict, b: dict, max_sentences: int = 8, anchor_every: int = ANCHOR_EVERY,
                        seed: int = 0) -> list[dict]:
    """Rating items in listening order: {id, kind ("take" | "anchor"), run ("A" | "B" | "real"),
    text, seed, ref, path}. A take and its counterpart from the other run (same sentence and
    seed) are both included, against the same reference recording, so their ratings can be
    compared pair by pair without the choice of reference entering the difference."""
    if a["spec_hash"] != b["spec_hash"]:
        raise ValueError("runs used different benchmark specs")
    real = a["real"]
    if len(real) < 2:
        raise ValueError("the rating page needs at least two real recordings: one to compare with, "
                         "and one to hide as an anchor")
    rng = random.Random(seed)
    tb = {(t["text"], t["seed"]): t for t in b["takes"]}
    shared = [(t, tb[(t["text"], t["seed"])]) for t in a["takes"] if (t["text"], t["seed"]) in tb]
    if not shared:
        raise ValueError("the two runs share no (sentence, seed) takes")
    by_text = {}
    for ta, tb_ in shared:
        by_text.setdefault(ta["text"], (ta, tb_))       # one seed per sentence keeps the page short
    pairs = rng.sample(list(by_text.values()), min(max_sentences, len(by_text)))
    items = []
    for ta, tb_ in pairs:
        ref = rng.choice(real)["path"]
        for run, t in (("A", ta), ("B", tb_)):
            items.append({"kind": "take", "run": run, "text": t["text"], "seed": t["seed"],
                          "ref": ref, "path": t["path"]})
    for _ in range(max(1, len(items) // anchor_every) if anchor_every else 0):
        ref, test = rng.sample(real, 2)
        items.append({"kind": "anchor", "run": "real", "text": test["text"], "seed": None,
                      "ref": ref["path"], "path": test["path"]})
    rng.shuffle(items)
    return [{"id": f"r{i:02d}", **t} for i, t in enumerate(items)]


def write_rating_page(run_a, run_b, out_html, max_sentences: int = 8, anchor_every: int = ANCHOR_EVERY, seed: int = 0,
                      log=print):
    """Writes out_html and its .key.json; returns both paths."""
    a, b = load_run(run_a), load_run(run_b)
    items = build_rating_trials(a, b, max_sentences, anchor_every, seed)
    page_id = hashlib.sha256(json.dumps(["rate", a["spec_hash"], str(run_a), str(run_b), seed]).encode()).hexdigest()[:12]
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    page = [{"id": t["id"], "ref": _data_uri(t["ref"]), "test": _data_uri(t["path"])} for t in items]
    out.write_text(PAGE.replace("__PAGE_ID__", page_id)
                   .replace("__ITEMS__", json.dumps(page).replace("</", "<\\/")), encoding="utf-8")
    key = out.with_suffix(".key.json")
    write_json(key, {"mode": "rate", "page_id": page_id, "run_a": str(run_a), "run_b": str(run_b),
                     "spec_hash": a["spec_hash"], "items": {t["id"]: {k: v for k, v in t.items() if k != "id"}
                                                            for t in items}})
    n_anchor = sum(t["kind"] == "anchor" for t in items)
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(items)} items, {n_anchor} anchors); "
        f"keep {key} to score the answers")
    return out, key


# ---------- answer codes ----------
def encode_answers(page_id: str, ratings: dict) -> str:
    """The code the page shows: page id and ratings only (never free text)."""
    body = {"p": page_id, "r": {i: [r.get(d) for d in DIMENSIONS] for i, r in ratings.items()}}
    raw = zlib.compress(json.dumps(body, separators=(",", ":")).encode(), 9)
    return CODE_PREFIX + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_answers(code: str, listener: str = "") -> dict:
    """An answer code (as posted in a `/listening CODE` comment) -> the answers-file shape.
    Anything malformed raises ValueError, so one bad comment can be skipped without stopping the rest."""
    try:
        return _decode(code, listener)
    except (KeyError, TypeError, AttributeError, binascii.Error, zlib.error, json.JSONDecodeError) as e:
        raise ValueError(f"not a readable answer code ({type(e).__name__})") from None


def _decode(code: str, listener: str) -> dict:
    code = code.strip()
    if not code.startswith(CODE_PREFIX) or len(code) > MAX_CODE_CHARS:
        raise ValueError("not an answer code from a rating page")
    raw = base64.urlsafe_b64decode(code[len(CODE_PREFIX):] + "=" * (-len(code) % 4))
    d = zlib.decompressobj()
    text = d.decompress(raw, 64_000)
    if d.unconsumed_tail:
        raise ValueError("answer code too large")
    body = json.loads(text)
    answers = []
    for item, vals in body["r"].items():
        row = {"item": str(item)}
        for dim, v in zip(DIMENSIONS, vals):
            if v is not None:
                if not isinstance(v, int) or not 1 <= v <= 5:
                    raise ValueError(f"rating {v!r} is not 1-5")
                row[dim] = v
        answers.append(row)
    return {"page_id": str(body["p"]), "listener": listener, "answers": answers}


# ---------- scoring ----------
def _run_label(key, run_a, run_b) -> dict:
    same = lambda x, y: Path(x).resolve() == Path(y).resolve()
    if same(run_a, key["run_a"]) and same(run_b, key["run_b"]):
        return {"A": "A", "B": "B", "real": "real"}
    if same(run_a, key["run_b"]) and same(run_b, key["run_a"]):
        return {"A": "B", "B": "A", "real": "real"}
    raise ValueError(f"the rating key is for {key['run_a']} vs {key['run_b']}, not these runs")


def rating_results(key: dict, answer_files: list[dict], run_a=None, run_b=None, n_boot: int = 10000) -> dict:
    """Mean ratings per run and dimension, B - A with a 95% interval, and the anchor ceiling,
    from qualified listeners only. A listener's later answers for the same page replace their
    earlier ones. run_a/run_b (either order) default to the runs the key names."""
    label = _run_label(key, run_a, run_b) if run_a and run_b else {"A": "A", "B": "B", "real": "real"}
    latest = {}
    for ans in answer_files:
        if ans.get("page_id") != key["page_id"]:
            continue    # answers for an older page (an earlier commit) or another test
        latest[ans.get("listener") or f"anonymous {len(latest) + 1}"] = ans
    n_anchors = sum(t["kind"] == "anchor" for t in key["items"].values())
    listeners, kept = [], []
    for who, ans in latest.items():
        rows = [(key["items"][r["item"]], r) for r in ans["answers"] if r.get("item") in key["items"]]
        anchor = [r["voice"] for t, r in rows if t["kind"] == "anchor" and r.get("voice")]
        # every anchor must be rated: one easy anchor at 5 with the rest skipped proves nothing
        ok = n_anchors > 0 and len(anchor) == n_anchors and float(np.mean(anchor)) >= ANCHOR_MIN
        listeners.append({"listener": who, "rated": len(rows), "anchor_voice": float(np.mean(anchor)) if anchor else None,
                          "qualified": ok})
        if ok:
            kept.append(rows)
    out = {"listeners": listeners, "qualified": len(kept), "dimensions": {}}
    for dim in DIMENSIONS:
        by_run = {"A": [], "B": [], "real": []}
        diffs = {}   # (listener, sentence, seed) -> {run: rating}
        for n, rows in enumerate(kept):
            for t, r in rows:
                if r.get(dim) is None:
                    continue
                run = label[t["run"]]
                by_run[run].append(r[dim])
                if t["kind"] == "take":
                    diffs.setdefault((n, t["text"], t["seed"]), {})[run] = r[dim]
        # one bootstrap group per listener: five sentences from one person are one voter, not five
        groups = {}
        for (n, _, _), d in diffs.items():
            if "A" in d and "B" in d:
                groups.setdefault(n, []).append(d["B"] - d["A"])
        row = {f"mean_{k.lower()}": float(np.mean(v)) if v else None for k, v in by_run.items()}
        row["pairs"] = sum(len(g) for g in groups.values())
        if groups:
            diff, lo, hi = bootstrap_ci(list(groups.values()), n=n_boot)
            row.update(diff=diff, ci=[lo, hi], verdict=verdict(lo, hi, higher_is_better=True))
        else:
            row.update(diff=None, ci=None, verdict="not rated")
        out["dimensions"][dim] = row
    return out


def format_ratings(res: dict) -> str:
    lines = [f"ratings (1-5) from {res['qualified']} qualified listener(s); anchor = real vs real"]
    for dim, r in res["dimensions"].items():
        f = lambda v: "-" if v is None else f"{v:.2f}"
        ci = "" if r["ci"] is None else f", B-A {r['diff']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]"
        lines.append(f"  {DIMENSION_TEXT[dim]:11s} A {f(r['mean_a'])}  B {f(r['mean_b'])}  anchor {f(r['mean_real'])}"
                     f"{ci} ({r['pairs']} pairs) -> {r['verdict']}")
    for li in res["listeners"]:
        anchor = "no anchors rated" if li["anchor_voice"] is None else f"rated the real anchors {li['anchor_voice']:.1f}"
        lines.append(f"  listener {li['listener']}: {li['rated']} items, {anchor}"
                     + ("" if li["qualified"] else " (left out: every anchor must be rated, "
                        f"averaging {ANCHOR_MIN:g} or more)"))
    return "\n".join(lines)


def listening_status(res: dict, dimension: str = "accent", min_listeners: int = MIN_LISTENERS) -> dict:
    """The `listening-test` commit status for a pull request that claims a better accent:
    pending until min_listeners qualified listeners have rated, then success only when
    B (the pull request) is rated better than A (main) on `dimension` with 95% confidence."""
    r = res["dimensions"][dimension]
    n = res["qualified"]
    if n < min_listeners:
        return {"state": "pending", "description": f"{n} of {min_listeners} qualified listeners so far"}
    name = DIMENSION_TEXT[dimension]
    if r["ci"] is None:
        return {"state": "pending", "description": f"no paired {name} ratings yet"}
    ci = f"B-A {r['diff']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}], {n} listeners"
    if r["ci"][0] > 0:
        return {"state": "success", "description": f"{name} rated better: {ci}"}
    if r["ci"][1] < 0:
        return {"state": "failure", "description": f"{name} rated worse: {ci}"}
    return {"state": "failure", "description": f"{name} not rated better: {ci}"}


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rating test</title>
<style>
:root { --bg: #fbfaf7; --fg: #1d1c1a; --muted: #6b6760; --card: #ffffff; --line: #e3dfd6; --accent: #8a4b14; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #171614; --fg: #ece9e2; --muted: #a39e94; --card: #211f1c; --line: #38352f; --accent: #e0a060; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.4rem; margin: 0 0 8px; }
.muted { color: var(--muted); }
.item { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 16px 0; }
.item h2 { font-size: 1rem; margin: 0 0 12px; }
.players { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
@media (max-width: 520px) { .players { grid-template-columns: 1fr; } }
.players div { display: flex; flex-direction: column; gap: 4px; }
audio { width: 100%; }
fieldset { border: 0; padding: 0; margin: 8px 0; }
legend { font-weight: 600; padding: 0; }
label { margin-right: 14px; white-space: nowrap; }
textarea, input[type=text] { font: inherit; width: 100%; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px;
  background: var(--card); color: var(--fg); }
input[type=text] { width: auto; }
.bar { position: sticky; bottom: 0; background: var(--bg); border-top: 1px solid var(--line);
       padding: 12px 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; justify-content: center; }
button { background: var(--accent); color: var(--bg); border: 0; border-radius: 6px; padding: 8px 16px; font: inherit; cursor: pointer; }
#code { display: none; margin: 12px auto; max-width: 760px; padding: 0 16px; }
#code textarea { font-family: ui-monospace, monospace; font-size: 13px; }
</style></head>
<body><main>
<h1>How close is each recording to the speaker?</h1>
<p class="muted">Each item has a real recording of the speaker (<b>Reference</b>) and a recording to rate (<b>Test</b>).
Listen to both, then rate the test recording from 1 (nothing like the speaker) to 5 (just like the speaker) on each line.
Some test recordings are real. When done, press <b>Get answer code</b> and post <code>/listening</code> followed by the
code on the pull request, or press <b>Download answers</b> and send the file back. Only the file includes your notes.</p>
<p><label>Your name or initials <input type="text" id="listener" maxlength="40"></label></p>
<div id="items"></div>
</main>
<div id="code"><p>Post this on the pull request as one comment:</p><textarea id="codetext" rows="4" readonly></textarea></div>
<div class="bar"><span id="progress" class="muted"></span><button id="getcode">Get answer code</button>
<button id="save">Download answers</button></div>
<script>
const PAGE_ID = "__PAGE_ID__";
const ITEMS = __ITEMS__;
const DIMS = [["voice", "Voice: does it sound like the same person?"], ["accent", "Accent: is it the speaker's accent and dialect?"],
              ["natural", "Naturalness: does it sound like real speech, not a machine?"], ["pacing", "Pacing: speed, pauses and rhythm"]];
const STORE = "rate-" + PAGE_ID;
let saved = {};
try { saved = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) {}
const ratings = saved.ratings || {}, notes = saved.notes || {};
const root = document.getElementById("items");
const esc = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
ITEMS.forEach((t, i) => {
  const box = document.createElement("section");
  box.className = "item";
  let h = `<h2>Item ${i + 1} of ${ITEMS.length}</h2><div class="players">`
    + `<div><b>Reference</b><audio controls preload="none" src="${t.ref}"></audio></div>`
    + `<div><b>Test</b><audio controls preload="none" src="${t.test}"></audio></div></div>`;
  DIMS.forEach(([d, q]) => {
    h += `<fieldset><legend>${q}</legend>`;
    [1, 2, 3, 4, 5].forEach(v => {
      const c = (ratings[t.id] || {})[d] === v ? " checked" : "";
      h += `<label><input type="radio" name="${t.id}|${d}" value="${v}"${c}> ${v}</label>`;
    });
    h += "</fieldset>";
  });
  h += `<label class="muted" style="white-space:normal">What sounds wrong? (optional; stays in the downloaded file)`
    + `<textarea name="${t.id}|note" rows="2">${esc(notes[t.id] || "")}</textarea></label>`;
  box.innerHTML = h;
  box.addEventListener("change", e => {
    const [id, d] = e.target.name.split("|");
    if (d === "note") notes[id] = e.target.value; else (ratings[id] = ratings[id] || {})[d] = Number(e.target.value);
    persist();
  });
  root.appendChild(box);
});
const who = document.getElementById("listener");
who.value = saved.listener || "";
who.addEventListener("input", persist);
function done() { return ITEMS.filter(t => DIMS.every(([d]) => (ratings[t.id] || {})[d])).length; }
function persist() {
  try { localStorage.setItem(STORE, JSON.stringify({ ratings, notes, listener: who.value })); } catch (e) {}
  document.getElementById("progress").textContent = `${done()} of ${ITEMS.length} rated`;
}
persist();
function ask() { return done() === ITEMS.length || confirm(`Only ${done()} of ${ITEMS.length} rated. Continue anyway?`); }
async function code() {
  const r = {};
  ITEMS.forEach(t => { if (ratings[t.id]) r[t.id] = DIMS.map(([d]) => ratings[t.id][d] || null); });
  const bytes = new TextEncoder().encode(JSON.stringify({ p: PAGE_ID, r }));
  const z = await new Response(new Blob([bytes]).stream().pipeThrough(new CompressionStream("deflate"))).arrayBuffer();
  let s = ""; new Uint8Array(z).forEach(b => { s += String.fromCharCode(b); });
  return "vc1." + btoa(s).replace(/\\+/g, "-").replace(/\\//g, "_").replace(/=+$/, "");
}
document.getElementById("getcode").addEventListener("click", async () => {
  if (!ask()) return;
  document.getElementById("codetext").value = "/listening " + await code();
  document.getElementById("code").style.display = "block";
  document.getElementById("codetext").select();
});
document.getElementById("save").addEventListener("click", () => {
  if (!ask()) return;
  const out = { page_id: PAGE_ID, listener: who.value.trim(), time: new Date().toISOString(),
    answers: ITEMS.filter(t => ratings[t.id] || notes[t.id]).map(t => ({ item: t.id, ...(ratings[t.id] || {}), note: notes[t.id] || "" })) };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], { type: "application/json" }));
  a.download = `ratings_${PAGE_ID}_${out.listener || "anon"}.json`;
  a.click();
});
</script></body></html>
"""
