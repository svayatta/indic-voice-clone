"use strict";
const $ = (id) => document.getElementById(id);
const BUSY = new Set(["uploaded", "queued", "cleaning", "transcribing"]);
const TRAIN_BUSY = new Set(["queued", "preparing", "training"]);
const STATUS_TEXT = { uploaded: "Uploading", queued: "Waiting", cleaning: "Cleaning", transcribing: "Transcribing",
  ready: "Ready", failed: "Failed" };
let config = { languages: {} };
let current = null;      // id of the voice on screen
let chosen = null;       // File or Blob for the new voice
let pollTimer = null;
let panelsFor = null;    // which voice (and trained state) the option panels were drawn for

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : (j.detail?.[0]?.msg || msg); } catch {}
    throw new Error(msg);
  }
  return r.status === 204 ? null : r.json();
}
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
const fmtSec = (s) => s == null ? "" : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

// ---------- options (schema from /api/config, see studio/options.py) ----------
const defaultsOf = (group) => Object.fromEntries(config.options[group].fields.map((f) => [f.key, f.default]));

function renderOptions(id, group, values, unavailable = {}) {
  const g = config.options[group];
  const v = { ...defaultsOf(group), ...(values || {}) };
  const fields = g.fields.map((f) => (unavailable[f.key] ? { ...f, available: false, why: unavailable[f.key] } : f));
  const grid = el("div", { className: "opt-grid" });
  for (const f of fields) {
    let input;
    if (f.type === "bool") input = el("input", { type: "checkbox", checked: !!v[f.key] });
    else if (f.type === "choice") input = el("select", {}, ...f.choices.map(([k, l]) => el("option", { value: k, textContent: l })));
    else input = el("input", { type: "number", min: f.min, max: f.max, step: f.step ?? 1, value: v[f.key] });
    if (f.type === "choice") input.value = v[f.key];
    Object.assign(input.dataset, { key: f.key, type: f.type });
    input.disabled = !f.available;
    const hint = f.help ? el("span", { className: "hint", textContent: f.help }) : "";
    const wrap = f.type === "bool"
      ? el("label", { className: "opt check" }, input, el("span", {}, f.label, el("br"), hint))
      : el("label", { className: "opt" }, el("span", { textContent: f.label }), input, hint);
    if (!f.available) { wrap.classList.add("na"); wrap.title = f.why || "Not installed on this server"; }
    if (f.needs) wrap.dataset.needs = f.needs;
    grid.append(wrap);
  }
  const sync = () => grid.querySelectorAll("[data-needs]").forEach((w) => {
    const on = grid.querySelector(`[data-key="${w.dataset.needs}"]`).checked;
    w.classList.toggle("off", !on);
    w.querySelector("[data-key]").disabled = !on || w.classList.contains("na");
  });
  const setAll = (on) => {
    grid.querySelectorAll('input[type=checkbox]:not(:disabled)').forEach((i) => (i.checked = on));
    sync();
  };
  const head = el("div", { className: "opt-head" }, el("h4", { textContent: g.label }));
  if (fields.filter((f) => f.type === "bool" && f.available).length > 1) {
    head.append(el("button", { type: "button", className: "btn btn-quiet small", textContent: "All on", onclick: () => setAll(true) }),
      el("button", { type: "button", className: "btn btn-quiet small", textContent: "All off", onclick: () => setAll(false) }));
  }
  head.append(el("button", { type: "button", className: "btn btn-quiet small", textContent: "Defaults",
    onclick: () => renderOptions(id, group, null, unavailable) }));
  grid.addEventListener("change", sync);
  $(id).replaceChildren(head, grid);
  sync();
}

function readOptions(id) {
  const o = {};
  $(id).querySelectorAll("[data-key]").forEach((i) => {
    if (i.closest(".na")) return;   // not offered here: the server keeps its default
    o[i.dataset.key] = i.dataset.type === "bool" ? i.checked : i.dataset.type === "choice" ? i.value : Number(i.value);
  });
  return o;
}

function describeClean(c) {
  if (!c) return "";
  const on = [c.declip && "declip", c.highpass && `high-pass ${c.highpass_hz} Hz`,
    c.denoise && `denoise ${c.denoise_db} dB`, c.enhance && `MossFormer2${c.super_res ? " + super-resolution" : ""}`,
    c.normalize && "normalise"].filter(Boolean);
  return `Cleaned with: ${on.length ? on.join(" · ") : "no processing"}`;
}

// ---------- list ----------
async function refreshList() {
  const voices = await api("/api/voices");
  const ul = $("voice-list");
  ul.replaceChildren(...voices.map((v) => {
    const t = v.train.status === "trained" ? "trained" : v.status;
    const cls = t === "ready" || t === "trained" ? t : t === "failed" ? "failed" : "busy";
    const label = t === "trained" ? "Trained" : STATUS_TEXT[t] || t;
    const b = el("button", { type: "button", onclick: () => go(`#/voice/${v.id}`) },
      el("span", { className: "vn", textContent: v.name }), el("span", { className: `pill ${cls}`, textContent: label }));
    if (v.id === current) b.setAttribute("aria-current", "true");
    return el("li", {}, b);
  }));
  $("empty-list").hidden = voices.length > 0;
}

// ---------- routing ----------
function go(hash) { if (location.hash === hash) route(); else location.hash = hash; }
function route() {
  clearTimeout(pollTimer);
  const m = location.hash.match(/^#\/voice\/([0-9a-f]{12})$/);
  current = m ? m[1] : null;
  $("view-create").hidden = !!m;
  $("view-voice").hidden = !m;
  if (m) loadVoice(); else $("f-name").focus();
  refreshList().catch(() => {});
}

// ---------- create ----------
function setChosen(fileOrBlob, name) {
  chosen = fileOrBlob;
  $("drop-idle").hidden = !!fileOrBlob;
  $("drop-chosen").hidden = !fileOrBlob;
  $("drop-recording").hidden = true;
  if (fileOrBlob) {
    $("chosen-name").textContent = name;
    $("chosen-audio").src = URL.createObjectURL(fileOrBlob);
  } else {
    $("chosen-audio").removeAttribute("src");
    $("f-file").value = "";
  }
}

function setupDrop() {
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) setChosen(f, f.name); });
  $("f-file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) setChosen(f, f.name); });
  $("chosen-clear").addEventListener("click", () => setChosen(null));
}

let recorder = null, recTick = null;
async function startRecording() {
  showError("create-error", "");
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } }); }
  catch { return showError("create-error", "Microphone access was not allowed."); }
  const parts = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => parts.push(e.data);
  recorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    clearInterval(recTick);
    const type = recorder.mimeType || "audio/webm";
    const ext = type.includes("mp4") ? "mp4" : type.includes("ogg") ? "ogg" : "webm";
    setChosen(new File(parts, `recording.${ext}`, { type }), `Recording (${$("rec-time").textContent})`);
  };
  const t0 = Date.now();
  $("rec-time").textContent = "0:00";
  recTick = setInterval(() => {
    const s = (Date.now() - t0) / 1000;
    $("rec-time").textContent = fmtSec(s);
    if (s >= 45) recorder.stop();
  }, 250);
  recorder.start();
  $("drop-idle").hidden = true;
  $("drop-recording").hidden = false;
}

async function submitCreate(e) {
  e.preventDefault();
  const name = $("f-name").value.trim();
  if (!name) return showError("create-error", "Give the voice a name.");
  if (!chosen) return showError("create-error", "Add a voice sample first.");
  if (chosen.size > config.sample_max_mb * 1048576) return showError("create-error", `The sample is over ${config.sample_max_mb} MB.`);
  const fd = new FormData();
  fd.append("name", name);
  fd.append("lang", $("f-lang").value);
  fd.append("options", JSON.stringify({ clean: readOptions("o-create-clean"), reference: readOptions("o-create-reference") }));
  fd.append("file", chosen, chosen.name || "sample.webm");
  const btn = $("create-submit");
  btn.disabled = true; btn.textContent = "Uploading…";
  try {
    const v = await api("/api/voices", { method: "POST", body: fd });
    $("create-form").reset(); setChosen(null); showError("create-error", "");
    renderCreateOptions();
    go(`#/voice/${v.id}`);
  } catch (err) { showError("create-error", err.message); }
  finally { btn.disabled = false; btn.textContent = "Clone voice"; }
}

function showError(id, msg) { $(id).textContent = msg; $(id).hidden = !msg; }

// ---------- voice ----------
async function loadVoice() {
  const id = current;
  let v;
  try { v = await api(`/api/voices/${id}`); }
  catch { location.hash = ""; return; }
  if (id !== current) return;
  renderVoice(v);
  const busy = BUSY.has(v.status) || TRAIN_BUSY.has(v.train.status) || v.previews.some((p) => p.status === "queued" || p.status === "running");
  if (busy) pollTimer = setTimeout(() => { loadVoice(); refreshList().catch(() => {}); }, 1500);
}

function setAudio(id, url) { const a = $(id); if (a.dataset.src !== url) { a.dataset.src = url; a.src = url; } }

function renderVoice(v) {
  const base = `/api/voices/${v.id}/audio`;
  $("v-name").textContent = v.name;
  $("v-meta").textContent = [config.languages[v.lang] || v.lang, v.sample_sec ? `${v.sample_sec} s sample` : ""].filter(Boolean).join(" · ");

  // the step each status is working on; "ready" has finished them all
  const at = { uploaded: 1, queued: 1, cleaning: 1, transcribing: 2, ready: 4, failed: -1 }[v.status];
  const labels = ["Uploaded", "Cleaning noise", "Transcribing reference", "Ready"];
  $("v-steps").replaceChildren(...labels.map((l, i) => {
    const cls = i === 0 || i < at || at === 4 ? "done" : i === at ? "now" : "";
    return el("li", { className: cls, textContent: l });
  }));
  if (v.status === "failed") $("v-steps").lastChild.className = "fail";
  showError("v-error", v.status === "failed" ? v.error : "");

  const trainedNow = v.train.status === "trained";
  if (panelsFor !== `${v.id}:${trainedNow}`) {   // draw once per voice, so polling never wipes edits
    panelsFor = `${v.id}:${trainedNow}`;
    renderOptions("o-voice-clean", "clean", v.options?.clean);
    renderOptions("o-voice-reference", "reference", v.options?.reference);
    renderOptions("o-generate", "generate", null, trainedNow ? {} : { trained: "Train the voice first" });
    renderOptions("o-train", "train", v.train.options);
  }
  $("v-clean-used").textContent = describeClean(v.options?.clean);
  $("v-reclean").hidden = BUSY.has(v.status);
  const cleaned = v.sample_sec != null;
  $("v-audio-card").hidden = !v.original;
  if (v.original) setAudio("a-original", `${base}/original`);
  if (cleaned) setAudio("a-clean", `${base}/clean`);
  $("a-clean").parentElement.hidden = !cleaned;

  const ready = v.status === "ready";
  $("v-ref-card").hidden = !ready;
  $("v-try-card").hidden = !ready;
  $("v-train-card").hidden = !ready;
  if (!ready) return;
  setAudio("a-reference", `${base}/reference`);
  const ta = $("v-ref-text");
  if (document.activeElement !== ta && ta.dataset.v !== v.ref_text) { ta.value = v.ref_text || ""; ta.dataset.v = v.ref_text; }
  ["v-ref-text", "v-say"].forEach((i) => $(i).lang = v.lang);

  const trained = v.train.status === "trained";
  $("v-model-badge").textContent = trained ? "Trained clone" : "Instant clone";
  $("v-model-badge").className = `badge ${trained ? "trained" : ""}`;
  $("v-previews").replaceChildren(...v.previews.map((p) => {
    const settings = p.params ? ` · temp ${p.params.temperature}, top-p ${p.params.top_p}` : "";
    const meta = p.status === "done" ? `${p.trained ? "trained" : "instant"} · seed ${p.seed}${settings} · ${p.seconds} s in ${p.gen_time} s`
      : p.status === "failed" ? `failed: ${p.error || ""}` : p.status === "running" ? "generating…" : "waiting…";
    const li = el("li", {}, el("p", { className: "pt", textContent: p.text, lang: v.lang }));
    if (p.status === "done") {
      li.append(el("audio", { controls: true, preload: "none", src: `${base}/preview?id=${p.id}` }));
      li.append(el("button", { className: "btn btn-quiet small", textContent: "Another take",
        onclick: () => generate(v.id, p.text, { ...readOptions("o-generate"), seed: p.seed + 1, takes: 1 }) }));
    }
    li.append(el("span", { className: "pm", textContent: ` ${meta}` }));
    return li;
  }));

  $("v-recordings").replaceChildren(...v.recordings.map((r) =>
    el("li", {}, el("span", { textContent: r.file.replace(/^\d{3}_/, "") }), el("span", { className: "muted", textContent: fmtSec(r.sec) }))));
  const t = v.train;
  const tBusy = TRAIN_BUSY.has(t.status);
  $("v-train").disabled = tBusy || !v.recordings.length;
  $("v-train").textContent = trained ? "Train again" : "Start training";
  $("v-rec-files").disabled = tBusy;
  const st = { none: v.recordings.length ? "" : "Add recordings to train.", queued: "Waiting for other work to finish…",
    preparing: "Cleaning, transcribing and filtering the recordings…", training: "Training… this takes several hours.",
    trained: `Trained on ${t.speech_min ?? "?"} min of speech. Previews now use the trained clone.`, failed: `Training failed: ${t.error}` }[t.status];
  $("v-train-status").textContent = st || "";
  $("v-train-status").className = `small ${t.status === "failed" ? "error" : ""}`;
  $("v-train-log").hidden = !t.log.length;
  $("v-train-log").textContent = t.log.join("\n");
  $("v-train-log").scrollTop = 1e9;
}

async function generate(id, text, options) {
  try { await api(`/api/voices/${id}/previews`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, options }) }); }
  catch (err) { return alert(err.message); }
  clearTimeout(pollTimer); loadVoice();
}

function setupVoice() {
  $("v-say").addEventListener("input", () => {
    const n = $("v-say").value.trim().split(/(?<=[.?!।॥])\s+/).filter(Boolean).length;
    $("v-say-count").textContent = n ? `${n} sentence${n > 1 ? "s" : ""}. The studio generates each sentence as one take and joins them.` : "";
  });
  $("v-generate").addEventListener("click", () => {
    const text = $("v-say").value.trim();
    if (text) generate(current, text, readOptions("o-generate"));
  });
  $("v-ref-save").addEventListener("click", async () => {
    const ref_text = $("v-ref-text").value.trim();
    if (!ref_text) return;
    await api(`/api/voices/${current}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ref_text }) });
    $("ref-saved").hidden = false; setTimeout(() => ($("ref-saved").hidden = true), 1500);
  });
  $("v-delete").addEventListener("click", async () => {
    if (!confirm(`Delete "${$("v-name").textContent}" and all its audio?`)) return;
    await api(`/api/voices/${current}`, { method: "DELETE" });
    go("#/new");
  });
  $("v-rec-files").addEventListener("change", async (e) => {
    const fd = new FormData();
    for (const f of e.target.files) {
      if (f.size > config.recording_max_mb * 1048576) return alert(`${f.name} is over ${config.recording_max_mb} MB.`);
      fd.append("files", f, f.name);
    }
    e.target.value = "";
    $("v-train-status").textContent = "Uploading…";
    try { await api(`/api/voices/${current}/recordings`, { method: "POST", body: fd }); } catch (err) { alert(err.message); }
    loadVoice();
  });
  $("v-reclean-go").addEventListener("click", async () => {
    const body = { clean: readOptions("o-voice-clean"), reference: readOptions("o-voice-reference") };
    try { await api(`/api/voices/${current}/clean`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); }
    catch (err) { return alert(err.message); }
    $("v-reclean").open = false;   // the panels keep what was just chosen
    clearTimeout(pollTimer); loadVoice();
  });
  $("v-train").addEventListener("click", async () => {
    const body = { options: readOptions("o-train") };
    try { await api(`/api/voices/${current}/train`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); }
    catch (err) { return alert(err.message); }
    clearTimeout(pollTimer); loadVoice();
  });
}

function renderCreateOptions() {
  renderOptions("o-create-clean", "clean");
  renderOptions("o-create-reference", "reference");
}

// ---------- start ----------
(async function init() {
  config = await api("/api/config");
  $("f-lang").replaceChildren(...Object.entries(config.languages).map(([k, n]) => el("option", { value: k, textContent: n })));
  $("max-mb").textContent = config.sample_max_mb;
  renderCreateOptions();
  if (config.min_train_minutes) $("min-min").textContent = config.min_train_minutes;
  setupDrop(); setupVoice();
  $("rec-start").addEventListener("click", startRecording);
  $("rec-stop").addEventListener("click", () => recorder && recorder.state !== "inactive" && recorder.stop());
  $("create-form").addEventListener("submit", submitCreate);
  $("new-voice").addEventListener("click", () => go("#/new"));
  window.addEventListener("hashchange", route);
  route();
  setInterval(() => refreshList().catch(() => {}), 5000);
})();
