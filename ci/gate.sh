#!/usr/bin/env bash
# The quality gate, run by .github/workflows/quality-gate.yml on the self-hosted runner.
#   ci/gate.sh baseline BASE_DIR           score a checkout of main (cached)
#   ci/gate.sh compare BASE_DIR HEAD_DIR   score both (cached), gate HEAD against BASE
#   ci/gate.sh listening BASE_DIR PR SHA COMMENTS   score /listening codes (listening-test.yml)
# Environment: VOICECLONE_DATA  the shared, read-only data folder (holds bench.json)
#              VOICECLONE_STATE the gate's own folder: runs, venvs, model caches
#              REQUIRE_IMPROVEMENT=1 for experiments
#              PR, SHA, ACCENT=1   for a pull request labelled `accent`: also write a rating page to
#                                  $VOICECLONE_STATE/listening/pr-PR/SHA/ for the owner to share
#              LISTENERS           comma-separated GitHub logins whose codes count (listening)
# Writes gate-report.md in the current folder; exit status 1 = the gate failed.
set -euo pipefail
: "${VOICECLONE_DATA:?set VOICECLONE_DATA to the shared data folder}"
: "${VOICECLONE_STATE:?set VOICECLONE_STATE to the gate folder}"
SPEC="$VOICECLONE_DATA/bench.json"
REPORT="$PWD/gate-report.md"
mkdir -p "$VOICECLONE_STATE/runs" "$VOICECLONE_STATE/venvs"

# Everything a score depends on: the package code (minus the web studio, which no recipe or
# benchmark runs), pinned dependencies, the recipe and the benchmark.
# Two checkouts with the same key produce the same scores, so a run is never repeated.
run_key() {
  { git -C "$1" ls-tree -r HEAD src | grep -v $'\tsrc/voiceclone/studio/'
    git -C "$1" rev-parse HEAD:pyproject.toml HEAD:constraints.txt HEAD:ci/recipe.json
    shasum -a 256 < "$SPEC"; } | shasum -a 256 | cut -c1-16
}

# These set globals instead of printing: bash turns `set -e` off inside $(...),
# so a failed step there would not stop the gate.
venv_for() {   # CHECKOUT KEY -> VENV, a virtualenv with that checkout installed
  VENV="$VOICECLONE_STATE/venvs/$2"
  if [[ ! -x "$VENV/bin/voiceclone" ]]; then
    rm -rf "$VENV"
    uv venv -q --python 3.12 "$VENV"
    uv pip install -q --python "$VENV/bin/python" "$1[all]" -c "$1/constraints.txt"
  fi
  touch "$VENV"
}

score() {   # CHECKOUT -> RUN, the run folder with bench/scores.json
  local key
  key=$(run_key "$1"); RUN="$VOICECLONE_STATE/runs/$key"
  if [[ -f "$RUN/bench/scores.json" ]]; then
    echo "reusing scored run $key" >&2
    return
  fi
  venv_for "$1" "$key"
  rm -rf "$RUN.partial"
  # cwd = the state folder, so every run shares one models/ cache (ECAPA etc.)
  (cd "$VOICECLONE_STATE" && "$VENV/bin/voiceclone" run-recipe --recipe "$1/ci/recipe.json" \
      --data "$VOICECLONE_DATA" --work "$RUN.partial" --spec "$SPEC") >&2
  mv "$RUN.partial" "$RUN"
}

find "$VOICECLONE_STATE/venvs" -mindepth 1 -maxdepth 1 -mtime +14 -exec rm -rf {} +

abspath() { (cd "$1" && pwd); }

case "${1:-}" in
  baseline)
    score "$(abspath "$2")" ;;
  compare)
    base=$(abspath "$2"); head=$(abspath "$3")
    base_key=$(run_key "$base"); head_key=$(run_key "$head")
    if [[ "$base_key" == "$head_key" ]]; then
      if [[ "${REQUIRE_IMPROVEMENT:-0}" == 1 ]]; then
        # shellcheck disable=SC2016  # the backticks are Markdown
        printf '### Quality gate: FAIL\n\n- labelled `experiment`, but nothing that affects the voice changed (code, dependencies, recipe and benchmark are identical to main)\n' > "$REPORT"
        exit 1
      fi
      printf '### Quality gate: PASS (nothing that affects the voice changed)\n\nThe package code, dependencies, recipe and benchmark are identical to main, so no run was needed.\n' > "$REPORT"
      exit 0
    fi
    score "$base"; base_run=$RUN
    score "$head"; head_run=$RUN
    # the rules come from main, so a pull request cannot loosen its own gate
    venv_for "$base" "$base_key"
    if ! "$VENV/bin/voiceclone" gate --help > /dev/null 2>&1; then
      echo "main has no gate command yet; using the pull request's" >&2
      venv_for "$head" "$head_key"
    fi
    gate="$VENV/bin/voiceclone"
    flag=""; [[ "${REQUIRE_IMPROVEMENT:-0}" == 1 ]] && flag="--require-improvement"
    rc=0
    "$gate" gate "$base_run/bench" "$head_run/bench" --report "$REPORT" $flag || rc=$?
    if [[ "${ACCENT:-0}" == 1 ]]; then
      # the rating page embeds benchmark audio, so it stays on this machine; the owner shares it
      dir="$VOICECLONE_STATE/listening/pr-$PR/$SHA"
      rater="$gate"
      "$rater" listen-page --help | grep -q -- --rate || { venv_for "$head" "$head_key"; rater="$VENV/bin/voiceclone"; }
      mkdir -p "$dir"
      "$rater" listen-page "$base_run/bench" "$head_run/bench" --rate --out "$dir/listen.html" >&2
      # shellcheck disable=SC2016  # the backticks are Markdown
      printf '\n### Listening test\n\nA rating page for main against this pull request is on the gate runner, in `listening/pr-%s/%s/listen.html`. The owner shares it privately with the listeners, who post `/listening CODE` here.\n' \
        "$PR" "$SHA" >> "$REPORT"
    fi
    exit $rc ;;
  listening)
    base=$(abspath "$2"); pr=$3; sha=$4; comments=$5
    key="$VOICECLONE_STATE/listening/pr-$pr/$sha/listen.key.json"
    if [[ ! -f "$key" ]]; then
      printf '{"state": "pending", "description": "no rating page for this commit yet: the gate builds it (benchmark label)", "report": ""}\n' > status.json
      exit 0
    fi
    venv_for "$base" "$(run_key "$base")"
    "$VENV/bin/voiceclone" listening-status --key "$key" --comments "$comments" \
      --listeners "${LISTENERS:?set LISTENERS}" --out status.json ;;
  *)
    echo "usage: ci/gate.sh baseline BASE_DIR | compare BASE_DIR HEAD_DIR | listening BASE_DIR PR SHA COMMENTS" >&2; exit 2 ;;
esac
