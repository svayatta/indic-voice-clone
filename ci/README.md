# The quality gate runner

`quality-gate` runs `ci/recipe.json` with the real 3B model, then benchmarks
the result against main's run of the same benchmark. GitHub's free machines
have neither the memory for the model nor the private recordings, so this runs
on a self-hosted runner: the Mac that holds the data.

## What it does

```
ci/gate.sh compare BASE HEAD
  for BASE and HEAD:   key = hash(src/, pyproject.toml, constraints.txt, ci/recipe.json, bench.json)
                       cached run for key?  -> reuse
                       else: venv with that checkout -> voiceclone run-recipe -> voiceclone benchmark
  voiceclone gate (main's version) BASE_RUN HEAD_RUN -> gate-report.md, exit 1 = fail
```

A pull request that changes nothing in that key (docs, CI, the studio's page)
passes without a run. Each run trains a LoRA, which takes hours; main's run is
cached, so a pull request pays for one run.

## Data folder (`VOICECLONE_DATA`, read-only for recipes)

```
bench.json        the frozen benchmark spec (voiceclone benchmark --spec); absolute paths inside
holdout/          held-out clips + metadata.jsonl, whole recordings never used for training
enroll/           clips that are certainly the speaker (metadata.jsonl), for build-clips
chunks/<TAG>/     transcribed chunk folders (segments.jsonl + transcripts.jsonl), one per recording
```

`ci/recipe.json` refers to these as `{data}/...`. The recordings named in
`bench.json` under `holdout.sources` must not appear in the recipe; the
benchmark's leak check stops the run if they do.

## Setting up the runner (once)

1. **Use a separate macOS user** (for example `gate`) that can read the data
   folder and nothing else of yours. Any collaborator with write access can
   push a workflow that runs on this runner, and the gate itself runs
   pull-request code there. A separate account limits what that code can reach.
2. As that user, install `uv` and `ffmpeg`, accept the IndicConformer terms on
   Hugging Face, and run `hf auth login` with a read-only token.
3. Register a repository runner (Settings -> Actions -> Runners -> New
   self-hosted runner, macOS ARM64) with the extra label `voiceclone-gate`, and
   install it as a service (`./svc.sh install && ./svc.sh start`).
4. Put the variables in the runner's `.env` file:
   ```
   VOICECLONE_DATA=/Users/Shared/voiceclone/data
   VOICECLONE_STATE=/Users/Shared/voiceclone/gate
   ```
5. In the repository's Settings -> Secrets and variables -> Actions ->
   Variables, set `GATE_APPROVER` to the GitHub login allowed to start gate
   runs. Until it is set, the label does nothing.
6. Create the labels `benchmark`, `experiment` and `accent`.
7. For accent claims, set the repository variable `LISTENERS` to the GitHub
   logins of the native listeners, comma-separated with no spaces. Only their
   `/listening` codes count. Optionally add `listening-test` to the required
   checks on `main`: pull requests without the `accent` label pass it at once.
8. Optionally score main ahead of time: Actions -> quality-gate -> Run workflow.
   Otherwise the first gated pull request scores it (and caches it).

## Listening tests

For a pull request labelled `accent`, `ci/gate.sh compare` also writes a
rating page for main against the pull request to
`$VOICECLONE_STATE/listening/pr-<number>/<commit>/listen.html`, with its key
next to it. The page embeds benchmark audio, so it never leaves the runner
through GitHub: send it to the listeners yourself. Each listener posts
`/listening CODE` on the pull request. `listening-test.yml` collects the codes
from listeners named in `LISTENERS`, scores them here with
`ci/gate.sh listening` (main's code, against the key), and posts the status
and ratings. A new commit needs a new gate run and a new page; codes for an
older page don't count.

## Disk

Each scored run keeps its clips, adapter and benchmark audio under
`$VOICECLONE_STATE/runs/<key>`, about 0.4 GB for a training set like the reference one (310 MB of clips, a 93 MB adapter). Virtualenvs unused for 14 days
are removed automatically. Delete old runs by hand when space runs short. A
deleted run is rebuilt only if it is needed again.

## Bootstrapping

`main` needs `ci/recipe.json` and the `run-recipe` and `gate` commands
before it can be scored. The gate works for pull requests opened after the
change that adds them has merged.
