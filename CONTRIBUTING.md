# Contributing and running experiments

The goal is one thing: a better clone from the software on `main`. That
means a voice closer to the speaker, words that are right, and an accent
that native listeners accept. Everything here is so that a change can be shown
to do that, or shown not to, without anyone's opinion deciding it.

## Licence of contributions

The repository is released under the [Apache License 2.0](LICENSE). By
submitting a pull request, you confirm that you have the right to contribute
the change and that you license it under Apache 2.0 (section 5 of the
licence). Contributors keep the copyright in their own work.

Audio is never contributed or licensed: see [NOTICE](NOTICE) and rule 5.

## Ground rules

1. **One hypothesis, one change.** Before training, write down what you expect
   and the single thing you change (data, a filter, a training setting, the
   denoiser). Keep everything else as it is on `main`. An experiment that changes
   three things cannot say which one helped.

2. **The experiment is the recipe.** [`ci/recipe.json`](ci/recipe.json) is the
   pipeline that main ships: the `voiceclone` commands that turn the shared data
   into an adapter. An experiment is a pull request that changes the recipe, the
   code it runs, or both. Results run on your own machine help you explore. They
   are not evidence. The evidence is the quality gate's run of your branch.

3. **The benchmark is fixed, and nobody trains on it.** The benchmark spec and its
   held-out recordings stay on the gate runner and change only by the owner, as
   a new benchmark version. The gate refuses an adapter whose training clips come
   from a held-out recording. Don't score against a file you trained on, and
   don't choose a conditioning clip by its score against the reference.

4. **Verbatim transcripts only.** Training text comes from IndicConformer
   (`voiceclone transcribe`) or a human, never from Whisper or another model
   that paraphrases or translates dialect speech.

5. **No data in git.** No audio, transcripts of the speaker, model weights,
   personal or family names, or local file links go in the repository. Recordings
   and adapters live in the shared data folder on the gate runner. New source
   recordings go to the owner, who adds them there. CI rejects audio and weight
   files, files over 1 MB, home-folder paths and `sys.path` edits.

6. **Runnable anywhere.** Every path is a command-line argument. Use the installed
   package (`uv pip install -e ".[all]" -c constraints.txt`). Don't hard-code a
   device; `pick_device()` chooses one. New pipeline steps get a `voiceclone`
   subcommand so a recipe can use them.

7. **Tests say why.** New code comes with tests that would fail if the behaviour
   that matters changed. The gate's metrics are not a substitute for tests, and
   tests are not a substitute for the gate.

8. **Small pull requests.** One experiment per pull request. Refactors and
   tooling go in their own pull requests, which the gate checks for "no worse",
   not for "better".

9. **Accent claims need ears.** The numbers measure voice similarity and
   intelligibility, not accent. A pull request that claims a better accent
   attaches a blind A/B result (`voiceclone listen-page`, then `compare --answers`).
   It needs at least three native speakers of the dialect who passed the hidden
   real-recording checks.

10. **Failed experiments are results.** Close a pull request that fails the gate
    with its gate comment in place, and add one line on what you learnt to the
    pull request. Don't delete branches that the gate scored.

## What has to pass before merging

| Check | Runs | On | Blocks the merge when |
|---|---|---|---|
| `ci` | every push and pull request | GitHub's machines | a test fails or a repository rule is broken |
| `quality-gate` | when the owner adds the `benchmark` label | the self-hosted gate runner, with the real model | any metric is worse than main with 95% confidence; for pull requests also labelled `experiment`, when no metric is better |

The metrics are voice similarity to the held-out recordings, character error
rate, length error, and whether generation stopped by itself. Each is
compared take by take against main's run of the same benchmark (same
sentences, seeds and conditioning clip), with a bootstrap over sentences. A
difference counts only when its 95% interval excludes zero.

Only the owner merges, and only green pull requests: both checks pass, and a
listening test is attached where rule 9 applies. `main` is protected: changes
arrive only through pull requests, `test` and `watermark` must pass, open
review conversations must be resolved, and nobody, admins included, can push
or force-push to it directly. The quality gate and the listening test are
rules the owner checks, not settings.

## Running an experiment

What to try next, in order, is in [docs/experiment-plan.md](docs/experiment-plan.md);
what has been tried is in [docs/experiments/log.md](docs/experiments/log.md).


1. Branch from `main` and fill in the experiment card in the pull request
   template.
2. Change the recipe or the code. Run `pytest` and `python ci/check_hygiene.py`
   locally.
3. Open the pull request and add the `experiment` label. The owner reviews the
   diff, then adds `benchmark` to start the gate. A gate run trains and scores
   the model, which takes hours. Its report is posted as a comment and as the
   `quality-gate` status.
4. If you push new commits, the owner relabels after reviewing them.

See [ci/README.md](ci/README.md) for how the gate runner is set up.

## Releasing

A release publishes the package to PyPI (`.github/workflows/release.yml`), so
`uv tool install "pratidhvani[all]"` works anywhere, and attaches the
wheel and sdist to a GitHub release. The Docker images publish on every merge
to `main`, separately.

One-time setup (owner):
1. On pypi.org, under Your account -> Publishing, add a pending trusted
   publisher: project `pratidhvani`, owner `svayatta`, repository
   `pratidhvani`, workflow `release.yml`, environment `pypi`.
2. In the repository, Settings -> Environments -> New environment `pypi`, with
   yourself as required reviewer, so no release goes out unapproved.

Each release: bump `version` in `pyproject.toml` in a pull request, merge it,
then tag the merge commit and push the tag:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

The workflow refuses a tag that doesn't match `pyproject.toml`. It builds the
wheel and checks that it installs and runs. Then it waits for your approval
of the `pypi` environment, publishes, and creates the GitHub release.
