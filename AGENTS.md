# CAVE-Mem-v3 Agent Instructions

## Current scope

The repository implements phase 1 in software. The next task is an exploratory
real-benchmark pilot: collect historical trajectories with a frozen actor,
generate repair candidates with a teacher model, audit checkpoint restoration,
and replay candidate repairs. Do not describe fixture tests as real results.
Do not implement phases 2-4 unless the user explicitly requests them.

## Research constraints

- Read `README.md`, `docs/phase1.md`, and `docs/experiment_design.md` before
  changing experiment behavior. The proposal controls the research design;
  its confirmed Eq. 2 correction uses lowercase `e_t` in `x_t`. Uppercase `E`
  in the memory tuple still denotes source evidence.
- Keep history, target-train, validation, and test tasks separated by task,
  template, and known dependency groups. Never use test outcomes to generate
  memories or tune pilot choices.
- A teacher proposal is a hypothesis, not verified evidence. Only independent
  source replay can set `v=1`. Do not weaken restoration checks, validity
  checks, or scoring rules to make a candidate pass.
- Freeze and record the benchmark revision, task assets, actor and teacher
  model identities, prompts, budgets, and seeds for each run. The current
  `server.toml` and benchmark catalog are templates, not ready-to-run settings.
- Keep credentials, local configurations, weights, and `runs/` out of Git.
  Preserve failed and incomplete run artifacts; do not silently overwrite or
  retry them.

## Working method

- Start from a named Git commit and use a separate branch for changes.
- Run the focused checks before reporting completion. State clearly which
  results are software tests and which are real benchmark observations.
- After every code change, list each added or modified file and what changed
  in it. Report experiment settings, actual outputs, and unresolved limits.
- If a task falls outside the reviewed Terminal-Bench filesystem protocol,
  report the incompatibility and propose a specific adapter or protocol change
  for review before running it.
