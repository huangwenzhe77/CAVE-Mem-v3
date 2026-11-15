# CAVE-Mem-v3 Agent Instructions

## Current scope

The repository implements phase 1 and both phase 2 collection routes in software.
The next research task is an exploratory real-benchmark source pilot: collect
historical trajectories with a frozen actor, generate repair candidates with a
teacher model, audit checkpoint restoration, and replay candidate repairs.
Do not describe fixture tests as real results. Do not implement phases 3-4
unless the user explicitly requests them.

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

## Single-task execution workflow (2026-11-15)

- For code implementation, use two coding agents and one independent review
  agent, as explicitly requested by the user. Split file ownership to avoid
  concurrent edits; the primary agent integrates and verifies the result.

- The user's current workflow is defined in `docs/v1_plan.md` and
  `docs/task_cards.md`. Read the selected task card before implementation.
- Execute only one clearly identified task card per turn. State its ID,
  verify its prerequisites, and stay within its Goal, Context, Rules, and
  Acceptance. A request to continue selects only the next eligible card.
- The roadmap is not authorization to implement every listed stage. Planning
  later stages does not override the current-scope restrictions above.
- After the selected card is complete, report every changed file, checks,
  actual outputs, and limitations; update that card's status and stop. Do not
  automatically start the next card. Preserve blocked and failed results.
