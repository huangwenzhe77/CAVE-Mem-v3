# CAVE-Mem-v3 Agent Instructions

## Current scope

The repository implements the core four-phase software workflow, including both
phase 2 collection routes, phase 3 learning and phase 4 frozen evaluation.
The software-only goal passed I-01 acceptance on 2026-11-16 (353 offline tests
and two independent reviews). The exploratory real-benchmark source pilot
remains a separate future task; it is not automatically authorized by completion
of the software goal. Do not describe fixture tests as real results or use them
to select research parameters. See docs/software_handoff.md for scope and limits.

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

## Single-task execution workflow (2026-11-16)

- For each code task, use exactly one implementation agent and two independent,
  read-only review agents, as explicitly requested by the user. Assign the
  implementation agent ownership of code edits; have one reviewer check research
  validity and data boundaries and the other check engineering correctness.
  The primary agent coordinates, integrates, and verifies without becoming a
  second code writer.

- The live workflow is defined in `docs/task_cards.md`; `docs/v1_plan.md` is
  historical design context. Read the selected task card before implementation.
- Execute only one clearly identified task card at a time. State its ID,
  verify its prerequisites, and stay within its Goal, Context, Rules, and
  Acceptance. The user explicitly authorized continuous sequential execution on
  2026-11-16: after a card passes both independent reviews and required tests,
  record its results, derive the next card from the proposal, the four flowcharts
  and current code, and proceed without waiting for another user message.
- The thread goal tracks the whole remaining software workflow. Never overlap
  implementation of cards or mark the overall goal complete after one card.
  Only complete the goal when the whole required software workflow passes its
  tests and reviews. A user pause/stop overrides continuous execution.
- The roadmap is not authorization to implement every listed stage. Planning
  later stages does not override the current-scope restrictions above.
- After each card, report changed files, checks, actual outputs and limitations;
  update its status before beginning the next eligible card. Preserve blocked
  and failed results. Formal data collection and real-model experiments remain
  outside this software-only goal.
