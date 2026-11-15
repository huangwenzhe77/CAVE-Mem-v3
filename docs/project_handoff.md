# CAVE-Mem-v3 Project Handoff

Historical handoff from the checkpoint-sampling implementation. For the current
code and server clone/run instructions, use [server_handoff.md](server_handoff.md).
The branch, test count, and stage-2 status below are superseded.

Status: 2026-11-15. This document is for the next coding agent continuing the
existing project. The current working tree contains substantial uncommitted
work; preserve it and continue in place. Do not reset, clean, commit, or push
unless the user asks. The latest full test run passed: 139 tests.

## Start Here

1. Confirm the checkout is `D:\XJTU_2026_2029\Github-code\CAVE-Mem-v3` on branch
   `codex/phase2-target-sampling` and inspect `git status` before editing.
2. Read this document, `docs/experiment_design.md`, `docs/phase1.md`, and
   `docs/architecture.md`. The proposal source is
   `D:\XJTU_2026_2029\Literature-review\llm-reasoning\proposal-2.pdf`.
3. Treat the proposal as authoritative. The confirmed notation correction is
   that Eq. 2 uses lowercase `e`; uppercase `E` remains source evidence in the
   memory tuple.
4. Continue only with the user's next requested step. They want work completed
   one step at a time, with a file-by-file explanation after each step. Do not
   implement later stages preemptively.

## Research Design and User Decisions

CAVE-Mem-v3 is organized around four stages:

1. Build and freeze memories from historical failures and paired source
   replays.
2. On target rollouts, sample an action-before checkpoint, cache the
   no-memory action `a0`, retrieve a memory, freeze `x_t`, then run the
   Silent/Remind experiment.
3. Learn/select a reminder policy using target intervention data.
4. Freeze the system and evaluate it on held-out test tasks.

Benchmark roles are `history`, `target_train`, `validation` (with calibration
and selection subroles), and `test`. Split connected tasks/templates together
to avoid leakage. Test tasks must not be used for development or policy
selection.

The user's explicit choices and corrections:

- Eq. 2's exposure parameter is lowercase `e`; do not change uppercase `E` in
  source-memory evidence.
- The user's intended memory replay gate is K=2: if the two correction replays
  pass, add the memory to the memory bank. Do not silently replace this with a
  95% confidence requirement. Clearly label K=2 evidence as pilot evidence;
  this is not a statistical confidence claim.
- Qwen2.5-7B can be the actor. Model weights may remain absent locally until
  the server run; do not download them unless requested.
- Silent means execute the already cached original `a0`. Remind is the
  alternate intervention from the same frozen checkpoint and state.
- In target sampling, risk is assessed online at each action-before node. The
  first selected point ends further checkpoint sampling for that rollout. The
  preparation workflow then freezes state and stops before validating or
  executing the selected action.
- The user explicitly requested that the current larger body of work remain
  uncommitted until the full second part is complete. No commit/push has been
  made for the current changes.
- Responses and handoffs should explain newly added/modified files and their
  purpose. Keep explanations concise and do not implement unrequested future
  steps.

## Completed Work

### Stage 1: Historical Memory Pipeline

The first-stage pipeline exists: task manifest/splits, historical rollout
collection, candidate generation, source replay verification, and frozen bank
loading. It records complete successful, failed, and infrastructure-error
attempts. Replay K is configured and run as paired original/correction replay;
the K=2 pilot decision is the user's chosen engineering criterion, not a claim
of statistical power. Detailed workflow, Terminal-Bench and H20 notes, command
examples, and pilot results are in [`phase1.md`](phase1.md). The experiment
design and known research caveats are in [`experiment_design.md`](experiment_design.md).

### Stage 2: Checkpoint Sampling, Retrieval, and Pre-Randomization State

Implemented in the current uncommitted working tree:

- Risk-biased online checkpoint sampling. The decision is recorded at each
  eligible node, selection stops further sampling for that rollout, and the
  rollout behavior is explicit for collection versus preparation mode.
- Frozen-encoder top-1 retrieval of at most one verified memory, including a
  no-match outcome and query/bank provenance hashes.
- `prepare-target`: once a checkpoint is selected, actor proposes and caches
  `a0`; retrieval and `x_t` are produced; the rollout stops before action
  validation/execution. There is no assignment, verifier call, or terminal
  outcome in this workflow.
- Full pre-treatment state record with the current field interpretations:
  `m`, source evidence `E`, `v`, applicability `rho`, exposure `e`, continuous
  divergence `d`, remaining computation budget `b`, and `F_omega` (frozen
  encoder identity plus text and structured features). It also stores `a0`,
  public context, retrieval metadata, and hashes.
- `e` is only derived from a frozen typed exposure rubric. Missing rules,
  missing per-memory rules, conflicts, or inconclusive predicates produce `?`;
  retrieval similarity is not used to infer exposure.
- `d` is continuous source/target public-prefix-action cosine distance with no
  hand-set threshold or applicability claim. `rho` is recomputed from public
  predicates; unavailable evidence remains unknown. `b` currently includes
  model-call and environment-step counts; no global token budget exists in the
  runtime contract.
- Private/outcome context keys are filtered, unsafe private fields in cached
  actions and exposure predicates are rejected, and exposure rules are fully
  checked before actor construction.
- The record explicitly says `randomization_status=not_assigned` and
  `outcome_access=none`.

`Silent` in the future experiment executes cached `a0`. This stage does not
randomly assign Silent/Remind and does not execute either arm.

## Current Git and File State

Current branch: `codex/phase2-target-sampling`, tracking
`origin/codex/phase2-target-sampling`. Current modifications are uncommitted.
The checkout is CAVE-Mem-v3; do not confuse it with the older CAVE-Mem-v2
checkout.

Modified tracked files:

- `pyproject.toml`: dependency/entry-point support for target retrieval.
- `src/cave_mem_v3/agents/runtime.py`: checkpoint-aware preparation callback;
  cache selected action and stop before validation/step; omit verifier result
  in pre-treatment mode.
- `src/cave_mem_v3/cli.py`: `collect-target`, `retrieve-target`, and
  `prepare-target` commands and their configuration.
- `src/cave_mem_v3/target_data/sampling.py`: online risk-based sampler and
  per-node selection records.
- `tests/test_cli.py`, `tests/test_runtime.py`,
  `tests/test_target_sampling.py`: CLI, runtime, sampler tests.

New untracked files:

- `src/cave_mem_v3/target_data/collection.py`: target rollout collection,
  frozen run identity, checkpoint sampling, and pre-treatment preparation.
- `src/cave_mem_v3/target_data/retrieval.py` and
  `retrieval_collection.py`: retrieval API, encoder adapter, top-1 retrieval,
  and persisted retrieval records.
- `src/cave_mem_v3/target_data/pre_treatment_state.py`: `x_t` construction,
  evidence/rule checks, public-field filtering, feature map, and state hashes.
- `tests/test_pre_treatment_state.py`, `tests/test_target_prepare.py`,
  `tests/test_target_retrieval.py`, `tests/test_target_retrieval_cli.py`, and
  `tests/test_target_retrieval_collection.py`: state contract, preparation,
  retrieval, CLI and collection tests.

This inventory reflects the handoff time. Always re-check `git status` first:
agents or the user may have changed the workspace after this document was
written.

## Validation and Limits

Latest command:

```powershell
python scripts/run_tests.py
```

Result: 139 tests passed. The preparation integration tests verify that the
selected action is retained, hashes are valid, no validation/step occurs after
selection, and no verifier/outcome is recorded. These are software tests, not
real target benchmark results.

Known limits to preserve in future work:

- An encoder's identity/revision is recorded as declared metadata; encoder
  weight files are not content-hashed. CLI downloads are disabled and a local
  model directory is required.
- If no verified memory exists, retrieval returns no match; `e`, `rho`, and
  memory-dependent features must remain unknown where the proposal requires it.
- Current sampling/prepare CLI accepts `target_train` and `validation`, not
  `test`.
- No Silent/Remind random assignment, branch replay, reminder prompt, paired
  outcome collection, policy training, or final evaluation has been
  implemented by this stage.
- Checkpoint snapshot/recovery fidelity for paired interventions remains to
  be specified and verified before experiments that require replay.

## Next Step for the Incoming Agent

The next implementation boundary is the proposal's processing of frozen
`x_t` before randomized assignment. First inspect the exact proposal equation
and the current `x_t` schema. Decide and document the policy output contract
(e.g. action recommendation versus assignment probability) and how the
proposal distinguishes recoverable from non-recoverable checkpoints. Do not
implement random assignment, treatment execution, or outcome collection until
that contract and the checkpoint replay semantics are clear and the user asks
for that next step.

After the user authorizes implementation, modify only the next requested
piece, add focused tests for its information/timing contract, run
`python scripts/run_tests.py`, and report every added/modified file and its
role. Leave changes in the current working tree; do not commit or push until
the user requests it or explicitly changes the existing instruction.
