"""Pinned ALFWorld TextWorld tasks with action-prefix checkpoint restoration.

Only public text feedback and admissible commands reach the actor. Restoring a
checkpoint starts a fresh game and replays the recorded prefix, then checks the
complete public boundary before any experimental branch can run.
"""
from __future__ import annotations

import copy
import difflib
from pathlib import Path

from cave_mem_v3.benchmarks.tasks import task_asset_hash
from cave_mem_v3.common.contracts import PublicContext, digest
from cave_mem_v3.environments.base import EnvStep


class ALFWorldEnvironment:
    def __init__(self, task: dict, *, max_steps: int):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("ALFWorld max_steps must be positive")
        self.task = copy.deepcopy(task)
        self.max_steps = max_steps
        self.root = Path(task["path"]).resolve(strict=True)
        self.gamefile = self.root / "game.tw-pddl"
        if not self.gamefile.is_file():
            raise ValueError("ALFWorld task lacks game.tw-pddl")
        if task_asset_hash(self.root) != task["asset_hash"]:
            raise ValueError("ALFWorld task assets changed since manifest freezing")
        if task.get("instruction") is not None and (not isinstance(task["instruction"], str) or not task["instruction"].strip()):
            raise ValueError("ALFWorld task instruction must be a nonempty string")
        self._env = None
        self._state = None

    def configuration(self):
        return {"adapter": "alfworld-text-prefix-replay.v1", "task_id": self.task["task_id"],
                "asset_hash": self.task["asset_hash"], "gamefile": str(self.gamefile),
                "max_steps": self.max_steps, "textworld_protocol": "start-reset-step-no-domain-randomization"}

    def capabilities(self):
        return {"restore": "fresh_game_and_checked_action_prefix", "restore_certified": True,
                "paired_primary_eligible": True, "randomized_single_supported": True,
                "isolated_sessions": True, "future_randomness": "fixed_game_no_domain_randomization",
                "step_seed_applied": False, "state_fingerprint_scope": "public_text_and_action_prefix"}

    def _start(self):
        import textworld

        self.close()
        infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True)
        self._env = textworld.start(str(self.gamefile), request_infos=infos)
        state = self._env.reset()
        feedback = str(state["feedback"])
        instruction = self.task.get("instruction")
        if not instruction:
            marker = "Your task is to:"
            instruction = feedback.split(marker, 1)[1].strip().splitlines()[0].strip() if marker in feedback else feedback
            self.task["instruction"] = instruction
        self._state = {"steps": 0, "history": [], "actions": [], "observation": feedback,
                       "admissible": sorted(set(state["admissible_commands"])),
                       "terminal": False, "success": None}
        return self.observe()

    def reset(self, seed: int = 0):
        return self._start()

    def _require(self):
        if self._state is None:
            raise RuntimeError("reset or restore ALFWorld before use")
        return self._state

    def observe(self):
        state = self._require()
        return PublicContext(self.task["task_id"], self.task["instruction"],
            state["observation"], {"phase": "terminal" if state["terminal"] else "acting",
            "benchmark": "alfworld-text", "task_type": self.task.get("task_type", "")},
            tuple(copy.deepcopy(state["history"])), tuple(state["admissible"]),
            max(0, self.max_steps - state["steps"]))

    def snapshot(self):
        return {"schema_version": "alfworld-checkpoint.v1", "configuration": self.configuration(),
                "state": copy.deepcopy(self._require())}

    def restore(self, checkpoint):
        if checkpoint.get("schema_version") != "alfworld-checkpoint.v1" or checkpoint.get("configuration") != self.configuration():
            raise ValueError("ALFWorld checkpoint configuration mismatch")
        desired = checkpoint.get("state")
        if not isinstance(desired, dict) or desired.get("terminal") is True:
            raise ValueError("ALFWorld checkpoint must be a nonterminal state")
        actions = desired.get("actions")
        if not isinstance(actions, list) or len(actions) != desired.get("steps"):
            raise ValueError("ALFWorld checkpoint action prefix is inconsistent")
        self._start()
        for action in actions:
            result = self.step({"type": "text", "command": action})
            if not result.valid or result.terminal:
                raise ValueError("ALFWorld action prefix did not restore a live state")
        if self.snapshot() != checkpoint:
            raise ValueError("ALFWorld restored state differs from checkpoint")
        return self.observe()

    def state_fingerprint(self):
        return digest(self.snapshot())

    def terminal_status(self):
        state = self._require()
        return state["terminal"], state["success"] if state["terminal"] else None

    def finalize(self):
        """Score an unfinished text game as failure at the frozen budget boundary."""
        state = self._require()
        if not state["terminal"]:
            state["terminal"] = True
            state["success"] = False
            state["observation"] = "Task execution ended at the frozen step budget."
        return EnvStep(self.observe(), True, state["success"], True, 0.0)

    def validate_action(self, action):
        if self._require()["terminal"]:
            return False, "ALFWorld game already ended"
        if (not isinstance(action, dict) or set(action) != {"type", "command"}
                or action.get("type") not in ("text", "json_object")):
            return False, "expected text(command) action"
        command = action.get("command")
        if not isinstance(command, str) or command not in self._require()["admissible"]:
            return False, "command is not in the public admissible action set"
        return True, None

    def repair_action(self, action, context=None):
        """Map a provider action error to one deterministic public action."""
        admissible = tuple(context.admissible_actions if context is not None
                           else self.observe().admissible_actions)
        if not admissible:
            return None
        command = action.get("command") if isinstance(action, dict) else None
        if isinstance(command, str):
            if command in admissible:
                return {"type": "text", "command": command}
            ranked = sorted(admissible, key=lambda item: (-difflib.SequenceMatcher(None, command, item).ratio(), item))
            if ranked and difflib.SequenceMatcher(None, command, ranked[0]).ratio() >= 0.35:
                return {"type": "text", "command": ranked[0]}
        preferred = next((item for item in admissible if item in ("look", "inventory")), admissible[0])
        return {"type": "text", "command": preferred}

    def step(self, action, seed: int = 0):
        valid, reason = self.validate_action(action)
        if not valid:
            return EnvStep(self.observe(), False, None, False, None, reason)
        command = action["command"]
        world, _, done = self._env.step(command)
        state = self._require()
        state["steps"] += 1
        state["actions"].append(command)
        state["observation"] = str(world["feedback"])
        # Some providers return a semantically valid command with the wrapper
        # type `json_object`. Normalize that agent protocol error at the
        # environment boundary so it remains replayable as a text action.
        normalized_action = {"type": "text", "command": command}
        state["history"].append({"action": normalized_action, "observation": state["observation"]})
        state["admissible"] = sorted(set(world["admissible_commands"]))
        state["terminal"] = bool(done or state["steps"] >= self.max_steps)
        state["success"] = bool(world["won"]) if state["terminal"] else None
        return EnvStep(self.observe(), state["terminal"], state["success"], True, 0.0)

    def close(self):
        if self._env is not None:
            self._env.close()
        self._env = None
        self._state = None
