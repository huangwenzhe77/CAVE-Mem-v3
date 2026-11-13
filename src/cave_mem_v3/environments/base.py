"""Execution protocol and a deterministic development fixture, not a benchmark."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Protocol

from cave_mem_v3.common.contracts import PublicContext, digest, finite


@dataclass(frozen=True)
class EnvStep:
    context: PublicContext
    terminal: bool
    success: bool | None
    valid: bool = True
    tool_cost: float | None = 0.0
    error: str | None = None

    def __post_init__(self):
        if type(self.terminal) is not bool or type(self.valid) is not bool:
            raise ValueError("terminal and valid must be boolean")
        if self.success is not None and type(self.success) is not bool:
            raise ValueError("success must be boolean or unknown")
        if self.terminal and type(self.success) is not bool:
            raise ValueError("terminal steps require an explicit outcome")
        if self.tool_cost is not None:
            finite(self.tool_cost, "tool_cost", minimum=0)


class Environment(Protocol):
    def reset(self, seed: int = 0) -> PublicContext: ...
    def observe(self) -> PublicContext: ...
    def snapshot(self) -> dict: ...
    def restore(self, checkpoint: dict) -> PublicContext: ...
    def state_fingerprint(self) -> str: ...
    def configuration(self) -> dict: ...
    def capabilities(self) -> dict: ...
    def terminal_status(self) -> tuple[bool, bool | None]: ...
    def validate_action(self, action: dict) -> tuple[bool, str | None]: ...
    def step(self, action: dict, seed: int = 0) -> EnvStep: ...
    def close(self) -> None: ...


class PreparationEnvironment:
    """Fixture: operating without a required preparation irreversibly fails.

    Source collection with ScriptedActor and safe_intent=False fails; replacing
    its first action by prepare allows a successful continuation. Every relevant
    state variable is serialized. This deliberately simple fixture is never LLM
    or real-benchmark evidence.
    """
    def __init__(self, *, task_id="fixture", requires_preparation=True,
                 safe_intent=False, max_steps=4):
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be nonempty")
        if type(requires_preparation) is not bool or type(safe_intent) is not bool:
            raise ValueError("fixture controls must be boolean")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.task_id, self.requires_preparation = task_id, requires_preparation
        self.safe_intent, self.max_steps = safe_intent, max_steps
        self._state = None

    def configuration(self):
        return {"adapter": "preparation-fixture.v3", "task_id": self.task_id,
                "requires_preparation": self.requires_preparation,
                "safe_intent": self.safe_intent, "max_steps": self.max_steps,
                "research_role": "fixture_only"}

    def capabilities(self):
        return {"restore_certified": True, "restore": "complete_deterministic_state",
                "isolated_sessions": True, "future_randomness": "deterministic",
                "step_seed_applied": True, "research_role": "fixture_only"}

    def reset(self, seed=0):
        self._state = {"phase": "ready", "prepared": False, "steps": 0,
                       "terminal": False, "success": None, "history": []}
        return self.observe()

    def _require(self):
        if self._state is None:
            raise RuntimeError("reset or restore before execution")
        return self._state

    def observe(self):
        state = self._require()
        instruction = "Prepare the resource when required, operate, then finish."
        if self.safe_intent:
            instruction += " Your current plan begins with preparation."
        return PublicContext(self.task_id, instruction,
            f"phase={state['phase']}; prepared={state['prepared']}; preparation_required={self.requires_preparation}",
            {"phase": state["phase"], "prepared": state["prepared"],
             "requires_preparation": self.requires_preparation,
             "intent": "prepare" if self.safe_intent else "operate"},
            tuple(copy.deepcopy(state["history"])),
            () if state["terminal"] else ("prepare", "operate", "finish"),
            max(0, self.max_steps - state["steps"]))

    def snapshot(self):
        return {"schema_version": "preparation-checkpoint.v3",
                "configuration": self.configuration(), "state": copy.deepcopy(self._require())}

    def restore(self, checkpoint):
        if checkpoint.get("schema_version") != "preparation-checkpoint.v3" or checkpoint.get("configuration") != self.configuration():
            raise ValueError("fixture checkpoint configuration mismatch")
        state = copy.deepcopy(checkpoint["state"])
        if set(state) != {"phase", "prepared", "steps", "terminal", "success", "history"}:
            raise ValueError("invalid fixture state")
        if state["phase"] not in ("ready", "operated", "finished", "failed"):
            raise ValueError("invalid phase")
        if type(state["steps"]) is not int or not 0 <= state["steps"] <= self.max_steps:
            raise ValueError("invalid step budget")
        if type(state["prepared"]) is not bool or type(state["terminal"]) is not bool or not isinstance(state["history"], list):
            raise ValueError("invalid fixture state types")
        if (state["terminal"] and type(state["success"]) is not bool) or (not state["terminal"] and state["success"] is not None):
            raise ValueError("invalid outcome state")
        self._state = state
        return self.observe()

    def state_fingerprint(self):
        return digest(self.snapshot())

    def terminal_status(self):
        state = self._require()
        return state["terminal"], state["success"]

    def validate_action(self, action):
        if self._require()["terminal"]:
            return False, "environment already terminal"
        valid = (isinstance(action, dict) and set(action) == {"type"}
                 and action["type"] in ("prepare", "operate", "finish"))
        return valid, None if valid else "expected prepare/operate/finish"

    def step(self, action, seed=0):
        valid, reason = self.validate_action(action)
        if not valid:
            return EnvStep(self.observe(), False, None, valid=False, error=reason)
        state = self._require()
        state["steps"] += 1
        operation = action["type"]
        if operation == "prepare":
            state["prepared"] = True
        elif operation == "operate":
            if self.requires_preparation and not state["prepared"]:
                state.update(phase="failed", terminal=True, success=False)
            else:
                state["phase"] = "operated"
        else:
            success = state["phase"] == "operated"
            state.update(phase="finished" if success else "failed", terminal=True, success=success)
        if not state["terminal"] and state["steps"] >= self.max_steps:
            state.update(phase="failed", terminal=True, success=False)
        state["history"].append({"action": copy.deepcopy(action),
                                 "observation": f"phase={state['phase']}; prepared={state['prepared']}"})
        return EnvStep(self.observe(), state["terminal"], state["success"])

    def close(self):
        pass
