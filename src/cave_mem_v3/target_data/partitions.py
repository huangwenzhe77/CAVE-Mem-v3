"""Explicit target partitions; test outcomes are unavailable in collection/training."""
from pathlib import Path

from cave_mem_v3.benchmarks.tasks import validate_manifest
from cave_mem_v3.common.contracts import read_json


def target_partition(role, validation_role=None):
    if role not in ("target_train", "validation"):
        raise ValueError("target partition must be target_train or validation; test outcomes are forbidden")
    if role == "validation":
        if validation_role not in ("calibration", "selection"):
            raise ValueError("validation requires explicit validation_role calibration or selection")
        return "validation/" + validation_role
    if validation_role is not None:
        raise ValueError("target_train cannot declare a validation_role")
    return role


def partition_fields(role, validation_role=None):
    target_partition(role, validation_role)
    return {"validation_role": validation_role} if role == "validation" else {}


def partition_tasks(manifest, role, validation_role=None):
    target_partition(role, validation_role)
    validate_manifest(manifest)
    tasks = [task for task in manifest["tasks"] if task["role"] == role
             and task.get("validation_role") == validation_role]
    if not tasks:
        raise ValueError("manifest has no tasks in requested target partition")
    return tasks


def check_partition_resume(output_dir, role, validation_role=None):
    """Reject changed partition before encoder/model/environment construction."""
    target_partition(role, validation_role)
    path = Path(output_dir) / "run.json"
    if path.exists():
        stored = read_json(path)
        if (stored.get("role"), stored.get("validation_role")) != (role, validation_role):
            raise ValueError("target run partition changed; choose a new output directory")


def validation_binding(run):
    """Add explicit validation metadata without changing legacy training schemas."""
    target_partition(run["role"], run.get("validation_role"))
    if run["role"] != "validation":
        return {}
    return {"role": "validation", "validation_role": run["validation_role"],
            "manifest_hash": run["manifest_hash"], "task_order": run["task_order"]}


def validate_validation_bank(manifest, bank):
    """Validate validation preparation sources before invoking target providers."""
    from cave_mem_v3.interventions.inputs import checked, _validate_source_partition
    from cave_mem_v3.memory_generation.bank import bank_memories
    from cave_mem_v3.memory_generation.replay import require_current_evidence
    checked(bank, "bank_hash", "verified-source-bank.v3")
    if bank.get("frozen") is not True:
        raise ValueError("validation requires a frozen history bank")
    _validate_source_partition({"manifest": manifest, "manifest_hash": manifest["manifest_hash"]}, bank)
    if bank.get("memories"):
        for memory in bank_memories(bank):
            require_current_evidence(memory.evidence, replay_pairs=2)
    elif bank.get("memories") != [] or bank.get("memory_count") != 0 or bank.get("audit_sources") != {}:
        raise ValueError("invalid empty bank")
