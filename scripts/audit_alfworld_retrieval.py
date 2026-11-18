"""Read-only audit of a frozen ALFWorld bank against saved target checkpoints."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cave_mem_v3.common.contracts import PublicContext, read_json
from cave_mem_v3.memory_generation.bank import bank_memories
from cave_mem_v3.memory_generation.predicates import evaluate_conditions
from cave_mem_v3.target_data.retrieval import SentenceTransformerEncoder, retrieve_memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--encoder-revision", required=True)
    args = parser.parse_args()

    bank = read_json(args.bank)
    memories = {memory.memory_id: memory for memory in bank_memories(bank)}
    encoder = SentenceTransformerEncoder(args.encoder_path, revision=args.encoder_revision)
    old_ids, new_ids, old_rho, new_rho = Counter(), Counter(), Counter(), Counter()
    family_match = Counter()
    applicable = Counter()
    for path in sorted((args.run / "pairs").glob("*/input.json")):
        if not (path.parent / "closed.json").exists():
            continue
        saved = read_json(path)
        context, action = PublicContext(**saved["context"]), saved["a0"]
        old_id = saved["memory_id"]
        new_id = retrieve_memory(bank, context=context, action=action, encoder=encoder)["selected_memory_id"]
        old_ids[old_id] += 1
        new_ids[new_id] += 1
        old_rho[str(saved["pre_treatment_state"]["x"]["rho"])] += 1
        selected = memories[new_id]
        new_rho[str(evaluate_conditions(selected.positive_conditions,
                                        selected.exclusion_conditions, context))] += 1
        task_type = context.facts.get("task_type")
        family = [condition["equals"] for condition in selected.positive_conditions
                  if condition.get("field") == "task_type" and "equals" in condition]
        family_match["match" if task_type in family else "untagged" if not family else "mismatch"] += 1
        matching = []
        for memory in memories.values():
            condition_families = [condition["equals"] for condition in memory.positive_conditions
                                  if condition.get("field") == "task_type" and "equals" in condition]
            if evaluate_conditions(memory.positive_conditions, memory.exclusion_conditions, context) == 1:
                matching.append(memory.memory_id)
                if task_type in condition_families:
                    applicable["same_family"] += 1
        applicable["checkpoints_with_any"] += bool(matching)
        applicable["total_matching_memories"] += len(matching)
    print(json.dumps({"audited_pairs": sum(new_ids.values()), "old_memory_counts": old_ids,
                      "new_memory_counts": new_ids, "old_rho": old_rho, "new_rho": new_rho,
                      "new_family_match": family_match, "applicable_opportunity": applicable},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
