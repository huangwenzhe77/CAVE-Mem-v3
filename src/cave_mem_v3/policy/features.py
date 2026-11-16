"""Frozen F_omega inputs: recorded embeddings plus explicit structured columns."""
import copy
import math

from cave_mem_v3.policy.dataset import _check_feature_fields

SCALARS = (("d", "value"), ("retrieval", "similarity_score"),
           ("E", "mean_difference"), ("E", "n_replay_pairs"), ("E", "acceptance_epsilon"))
TRISTATES = ("v", "rho", "e")


def feature_contract(x):
    if not isinstance(x, dict) or x.get("schema_version") != "pre-treatment-state.v1":
        raise ValueError("unsupported frozen feature version")
    if set(x) != {"schema_version", "context", "a0", "m", "E", "d", "v", "rho", "b", "e", "retrieval", "feature_map"}:
        raise ValueError("unexpected fields in frozen model input")
    _check_feature_fields(x)
    feature = x["feature_map"]
    if (feature.get("status") != "frozen_text_encoder_plus_structured.v1"
            or feature.get("structured_schema_version") != "pre-treatment-structured.v1"
            or not isinstance(feature.get("encoder_identity"), dict) or not feature["encoder_identity"]):
        raise ValueError("missing frozen encoder/structured identity")
    embedding = feature.get("text_embedding")
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("empty frozen embedding")
    names = ["embedding." + str(i) for i in range(len(embedding))]
    for name in TRISTATES:
        names.extend(name + "." + suffix for suffix in ("zero", "one", "unknown"))
    names.extend("e.confidence." + value for value in ("low", "medium", "high", "unknown"))
    for path in SCALARS:
        names.extend((".".join(path), ".".join(path) + ".unknown"))
    for key in ("model_calls", "environment_steps"):
        names.extend(("remaining." + key, "remaining." + key + ".unknown"))
    return {"schema_version": "policy-feature-contract.v1", "embedding_dimension": len(embedding),
            "encoder_identity": copy.deepcopy(feature["encoder_identity"]), "columns": names,
            "text_representation": "already_frozen_public_context_and_a0_embedding",
            "structured_representation": "tri_state_one_hot_and_numeric_missing_indicators",
            "omitted": ["raw_text", "raw_memory_text", "task_identity", "labels", "assignment", "target_costs", "provenance"]}


def feature_vector(x, contract):
    if feature_contract(x) != contract:
        raise ValueError("frozen feature version/encoder/dimension mismatch")
    result = []
    for value in x["feature_map"]["text_embedding"]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("nonfinite embedding")
        result.append(float(value))
    if not any(result):
        raise ValueError("zero embedding")
    for key in TRISTATES:
        value = x[key]["value"] if key == "e" else x[key]
        if value != "?" and (type(value) is not int or value not in (0, 1)):
            raise ValueError("invalid tri-state feature")
        result.extend(float(value == candidate) for candidate in (0, 1, "?"))
    confidence = x["e"].get("confidence", "?")
    if confidence not in ("low", "medium", "high", "?"):
        raise ValueError("invalid exposure confidence category")
    result.extend(float(confidence == value) for value in ("low", "medium", "high", "?"))
    def scalar(value):
        if value is None or value == "?":
            result.extend((0.0, 1.0))
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("invalid numeric feature")
        else:
            result.extend((float(value), 0.0))
    for path in SCALARS:
        scalar(x[path[0]].get(path[1]))
    for key in ("model_calls", "environment_steps"):
        scalar(x["b"].get("remaining", {}).get(key) if x["b"].get("status") == "known" else None)
    return result
