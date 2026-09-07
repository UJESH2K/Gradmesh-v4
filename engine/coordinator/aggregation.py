"""Contribution-weighted FedAvg.

federated_training.average_state_dicts takes an unweighted mean, which is only
correct when every shard is the same size. v4 hands workers unequal shards on
purpose, so the aggregation step has to weight each state dict by the number of
samples behind it. That is the original FedAvg estimator:

    w_global = sum_i (n_i / n_total) * w_i

torch is imported lazily so the coordinator can boot and serve the dashboard
before the training-plane wheels have finished installing in the background.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from federated_training import decode_state_dict, encode_state_dict


def torch_ready() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def weighted_average_state_dicts(
    encoded_state_dicts: Sequence[str],
    weights: Sequence[float],
) -> Dict[str, Any]:
    """Weighted mean of serialised state dicts. Weights are normalised here."""
    import torch

    if not encoded_state_dicts:
        raise ValueError("At least one state dict is required")
    if len(encoded_state_dicts) != len(weights):
        raise ValueError("Each state dict needs exactly one weight")

    total = float(sum(weights))
    if total <= 0:
        normalised = [1.0 / len(weights)] * len(weights)
    else:
        normalised = [float(weight) / total for weight in weights]

    state_dicts: List[Dict[str, Any]] = [decode_state_dict(item) for item in encoded_state_dicts]
    reference = state_dicts[0]
    averaged: Dict[str, Any] = {}

    for key, reference_value in reference.items():
        if not hasattr(reference_value, "detach"):
            averaged[key] = reference_value
            continue

        accumulator = None
        weight_sum = 0.0
        for state_dict, weight in zip(state_dicts, normalised):
            value = state_dict.get(key)
            if value is None or not hasattr(value, "detach"):
                continue
            if tuple(value.shape) != tuple(reference_value.shape):
                # A head reshaped by a different class count. Skip rather than
                # crash, exactly as the v3 worker does when loading weights.
                continue
            contribution = value.detach().float().cpu() * weight
            accumulator = contribution if accumulator is None else accumulator + contribution
            weight_sum += weight

        if accumulator is None or weight_sum <= 0:
            averaged[key] = reference_value
            continue

        # Renormalise in case some workers were missing this tensor.
        averaged[key] = (accumulator / weight_sum).to(reference_value.dtype)

    return averaged


def aggregate(results: Sequence[dict], weights: Dict[str, float]) -> str:
    """Aggregate finished round results into a new encoded global state dict."""
    ordered = [result for result in results if result.get("weights_b64")]
    if not ordered:
        raise ValueError("No completed shards carried weights")
    encoded = [result["weights_b64"] for result in ordered]
    weight_vector = [weights.get(result["batch_id"], 0.0) for result in ordered]
    return encode_state_dict(weighted_average_state_dicts(encoded, weight_vector))
