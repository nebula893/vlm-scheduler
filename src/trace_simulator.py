# src/trace_simulator.py

from typing import List, Literal, Dict, Any
import pandas as pd

from request_metadata import RequestMetadata
from cost_model import estimate_total_cost, estimate_prefill_cost, estimate_decode_cost

PolicyName = Literal["fifo", "two_stage"]

def simulate_trace(
    requests: List[RequestMetadata],
    policy: PolicyName,
) -> pd.DataFrame:
    """
    return start_time, finish_time, waiting_time, ttft_approx of every requests。
    """
    requests_sorted = sorted(requests, key=lambda r: r.arrival_time)
    n = len(requests_sorted)

    current_time = 0.0
    idx_unfinished = set(range(n))  
    records: List[Dict[str, Any]] = []

    while idx_unfinished:
        ready_indices = [
            i for i in idx_unfinished
            if requests_sorted[i].arrival_time <= current_time
        ]

        if not ready_indices:
            next_i = min(idx_unfinished, key=lambda i: requests_sorted[i].arrival_time)
            current_time = requests_sorted[next_i].arrival_time
            ready_indices = [next_i]

        if policy == "fifo":
            chosen_i = min(ready_indices, key=lambda i: requests_sorted[i].arrival_time)
        elif policy == "two_stage":
            def score(i: int) -> float:
                r = requests_sorted[i]
                return estimate_total_cost(r.input_tokens, r.output_tokens)
            chosen_i = min(ready_indices, key=score)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        req = requests_sorted[chosen_i]

        service_time = estimate_total_cost(req.input_tokens, req.output_tokens)
        start_time = max(current_time, req.arrival_time)
        finish_time = start_time + service_time

        prefill_time = estimate_prefill_cost(req.input_tokens)
        ttft_approx = start_time + prefill_time - req.arrival_time

        record = {
            "request_id": req.request_id,
            "arrival_time": req.arrival_time,
            "start_time": start_time,
            "finish_time": finish_time,
            "waiting_time": start_time - req.arrival_time,
            "service_time": service_time,
            "ttft_approx": ttft_approx,
            "input_tokens": req.input_tokens,
            "output_tokens": req.output_tokens,
            "policy": policy,
        }
        records.append(record)

        current_time = finish_time
        idx_unfinished.remove(chosen_i)

    df = pd.DataFrame(records)
    return df
