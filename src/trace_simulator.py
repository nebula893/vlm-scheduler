# src/trace_simulator.py

from typing import List, Literal, Dict, Any
import pandas as pd

from request_metadata import RequestMetadata
from cost_model import (
    estimate_total_cost,
    estimate_prefill_cost,
    estimate_decode_cost,
)

PolicyName = Literal[
    "fifo",
    "two_stage",
    "two_stage_no_batch",
    "two_stage_no_prefill",
]


# ---------------------------------------------------------------------
# FIFO BASELINE
# ---------------------------------------------------------------------
def simulate_fifo(requests: List[RequestMetadata]) -> pd.DataFrame:
    """
    baseline：单阶段 FIFO。
    service_time = prefill + decode
    """
    reqs = sorted(requests, key=lambda r: r.arrival_time)
    records = []
    current_time = 0.0

    for r in reqs:
        service_time = estimate_total_cost(r.input_tokens, r.output_tokens)
        start = max(r.arrival_time, current_time)
        finish = start + service_time

        prefill_time = estimate_prefill_cost(r.input_tokens)
        ttft = start + prefill_time - r.arrival_time

        records.append(
            dict(
                request_id=r.request_id,
                arrival_time=r.arrival_time,
                start_time=start,
                finish_time=finish,
                waiting_time=start - r.arrival_time,
                service_time=service_time,
                ttft_approx=ttft,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                policy="fifo",
            )
        )

        current_time = finish

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# TWO-STAGE BASELINE (你的版本)
# ---------------------------------------------------------------------
def simulate_two_stage(requests: List[RequestMetadata]) -> pd.DataFrame:
    """
    你的 Two-Stage 调度：
      - Prefill: SJF（prefill cost 最小优先）
      - Decode: 
          short decode ≤ threshold → 单独执行
          long decode > threshold  → 一次 batch 所有 ready 的 long
    """
    return _simulate_two_stage_core(requests,
                                    use_prefill_sjf=True,
                                    use_decode_batch=True,
                                    policy_name="two_stage")


# ---------------------------------------------------------------------
# TWO-STAGE (NO BATCH) —— ABLATION 1
# ---------------------------------------------------------------------
def simulate_two_stage_no_batch(requests: List[RequestMetadata]) -> pd.DataFrame:
    """
    Ablation 1:
      - Prefill: SJF（和 two_stage 相同）
      - Decode: 不做 batching，每个请求 decode 独立执行
    """
    return _simulate_two_stage_core(requests,
                                    use_prefill_sjf=True,
                                    use_decode_batch=False,
                                    policy_name="two_stage_no_batch")


# ---------------------------------------------------------------------
# TWO-STAGE (NO PREFILL-SJF) —— ABLATION 2
# ---------------------------------------------------------------------
def simulate_two_stage_no_prefill(requests: List[RequestMetadata]) -> pd.DataFrame:
    """
    Ablation 2:
      - Prefill: FIFO（不使用 SJF）
      - Decode: 和 two_stage 相同（短独立 + 长 batch）
    """
    return _simulate_two_stage_core(requests,
                                    use_prefill_sjf=False,
                                    use_decode_batch=True,
                                    policy_name="two_stage_no_prefill")


# ---------------------------------------------------------------------
# CORE IMPLEMENTATION FOR ALL TWO-STAGE VARIANTS
# ---------------------------------------------------------------------
def _simulate_two_stage_core(
    requests: List[RequestMetadata],
    use_prefill_sjf: bool,
    use_decode_batch: bool,
    policy_name: str,
) -> pd.DataFrame:

    T_SMALL = 32  # 可调 decode 阈值

    # 排序（统一按 arrival 排）
    reqs = list(sorted(requests, key=lambda r: r.arrival_time))
    n = len(reqs)

    # 预计算成本
    prefill_cost = [estimate_prefill_cost(r.input_tokens) for r in reqs]
    decode_cost = [estimate_decode_cost(r.output_tokens) for r in reqs]

    # 状态
    prefill_start = [None] * n
    prefill_done = [None] * n
    decode_done = [None] * n

    # index sets
    idx_prefill_remain = set(range(n))
    idx_decode_pending = set()
    idx_finished = set()

    current = 0.0
    records = []

    while len(idx_finished) < n:

        # 哪些 prefill ready（到达时间已到）
        prefill_ready = [
            i for i in idx_prefill_remain
            if reqs[i].arrival_time <= current
        ]

        # 哪些 decode ready
        decode_ready = [
            i for i in idx_decode_pending
            if prefill_done[i] is not None
            and prefill_done[i] <= current
        ]

        # -----------------------------------------------------
        # 1) PREFILL 阶段
        # -----------------------------------------------------
        if prefill_ready:
            if use_prefill_sjf:
                # 用 prefill_cost SJF
                chosen = min(prefill_ready, key=lambda i: prefill_cost[i])
            else:
                # FIFO：按 arrival_time
                chosen = min(prefill_ready, key=lambda i: reqs[i].arrival_time)

            start = max(current, reqs[chosen].arrival_time)
            finish = start + prefill_cost[chosen]

            prefill_start[chosen] = start
            prefill_done[chosen] = finish

            idx_prefill_remain.remove(chosen)
            idx_decode_pending.add(chosen)

            current = finish
            continue

        # -----------------------------------------------------
        # 2) DECODE 阶段
        # -----------------------------------------------------
        if decode_ready:
            short_ready = [
                i for i in decode_ready
                if reqs[i].output_tokens <= T_SMALL
            ]
            long_ready = [
                i for i in decode_ready
                if reqs[i].output_tokens > T_SMALL
            ]

            # 短 decode: always independent
            if short_ready:
                chosen = min(short_ready, key=lambda i: decode_cost[i])
                start = current
                finish = start + decode_cost[chosen]

                decode_done[chosen] = finish
                idx_decode_pending.remove(chosen)
                idx_finished.add(chosen)

                current = finish
                continue

            # 长 decode: batch or not?
            if long_ready:
                if use_decode_batch:
                    # batch 所有 ready 的长请求
                    batch = list(long_ready)
                    start = current
                    btime = max(decode_cost[i] for i in batch)
                    finish = start + btime

                    for i in batch:
                        decode_done[i] = finish
                        idx_decode_pending.remove(i)
                        idx_finished.add(i)

                    current = finish
                    continue
                else:
                    # no batch：挑 decode_cost 最小的
                    chosen = min(long_ready, key=lambda i: decode_cost[i])
                    start = current
                    finish = start + decode_cost[chosen]

                    decode_done[chosen] = finish
                    idx_decode_pending.remove(chosen)
                    idx_finished.add(chosen)

                    current = finish
                    continue

        # -----------------------------------------------------
        # 3) 当前无 ready，跳到下一 arrival
        # -----------------------------------------------------
        future_arrivals = [
            reqs[i].arrival_time
            for i in idx_prefill_remain
        ]
        if not future_arrivals:
            break
        current = min(future_arrivals)

    # ---------------------------------------------------------
    # 汇总记录
    # ---------------------------------------------------------
    for i, r in enumerate(reqs):
        if prefill_start[i] is None or prefill_done[i] is None or decode_done[i] is None:
            continue

        service_time = (prefill_done[i] - prefill_start[i]) + decode_cost[i]
        ttft = prefill_done[i] - r.arrival_time

        records.append(
            dict(
                request_id=r.request_id,
                arrival_time=r.arrival_time,
                start_time=prefill_start[i],
                finish_time=decode_done[i],
                waiting_time=prefill_start[i] - r.arrival_time,
                service_time=service_time,
                ttft_approx=ttft,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                policy=policy_name,
            )
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# PUBLIC INTERFACE
# ---------------------------------------------------------------------
def simulate_trace(
    requests: List[RequestMetadata],
    policy: PolicyName,
) -> pd.DataFrame:
    """
    对外接口：根据 policy 调用不同仿真器。
    """
    if policy == "fifo":
        return simulate_fifo(requests)
    elif policy == "two_stage":
        return simulate_two_stage(requests)
    elif policy == "two_stage_no_batch":
        return simulate_two_stage_no_batch(requests)
    elif policy == "two_stage_no_prefill":
        return simulate_two_stage_no_prefill(requests)
    else:
        raise ValueError(f"Unknown policy: {policy}")
