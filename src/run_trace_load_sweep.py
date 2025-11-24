import argparse
from pathlib import Path
import pandas as pd

from trace_loader import load_trace_csv
from trace_simulator import simulate_trace, PolicyName

def run_sweep(
    trace_path: str,
    output_path: str,
    time_scales: list[float],
    max_requests: int,
    policies: list[PolicyName],
):
    rows = []
    for ts in time_scales:
        print(f"[INFO] time_scale = {ts}")
        requests = load_trace_csv(
            trace_path,
            max_requests=max_requests,
            time_scale=ts,
        )

        for policy in policies:
            print(f"  [INFO] policy = {policy}")
            df = simulate_trace(requests, policy=policy)

            p50_ttft = df["ttft_approx"].quantile(0.5)
            p95_ttft = df["ttft_approx"].quantile(0.95)

            e2e = df["finish_time"] - df["arrival_time"]
            p50_e2e = e2e.quantile(0.5)
            p95_e2e = e2e.quantile(0.95)

            t_start = df["arrival_time"].min()
            t_end = df["finish_time"].max()
            duration = max(t_end - t_start, 1e-9)
            throughput_qps = len(df) / duration

            rows.append(
                dict(
                    trace=Path(trace_path).name,
                    time_scale=ts,
                    policy=policy,
                    p50_ttft=p50_ttft,
                    p95_ttft=p95_ttft,
                    p50_e2e=p50_e2e,
                    p95_e2e=p95_e2e,
                    throughput_qps=throughput_qps,
                    num_requests=len(df),
                )
            )

    out_df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved sweep results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--out", type=str, default="logs/trace/load_sweep.csv")
    parser.add_argument("--max_requests", type=int, default=20000)
    args = parser.parse_args()

    time_scales = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    policies: list[PolicyName] = [
        "fifo",
        "two_stage",
        "two_stage_no_batch",
        "two_stage_no_prefill",
    ]

    run_sweep(args.trace, args.out, time_scales, args.max_requests, policies)
