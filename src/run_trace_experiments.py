# src/run_trace_experiments.py

from trace_loader import load_trace_csv
from trace_simulator import simulate_trace
import pandas as pd
from pathlib import Path

def summarize(df: pd.DataFrame, policy: str):
    sub = df[df["policy"] == policy]
    p50_ttft = sub["ttft_approx"].quantile(0.5)
    p95_ttft = sub["ttft_approx"].quantile(0.95)
    p50_e2e = (sub["finish_time"] - sub["arrival_time"]).quantile(0.5)
    p95_e2e = (sub["finish_time"] - sub["arrival_time"]).quantile(0.95)
    avg_qps = len(sub) / (sub["finish_time"].max() - sub["arrival_time"].min())
    return {
        "policy": policy,
        "p50_ttft": p50_ttft,
        "p95_ttft": p95_ttft,
        "p50_e2e": p50_e2e,
        "p95_e2e": p95_e2e,
        "throughput_qps": avg_qps,
    }

if __name__ == "__main__":
    trace_path = "data/azure_trace_sample.csv"
    requests = load_trace_csv(trace_path, max_requests=2000, time_scale=1.0)

    df_fifo = simulate_trace(requests, policy="fifo")
    df_two = simulate_trace(requests, policy="two_stage")

    Path("logs/trace").mkdir(parents=True, exist_ok=True)
    df_all = pd.concat([df_fifo, df_two], ignore_index=True)
    df_all.to_csv("logs/trace/trace_sim_results.csv", index=False)

    stats = []
    stats.append(summarize(df_all, "fifo"))
    stats.append(summarize(df_all, "two_stage"))
    print(pd.DataFrame(stats))
