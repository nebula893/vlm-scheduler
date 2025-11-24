import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helper: ensure output dir
# -----------------------------
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Graph 1: Offline profiling
# -----------------------------
def load_offline_profile(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    prefill_x, prefill_y = [], []
    decode_x, decode_y = [], []

    for item in data:
        if item["type"] == "prefill":
            prefill_x.append(item["input_tokens"])
            prefill_y.append(item["latency"])
        elif item["type"] == "decode":
            decode_x.append(item["output_tokens"])
            decode_y.append(item["latency"])

    return (
        np.array(prefill_x),
        np.array(prefill_y),
        np.array(decode_x),
        np.array(decode_y),
    )


def plot_offline_profile(profile_path: str):
    pre_x, pre_y, dec_x, dec_y = load_offline_profile(profile_path)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        pre_x,
        pre_y,
        marker="o",
        linewidth=2,
        label="Prefill latency",
    )
    ax.plot(
        dec_x,
        dec_y,
        marker="s",
        linewidth=2,
        label="Decode latency",
    )

    ax.set_xlabel("Tokens", fontsize=12)
    ax.set_ylabel("Latency (s)", fontsize=12)
    ax.set_title("Offline Profiling: Prefill vs Decode Latency", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = PLOTS_DIR / "graph1_offline_profile.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# -----------------------------
# Trace results helpers
# -----------------------------
def load_trace_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def empirical_cdf(values: np.ndarray):
    """Return x (sorted values) and y (CDF in [0,1])."""
    values = np.sort(values)
    n = len(values)
    if n == 0:
        return values, np.array([])
    y = np.arange(1, n + 1) / n
    return values, y


# -----------------------------
# Graph 2: TTFT CDF
# -----------------------------
def plot_ttft_cdf(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))

    for policy, group in df.groupby("policy"):
        x, y = empirical_cdf(group["ttft_approx"].values)
        ax.plot(x, y, linewidth=2, label=policy)

    ax.set_xlabel("TTFT (s)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("TTFT CDF (FIFO vs Two-Stage)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = PLOTS_DIR / "graph2_ttft_cdf.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# -----------------------------
# Graph 3: End-to-End Latency CDF
# -----------------------------
def plot_e2e_cdf(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))

    for policy, group in df.groupby("policy"):
        e2e = group["finish_time"].values - group["arrival_time"].values
        x, y = empirical_cdf(e2e)
        ax.plot(x, y, linewidth=2, label=policy)

    ax.set_xlabel("End-to-End Latency (s)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("End-to-End Latency CDF", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = PLOTS_DIR / "graph3_e2e_latency_cdf.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# -----------------------------
# Graph 4: Throughput Bar Chart
# -----------------------------
def compute_throughput(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in df.groupby("policy"):
        t_start = group["arrival_time"].min()
        t_end = group["finish_time"].max()
        duration = max(t_end - t_start, 1e-9)
        qps = len(group) / duration
        rows.append({"policy": policy, "throughput_qps": qps})
    return pd.DataFrame(rows)


def plot_throughput_bar(df: pd.DataFrame):
    stats = compute_throughput(df)

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(stats["policy"], stats["throughput_qps"])
    ax.set_xlabel("Policy", fontsize=12)
    ax.set_ylabel("Throughput (QPS)", fontsize=12)
    ax.set_title("Throughput by Policy", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    out_path = PLOTS_DIR / "graph4_throughput_bar.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")

def plot_load_sweep_curve(path="logs/trace/load_sweep.csv"):
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    for policy, g in df.groupby("policy"):
        # 按 throughput 排序，保证是连续曲线
        g = g.sort_values("throughput_qps")
        ax.plot(
            g["throughput_qps"],
            g["p95_ttft"],
            marker="o",
            linewidth=2,
            label=policy,
        )

    ax.set_xlabel("Throughput (QPS)", fontsize=12)
    ax.set_ylabel("P95 TTFT (s)", fontsize=12)
    ax.set_title("Throughput vs P95 TTFT (Azure Trace Load Sweep)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path = PLOTS_DIR / "graph_load_sweep_qps_p95ttft.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    profile_path = "data/offline_profile.json"
    trace_path = "logs/trace/trace_sim_results.csv"

    if os.path.exists(profile_path):
        plot_offline_profile(profile_path)
    else:
        print(f"[Warning] Offline profile file not found: {profile_path}")

    if os.path.exists(trace_path):
        df = load_trace_results(trace_path)
        plot_ttft_cdf(df)
        plot_e2e_cdf(df)
        plot_throughput_bar(df)
    else:
        print(f"[Warning] Trace results file not found: {trace_path}")

    plot_load_sweep_curve()
    

if __name__ == "__main__":
    main()
