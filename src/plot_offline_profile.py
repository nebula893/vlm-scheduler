import json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    data_path = Path("data/offline_profile.json")
    assert data_path.exists(), f"{data_path} not found"

    with open(data_path, "r") as f:
        data = json.load(f)

    pre_x, pre_y = [], []
    dec_x, dec_y = [], []

    for item in data:
        if item["type"] == "prefill":
            pre_x.append(item["input_tokens"])
            pre_y.append(item["latency"])
        elif item["type"] == "decode":
            dec_x.append(item["output_tokens"])
            dec_y.append(item["latency"])

    plt.figure()
    plt.plot(pre_x, pre_y, marker="o", label="Prefill latency")
    plt.plot(dec_x, dec_y, marker="s", label="Decode latency")
    plt.xlabel("Tokens")
    plt.ylabel("Latency (seconds)")
    plt.title("Offline profiling: Prefill vs Decode latency")
    plt.grid(True)
    plt.legend()

    Path("plots").mkdir(exist_ok=True)
    out_path = Path("plots/graph1_offline_profile.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
