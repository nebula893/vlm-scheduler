import pandas as pd

def prepare_trace(raw_csv_path, out_csv_path, max_requests=None, time_scale=1.0):
    df = pd.read_csv(raw_csv_path)

    df["parsed_time"] = pd.to_datetime(df["TIMESTAMP"])

    t0 = df["parsed_time"].min()
    df["arrival_time"] = (df["parsed_time"] - t0).dt.total_seconds()

    df["input_tokens"] = df["ContextTokens"]
    df["output_tokens"] = df["GeneratedTokens"]

    if max_requests is not None:
        df = df.head(max_requests)

    df["arrival_time"] = df["arrival_time"] * time_scale

    out_df = df[["arrival_time", "input_tokens", "output_tokens"]]
    out_df.to_csv(out_csv_path, index=False)
    print("Saved processed trace to:", out_csv_path)


if __name__ == "__main__":
    prepare_trace(
        raw_csv_path="data/AzureLMMInferenceTrace_multimodal.csv",
        out_csv_path="data/azure_trace_sample.csv",
        max_requests=5000,    
        time_scale=1.0         
    )
