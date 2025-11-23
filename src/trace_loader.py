# src/trace_loader.py

import pandas as pd
from typing import List
from request_metadata import RequestMetadata

def load_trace_csv(
    path: str,
    max_requests: int | None = None,
    arrival_col: str = "arrival_time",
    input_col: str = "input_tokens",
    output_col: str = "output_tokens",
    time_scale: float = 1.0,
) -> List[RequestMetadata]:
    """
    read request from CSV of Azure LMM trace and turn into RequestMetadata lists。
    time_scale can be used to adjust load-bearing strength
    """
    df = pd.read_csv(path)

    if max_requests is not None:
        df = df.head(max_requests)

    t0 = df[arrival_col].min()
    df["arrival_norm"] = (df[arrival_col] - t0) * time_scale

    requests: List[RequestMetadata] = []
    for i, row in df.iterrows():
        r = RequestMetadata(
            request_id=int(i),
            arrival_time=float(row["arrival_norm"]),
            input_tokens=int(row[input_col]),
            output_tokens=int(row[output_col]),
        )
        requests.append(r)

    return requests
