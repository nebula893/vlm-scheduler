# src/test_scheduler.py

from request_metadata import RequestMetadata
from baseline_scheduler import BaselineScheduler
from two_stage_scheduler import TwoStageScheduler

if __name__ == "__main__":
    # construct 3 simulation requests
    reqs = [
        RequestMetadata(1, 0.1, 128, 16),
        RequestMetadata(2, 0.2, 512, 16),
        RequestMetadata(3, 0.3, 32, 16),
    ]

    print("=== Baseline ===")
    b = BaselineScheduler()
    for r in b.schedule(reqs):
        print(r)

    print("\n=== Two-Stage Scheduler ===")
    t = TwoStageScheduler(prefill_weight=1.0)
    for r in t.schedule(reqs):
        print(r)
