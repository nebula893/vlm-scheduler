# src/two_stage_scheduler.py

from typing import List
from request_metadata import RequestMetadata
from cost_model import estimate_prefill_cost, estimate_decode_cost

class TwoStageScheduler:

    def __init__(self, prefill_weight=1.0, decode_weight=1.0, batch_window=0.05):
        self.prefill_weight = prefill_weight
        self.decode_weight = decode_weight
        self.batch_window = batch_window

    def schedule(self, requests: List[RequestMetadata]) -> List[RequestMetadata]:

        scored = []
        for r in requests:
            prefill_c = estimate_prefill_cost(r.input_tokens)
            decode_c  = estimate_decode_cost(r.output_tokens)
            score = self.prefill_weight * prefill_c + self.decode_weight * decode_c
            scored.append((score, r))

        ordered = [r for (s, r) in sorted(scored, key=lambda x: x[0])]

        return ordered
