# src/baseline_scheduler.py

from typing import List
from request_metadata import RequestMetadata

class BaselineScheduler:

    def schedule(self, requests: List[RequestMetadata]) -> List[RequestMetadata]:
        return sorted(requests, key=lambda r: r.arrival_time)
