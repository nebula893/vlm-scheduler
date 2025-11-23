# src/request_metadata.py
from dataclasses import dataclass

@dataclass
class RequestMetadata:
    request_id: int
    arrival_time: float        
    input_tokens: int          
    output_tokens: int  
