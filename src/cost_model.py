# src/cost_model.py

from dataclasses import dataclass

@dataclass
class StageCostParams:
    prefill_base: float  
    prefill_alpha: float 
    decode_base: float  
    decode_beta: float   

DEFAULT_COST_PARAMS = StageCostParams(
    prefill_base=0.05,     
    prefill_alpha=1.0e-4,  
    decode_base=0.0,       
    decode_beta=7.2e-3,   
)

def estimate_prefill_cost(input_tokens: int,
                          params: StageCostParams = DEFAULT_COST_PARAMS) -> float:

    return params.prefill_base + params.prefill_alpha * max(input_tokens, 0)

def estimate_decode_cost(output_tokens: int,
                         params: StageCostParams = DEFAULT_COST_PARAMS) -> float:

    return params.decode_base + params.decode_beta * max(output_tokens, 0)

def estimate_total_cost(input_tokens: int, output_tokens: int,
                        params: StageCostParams = DEFAULT_COST_PARAMS) -> float:

    return estimate_prefill_cost(input_tokens, params) + \
           estimate_decode_cost(output_tokens, params)

if __name__ == "__main__":
    for L in [16, 64, 128, 256, 512, 1024]:
        print("prefill L=", L, " -> ", estimate_prefill_cost(L))
    for O in [8, 16, 32, 64]:
        print("decode O=", O, " -> ", estimate_decode_cost(O))
