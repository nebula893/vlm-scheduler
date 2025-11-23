from vllm import LLM, SamplingParams
import time

def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    print(f"Loading model: {model_name}")
    llm = LLM(model=model_name, tensor_parallel_size=1)

    prompt = "You are running on a Georgia Tech PACE-ICE GPU node. Say hello in one short sentence."
    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=32,
    )

    start = time.time()
    outputs = llm.generate([prompt], sampling_params)
    end = time.time()

    print("Output:", outputs[0].outputs[0].text.strip())
    print("Latency (s):", end - start)

if __name__ == "__main__":
    main()

