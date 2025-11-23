from vllm import LLM, SamplingParams
import time, json, argparse, random, string

def random_text(num_tokens):
    words = []
    for _ in range(num_tokens):
        token = ''.join(random.choices(string.ascii_lowercase, k=8))
        words.append(token)
    return " ".join(words)

def main(model, out_file):
    llm = LLM(model=model, tensor_parallel_size=1)
    results = []

    input_lengths = [16, 32, 64, 128, 256, 512, 1024]

    for L in input_lengths:
        prompt = random_text(L)
        start = time.time()
        llm.generate([prompt], SamplingParams(max_tokens=8))
        end = time.time()
        latency = end - start
        print(f"[prefill] tokens={L}, latency={latency:.3f}")
        results.append({"type": "prefill", "input_tokens": L, "latency": latency})

    output_lengths = [8, 16, 32, 64]
    for O in output_lengths:
        start = time.time()
        llm.generate(["test input"], SamplingParams(max_tokens=O))
        end = time.time()
        latency = end - start
        print(f"[decode] tokens={O}, latency={latency:.3f}")
        results.append({"type": "decode", "output_tokens": O, "latency": latency})

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved to", out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, default="offline_profile.json")
    args = parser.parse_args()
    main(args.model, args.out)
