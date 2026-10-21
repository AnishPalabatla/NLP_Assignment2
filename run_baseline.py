import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from alignment.drgrpo_grader import r1_zero_reward_fn

MODEL = "/work/afp6168/models/Qwen2.5-Math-1.5B"
DATA = "data/MATH/validation.jsonl"
PROMPT = Path("alignment/prompts/r1_zero.prompt").read_text().strip()

rows = []
with open(DATA) as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
rows = rows[:20]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(
    model=MODEL,
    dtype="float16",
    gpu_memory_utilization=0.80,
    max_model_len=2048,
    enforce_eager=True,
)

prompts = [
    PROMPT.replace("{question}", row["problem"])
    for row in rows
]

params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    stop=["</answer>"],
    include_stop_str_in_output=True,
)

outputs = llm.generate(prompts, params)

results = []
correct = 0
for row, prompt, output in zip(rows, prompts, outputs):
    response = output.outputs[0].text
    reward = r1_zero_reward_fn(response, row["solution"])
    if isinstance(reward, dict):
        format_reward = reward.get("format_reward", reward.get("format", 0))
        answer_reward = reward.get("answer_reward", reward.get("correct", 0))
    else:
        format_reward = answer_reward = reward

    correct += int(bool(answer_reward))
    results.append({
        "problem": row["problem"],
        "solution": row["solution"],
        "prompt": prompt,
        "response": response,
        "reward": reward,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    })

Path("results").mkdir(exist_ok=True)
with open("results/baseline_20.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

print(f"Evaluated: {len(results)}")
print(f"Answer accuracy: {correct / len(results):.1%}")
print("Saved: results/baseline_20.jsonl")
print("First response:")
print(results[0]["response"])
