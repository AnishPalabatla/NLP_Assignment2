import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from alignment.drgrpo_grader import r1_zero_reward_fn

BASE = "/work/afp6168/models/Qwen2.5-Math-1.5B"
ADAPTER = "checkpoints/sft-qwen-math"
DATA = "data/MATH/validation.jsonl"
PROMPT = Path("alignment/prompts/r1_zero.prompt").read_text().strip()

rows = []
with open(DATA) as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
rows = rows[:20]

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

results = []
for i, row in enumerate(rows, 1):
    prompt = PROMPT.replace("{question}", row["problem"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    reward = r1_zero_reward_fn(response, row["solution"])

    results.append({
        "problem": row["problem"],
        "solution": row["solution"],
        "response": response,
        "format_reward": reward["format_reward"],
        "answer_reward": reward["answer_reward"],
        "reward": reward["reward"],
    })
    print(f"{i}/{len(rows)} | format={reward['format_reward']} "
          f"| correct={reward['answer_reward']}")

Path("results").mkdir(exist_ok=True)
with open("results/sft_20.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

n = len(results)
formatted = sum(r["format_reward"] for r in results)
correct = sum(r["answer_reward"] for r in results)
print("\nSFT RESULTS")
print(f"Evaluated: {n}")
print(f"Format compliance: {formatted}/{n} ({formatted/n:.1%})")
print(f"Answer accuracy: {correct}/{n} ({correct/n:.1%})")
print(f"Mean reward: {sum(r['reward'] for r in results)/n:.3f}")
print("Saved: results/sft_20.jsonl")
