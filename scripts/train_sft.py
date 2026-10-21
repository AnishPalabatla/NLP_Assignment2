
import json
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Adafactor

from alignment.sft_utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)

MODEL_PATH = "/work/afp6168/models/Qwen2.5-Math-1.5B"
DATA_PATH = "data/sft.jsonl"
OUTPUT_PATH = "checkpoints/sft-qwen-math"

BATCH_SIZE = 1
MAX_STEPS = 20
MAX_PROMPT_TOKENS = 256
MAX_RESPONSE_TOKENS = 384
LEARNING_RATE = 1e-5


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    device = "cuda"
    dtype = torch.bfloat16

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
    ).to(device)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = Adafactor(
        model.parameters(),
        lr=LEARNING_RATE,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    print("Loading SFT dataset...")
    records = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} examples")

    # Start with a small subset for the first training run.
    records = records[:MAX_STEPS]

    dataloader = DataLoader(
        records,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    step = 0

    for batch in dataloader:
        prompts = list(batch["prompt"])
        responses = list(batch["response"])

        # Bound sequence length to reduce GPU memory use.
        prompts = [
            tokenizer.decode(
                tokenizer.encode(p, add_special_tokens=False)
                [:MAX_PROMPT_TOKENS],
                skip_special_tokens=False,
            )
            for p in prompts
        ]
        responses = [
            tokenizer.decode(
                tokenizer.encode(r, add_special_tokens=False)
                [:MAX_RESPONSE_TOKENS],
                skip_special_tokens=False,
            )
            for r in responses
        ]

        tokens = tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=responses,
            tokenizer=tokenizer,
        )

        input_ids = tokens["input_ids"].to(device)
        labels = tokens["labels"].to(device)
        response_mask = tokens["response_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = get_response_log_probs(
            model=model,
            input_ids=input_ids,
            labels=labels,
        )

        avg_response_tokens = max(
            1.0, response_mask.sum(dim=1).float().mean().item()
        )

        loss, _ = sft_microbatch_train_step(
            policy_log_probs=outputs["log_probs"],
            response_mask=response_mask,
            gradient_accumulation_steps=1,
            normalize_constant=avg_response_tokens,
        )

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        print(f"Step {step}/{MAX_STEPS} | loss: {loss.item():.4f}")

        if step >= MAX_STEPS:
            break

    print("Saving trained model...")
    model.config.use_cache = True
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    print(f"Training complete. Model saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

