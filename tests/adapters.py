from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    from alignment.sft_utils import tokenize_prompt_and_output
    return tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    from alignment.grpo import compute_group_normalized_rewards
    return compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, normalize_by_std)


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    from alignment.sft_utils import compute_entropy
    return compute_entropy(logits)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    from alignment.sft_utils import get_response_log_probs
    return get_response_log_probs(model, input_ids, labels, return_token_entropy)


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    from alignment.grpo import compute_naive_policy_gradient_loss
    return compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs)


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    from alignment.grpo import compute_grpo_clip_loss
    return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    from alignment.grpo import compute_policy_gradient_loss
    return compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)


def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    from alignment.grpo import masked_mean
    return masked_mean(tensor, mask, dim)

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    from alignment.sft_utils import sft_microbatch_train_step
    return sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant)

    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    from alignment.grpo import grpo_microbatch_train_step
    return grpo_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, loss_type, raw_rewards, advantages, old_log_probs, cliprange)


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    from alignment.sft_utils import masked_normalize
    return masked_normalize(tensor, mask, normalize_constant=normalize_constant, dim=dim)


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""

######################temp##########################


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    class PackedSFTDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, index):
            return self.examples[index]

    if seq_length <= 0:
        raise ValueError("seq_length must be positive")

    with open(dataset_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if shuffle:
        random.shuffle(records)

    # Concatenate raw text before tokenization.
    text = "".join(
        record["prompt"] + record["response"]
        for record in records
    )

    token_ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    examples = []

    # Each input has seq_length tokens; labels are shifted by one.
    for start in range(0, len(token_ids) - seq_length, seq_length):
        block = token_ids[start : start + seq_length + 1]

        examples.append({
            "input_ids": torch.tensor(block[:-1], dtype=torch.long),
            "labels": torch.tensor(block[1:], dtype=torch.long),
        })

    return PackedSFTDataset(examples)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    from torch.utils.data import DataLoader

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    import re

    if not model_output:
        return None

    # Prefer an explicit answer label, e.g. "The answer is B."
    patterns = [
        r"(?i)\b(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)\s*\(?([ABCD])\)?\b",
        r"(?i)\boption\s*\(?([ABCD])\)?\b",
        r"(?i)\bfinal\s+answer\s*(?:is|:)\s*\(?([ABCD])\)?\b",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, model_output)
        if matches:
            return matches[-1].upper()

    # Accept a standalone final letter only when the output is short
    # enough to plausibly be an answer rather than a long explanation.
    cleaned = model_output.strip()
    match = re.fullmatch(
        r"(?is)(?:\(?([ABCD])\)?[.)]?)",
        cleaned,
    )
    if match:
        return match.group(1).upper()

    return None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    import re

    if not model_output:
        return None

    # Prefer the final answer after the standard GSM8K delimiter.
    answer_marker = re.findall(
        r"(?im)####\s*(-?\d[\d,]*(?:\.\d+)?)",
        model_output,
    )
    if answer_marker:
        return answer_marker[-1].replace(",", "")

    # Otherwise, use the last numeric value in the output.
    numbers = re.findall(
        r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?",
        model_output,
    )
    if not numbers:
        return None

    return numbers[-1].replace(",", "")


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    import torch.nn.functional as F

    if beta <= 0:
        raise ValueError("beta must be positive")

    def get_response_log_prob(
        model: torch.nn.Module,
        prompt: str,
        response: str,
    ) -> torch.Tensor:
        device = next(model.parameters()).device

        # Tokenize the prompt and the complete sequence without adding
        # special tokens, so both use the same tokenization convention.
        prompt_ids = tokenizer(
            prompt, add_special_tokens=False
        )["input_ids"]

        full_ids = tokenizer(
            prompt + response, add_special_tokens=False
        )["input_ids"]

        # Find the common token prefix. Tokenization at the prompt/response
        # boundary can change the final prompt token.
        common_prefix_len = 0
        for prompt_token, full_token in zip(prompt_ids, full_ids):
            if prompt_token != full_token:
                break
            common_prefix_len += 1

        if common_prefix_len == 0:
            raise ValueError("Could not identify the prompt token prefix")

        if len(full_ids) <= common_prefix_len:
            raise ValueError("Response must contain at least one token")

        input_ids = torch.tensor(
            [full_ids[:-1]], dtype=torch.long, device=device
        )
        labels = torch.tensor(
            [full_ids[1:]], dtype=torch.long, device=device
        )

        logits = model(input_ids=input_ids).logits
        log_probs = F.log_softmax(logits, dim=-1)

        token_log_probs = log_probs.gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1)

        # Logit position j predicts token position j + 1.
        response_start = common_prefix_len - 1
        return token_log_probs[:, response_start:].sum()

    policy_chosen = get_response_log_prob(lm, prompt, response_chosen)
    policy_rejected = get_response_log_prob(lm, prompt, response_rejected)

    with torch.no_grad():
        reference_chosen = get_response_log_prob(
            lm_ref, prompt, response_chosen
        )
        reference_rejected = get_response_log_prob(
            lm_ref, prompt, response_rejected
        )

    policy_preference = policy_chosen - policy_rejected
    reference_preference = reference_chosen - reference_rejected

    return -F.logsigmoid(
        beta * (policy_preference - reference_preference)
    )