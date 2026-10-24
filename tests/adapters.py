from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


class _PackedSFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, Tensor]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.examples[index]


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
    from cs336_alignment.sft_utils import tokenize_prompt_and_output

    return tokenize_prompt_and_output(
        prompt_strs=prompt_strs,
        output_strs=output_strs,
        tokenizer=tokenizer,
    )


def run_compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
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
    from cs336_alignment.grpo import compute_group_normalized_rewards

    return compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
    )


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    from cs336_alignment.sft_utils import compute_entropy

    return compute_entropy(logits)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
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
    from cs336_alignment.sft_utils import get_response_log_probs

    return get_response_log_probs(
        model=model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=return_token_entropy,
    )


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
    from cs336_alignment.grpo import compute_naive_policy_gradient_loss

    return compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages=raw_rewards_or_advantages,
        policy_log_probs=policy_log_probs,
    )


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
    from cs336_alignment.grpo import compute_grpo_clip_loss

    return compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    from cs336_alignment.grpo import compute_policy_gradient_loss

    return compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )


def run_masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
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
    from cs336_alignment.grpo import masked_mean

    return masked_mean(tensor=tensor, mask=mask, dim=dim)


def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    from cs336_alignment.sft_utils import sft_microbatch_train_step

    return sft_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        normalize_constant=normalize_constant,
    )


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
    from cs336_alignment.grpo import grpo_microbatch_train_step

    return grpo_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )


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
    from cs336_alignment.sft_utils import masked_normalize

    return masked_normalize(
        tensor=tensor,
        mask=mask,
        normalize_constant=normalize_constant,
        dim=dim,
    )


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an eos_token_id")

    with open(dataset_path, encoding="utf-8") as dataset_file:
        documents = [json.loads(line) for line in dataset_file if line.strip()]
    if shuffle:
        random.shuffle(documents)

    prompt_template_path = (
        Path(__file__).resolve().parents[1]
        / "alignment"
        / "prompts"
        / "alpaca_sft.prompt"
    )
    with prompt_template_path.open(encoding="utf-8") as template_file:
        prompt_template = template_file.read()

    token_stream: list[int] = []
    for document in documents:
        text = prompt_template.format(
            instruction=document["prompt"],
            response=document["response"],
        ).strip()
        token_stream.extend(tokenizer(text, add_special_tokens=True)["input_ids"])
        token_stream.append(tokenizer.eos_token_id)

    examples: list[dict[str, Tensor]] = []
    for start in range(0, len(token_stream) - seq_length, seq_length):
        token_window = token_stream[start : start + seq_length + 1]
        if len(token_window) != seq_length + 1:
            break
        examples.append(
            {
                "input_ids": torch.tensor(token_window[:-1], dtype=torch.long),
                "labels": torch.tensor(token_window[1:], dtype=torch.long),
            }
        )
    return _PackedSFTDataset(examples)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    options = mmlu_example.get("options")
    if not isinstance(options, list) or not options or len(options) > 4:
        return None

    choices = "ABCD"[: len(options)]
    choice = rf"[{choices}]"
    explicit_patterns = (
        rf"\banswer\s*(?:is|:)\s*\(?({choice})\)?\b",
        rf"\boption\s*\(?({choice})\)?\b",
        rf"\\boxed\s*\{{\s*({choice})\s*\}}",
    )
    for pattern in explicit_patterns:
        matches = re.findall(pattern, model_output, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    matches = re.findall(
        rf"(?<![A-Za-z0-9])({choice})(?![A-Za-z0-9])",
        model_output,
        flags=re.IGNORECASE,
    )
    return matches[-1].upper() if matches else None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    numbers = re.findall(
        r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?",
        model_output,
    )
    return numbers[-1].replace(",", "") if numbers else None


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    if tokenizer.eos_token is None:
        raise ValueError("tokenizer must define an eos_token")

    def response_log_prob_sum(
        model: torch.nn.Module, response: str
    ) -> Tensor:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        encoded = tokenizer(
            prompt + response + tokenizer.eos_token,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_start = len(prompt_ids) - 1
        if response_start < 0:
            response_start = 0

        logits = model(input_ids=input_ids).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        token_log_probs = torch.log_softmax(logits, dim=-1).gather(
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)
        return token_log_probs[:, response_start:].sum()

    policy_was_training = lm.training
    reference_was_training = lm_ref.training
    lm.eval()
    lm_ref.eval()
    try:
        policy_chosen = response_log_prob_sum(lm, response_chosen)
        policy_rejected = response_log_prob_sum(lm, response_rejected)
        with torch.no_grad():
            reference_chosen = response_log_prob_sum(lm_ref, response_chosen)
            reference_rejected = response_log_prob_sum(lm_ref, response_rejected)
    finally:
        if policy_was_training:
            lm.train()
        if reference_was_training:
            lm_ref.train()

    preference_logit = beta * (
        policy_chosen
        - policy_rejected
        - reference_chosen
        + reference_rejected
    )
    return -torch.nn.functional.logsigmoid(preference_logit)
