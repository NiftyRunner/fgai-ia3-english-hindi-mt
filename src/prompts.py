"""Prompt formatting and chat template integration for Decoder-Only LLMs (LLaMA & Gemma).

Ensures prompt tokens are masked with label -100 during fine-tuning so cross-entropy loss
is computed strictly on the Hindi translation tokens.
"""

from typing import Any, Dict, List, Optional
import torch


TRANSLATION_INSTRUCTION = (
    "Translate the following English sentence to Hindi. Output only the Hindi translation.\n\nEnglish: {src}"
)


def build_translation_messages(
    src_text: str,
    tgt_text: Optional[str] = None,
    include_system: bool = False,
) -> List[Dict[str, str]]:
    """Construct role-based chat conversation messages.

    Notice: Trailing 'Hindi:' is intentionally omitted to prevent chat models
    from echoing the prefix or confusing generation delimiters.
    """
    user_content = TRANSLATION_INSTRUCTION.format(src=src_text.strip())

    messages = []
    if include_system:
        messages.append({
            "role": "system",
            "content": "You are a professional English to Hindi translator. Output only the Hindi translation without any extra text or notes.",
        })

    messages.append({"role": "user", "content": user_content})

    if tgt_text is not None:
        messages.append({"role": "assistant", "content": tgt_text.strip()})

    return messages


def format_prompt_for_inference(
    src_text: str,
    tokenizer: Any,
) -> str:
    """Format an English source sentence using the model's official chat template for generation."""
    # Gemma chat template works best without separate system role
    messages = build_translation_messages(src_text, tgt_text=None, include_system=False)

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback if chat template is missing
    return f"Translate the following English sentence to Hindi. Output only the Hindi translation.\n\nEnglish: {src_text}\n"


class ResponseOnlyDataCollator:
    """Data collator for SFT training that masks prompt tokens with label -100.

    Why this matters for viva:
    In autoregressive language modeling, standard loss calculates prediction error across
    both the instruction prompt and the target answer. Since the prompt is fixed and given at
    test time, penalizing the model for prompt tokens wastes capacity and dilutes gradients.
    Masking prompt tokens with -100 forces the model to optimize P(Hindi | English) exclusively.
    """

    def __init__(self, tokenizer: Any, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for item in batch:
            src = item["en"]
            tgt = item["hi"]

            # Prompt only
            prompt_msgs = build_translation_messages(src, tgt_text=None, include_system=False)
            prompt_str = self.tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)

            # Full prompt + target answer
            full_msgs = build_translation_messages(src, tgt_text=tgt, include_system=False)
            full_str = self.tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )
            full_ids = self.tokenizer.encode(full_str, add_special_tokens=False)

            # Add EOS token if missing
            if not full_ids or full_ids[-1] != self.tokenizer.eos_token_id:
                full_ids.append(self.tokenizer.eos_token_id)

            # Truncate
            full_ids = full_ids[: self.max_length]
            prompt_len = min(len(prompt_ids), len(full_ids))

            # Mask prompt tokens with -100
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            mask = [1] * len(full_ids)

            input_ids_list.append(full_ids)
            labels_list.append(labels)
            attention_mask_list.append(mask)

        # Batch padding (Right padding is standard for causal LM training)
        max_batch_len = max(len(ids) for ids in input_ids_list)

        padded_input_ids = []
        padded_labels = []
        padded_masks = []

        for ids, labs, msk in zip(input_ids_list, labels_list, attention_mask_list):
            pad_len = max_batch_len - len(ids)
            padded_input_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)
            padded_masks.append(msk + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
        }
