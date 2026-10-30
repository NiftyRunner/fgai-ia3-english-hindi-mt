"""Joint SentencePiece tokenizer training and inference for Model A (Transformer Scratch).

Trains a joint English-Hindi subword model with special tokens:
<pad>=0, <unk>=1, <bos>=2, <eos>=3.
"""

import json
import logging
import os
import tempfile
from typing import Dict, List, Optional, Union

import sentencepiece as spm
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_sentencepiece_tokenizer(
    train_jsonl_path: str,
    output_prefix: str = "data/spm_joint_16k",
    vocab_size: int = 16000,
    model_type: str = "unigram",
) -> str:
    """Train a joint SentencePiece subword tokenizer on the training split only.

    Why this matters for viva:
    Joint tokenization enables a single shared vocabulary for both source (English)
    and target (Hindi) languages. This allows the model to share representations
    for shared tokens (numbers, punctuation, international proper nouns) while
    bounding the embedding matrix size to 16,000 * 256 (~4.1M parameters).
    """
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    model_file = f"{output_prefix}.model"

    if os.path.exists(model_file):
        logger.info("SentencePiece model already exists at %s. Skipping training.", model_file)
        return model_file

    logger.info("Extracting parallel sentences from %s for SentencePiece training...", train_jsonl_path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as temp_corpus:
        with open(train_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    temp_corpus.write(item["en"].strip() + "\n")
                    temp_corpus.write(item["hi"].strip() + "\n")
        temp_corpus_path = temp_corpus.name

    logger.info("Training SentencePiece (%s, vocab_size=%d)...", model_type, vocab_size)
    # Special tokens configuration:
    # pad_id=0, unk_id=1, bos_id=2, eos_id=3
    spm.SentencePieceTrainer.train(
        input=temp_corpus_path,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        character_coverage=0.9995,  # Ensures high Devanagari coverage
        normalization_rule_name="nmt_nfkc",
    )

    try:
        os.remove(temp_corpus_path)
    except OSError:
        pass

    logger.info("Trained SentencePiece model successfully saved to %s", model_file)
    return model_file


class JointSentencePieceTokenizer:
    """Wrapper around SentencePieceProcessor providing PyTorch tensor batching."""

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece model not found at: {model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.vocab_size = int(self.sp.get_piece_size())

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = self.sp.encode_as_ids(text)
        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        # Strip special token IDs before string decoding
        clean_ids = [i for i in ids if i not in (self.PAD_ID, self.BOS_ID, self.EOS_ID)]
        return self.sp.decode_ids(clean_ids)

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 128,
        add_bos: bool = True,
        add_eos: bool = True,
        pad_to_max: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Encode a batch of strings into padded PyTorch tensors."""
        encoded_list = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

        # Truncate to max_length if necessary
        truncated_list = [enc[:max_length] for enc in encoded_list]
        actual_max = max(len(enc) for enc in truncated_list) if not pad_to_max else max_length
        actual_max = min(actual_max, max_length)

        batch_ids = []
        batch_masks = []

        for enc in truncated_list:
            pad_len = actual_max - len(enc)
            padded = enc + [self.PAD_ID] * pad_len
            # Mask is True for real tokens, False for padding (or 1/0)
            mask = [1] * len(enc) + [0] * pad_len
            batch_ids.append(padded)
            batch_masks.append(mask)

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
        }
