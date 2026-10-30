"""Data loading, cleaning, filtering, and split management for English-to-Hindi MT.

Dataset: IIT Bombay English-Hindi Parallel Corpus (cfilt/iitb-english-hindi).
Adheres strictly to the seed=42 deterministic partitioning policy with zero train-test overlap.
"""

import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Unicode range for Devanagari script: U+0900 to U+097F
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")
# Regex to isolate alphabetic characters across scripts (excluding digits, symbols, and whitespace)
ALPHA_REGEX = re.compile(r"[^\W\d_]", re.UNICODE)


def is_valid_devanagari(text: str, min_alpha_ratio: float = 0.50) -> bool:
    """Verify that Hindi text has substantial Devanagari character content.

    Why this matters for viva:
    Noise in web-scraped Hindi corpora often includes untranslated English text,
    corrupted encodings, or pure punctuation. Evaluating against alphabetic characters
    only (ignoring digits and punctuation) prevents false positives or false negatives.

    Args:
        text: The target Hindi sentence string.
        min_alpha_ratio: Minimum fraction of alphabetic characters that must be Devanagari.

    Returns:
        True if text has at least one alphabetic character and >= min_alpha_ratio are Devanagari.
    """
    if not text:
        return False

    alpha_chars = ALPHA_REGEX.findall(text)
    if not alpha_chars:
        return False

    devanagari_chars = [c for c in alpha_chars if DEVANAGARI_REGEX.match(c)]
    ratio = len(devanagari_chars) / len(alpha_chars)
    return ratio >= min_alpha_ratio


def clean_sentence_pair(
    src_text: str,
    tgt_text: str,
    max_words: int = 40,
    min_ratio: float = 0.33,
    max_ratio: float = 3.00,
    min_devanagari_alpha_ratio: float = 0.50,
) -> bool:
    """Determine whether an English-Hindi sentence pair passes quality and length thresholds.

    Criteria:
    1. Neither side is empty or whitespace-only.
    2. Word count on both sides is <= max_words (40 words avoids subword truncation).
    3. Length ratio (src_words / tgt_words) is within [min_ratio, max_ratio] to exclude truncation errors.
    4. Target text satisfies the Devanagari alphabetic threshold.
    """
    if not src_text or not tgt_text:
        return False

    src_str = src_text.strip()
    tgt_str = tgt_text.strip()

    if not src_str or not tgt_str:
        return False

    src_words = src_str.split()
    tgt_words = tgt_str.split()
    src_len = len(src_words)
    tgt_len = len(tgt_words)

    # Word count constraint (prevents OOM and truncation in models)
    if src_len > max_words or tgt_len > max_words or src_len == 0 or tgt_len == 0:
        return False

    # Length ratio constraint (flags misaligned or partial translations)
    ratio = src_len / tgt_len
    if ratio < min_ratio or ratio > max_ratio:
        return False

    # Devanagari script content check
    if not is_valid_devanagari(tgt_str, min_devanagari_alpha_ratio):
        return False

    return True


def load_raw_iitb_dataset(
    dataset_name: str = "cfilt/iitb-english-hindi",
    src_lang: str = "en",
    tgt_lang: str = "hi",
) -> Dict[str, List[Dict[str, str]]]:
    """Load the raw IITB dataset from Hugging Face datasets.

    Extracts dictionary pairs {'en': ..., 'hi': ...} from the nested 'translation' field.
    """
    from datasets import load_dataset

    logger.info("Downloading/loading dataset: %s", dataset_name)
    ds = load_dataset(dataset_name)

    raw_splits: Dict[str, List[Dict[str, str]]] = {}
    for split_key in ds.keys():
        examples = []
        for item in ds[split_key]:
            trans = item.get("translation", {})
            src = trans.get(src_lang, "").strip()
            tgt = trans.get(tgt_lang, "").strip()
            if src and tgt:
                examples.append({"en": src, "hi": tgt})
        raw_splits[split_key] = examples
        logger.info("Loaded raw split '%s': %d sentence pairs", split_key, len(examples))

    return raw_splits


def build_deterministic_splits(
    raw_splits: Dict[str, List[Dict[str, str]]],
    seed: int = 42,
    train_size: int = 50000,
    val_size: int = 1000,
    test_size: int = 1000,
    max_words: int = 40,
    min_ratio: float = 0.33,
    max_ratio: float = 3.00,
    min_devanagari_alpha_ratio: float = 0.50,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Build train, val, and test splits with strict zero source leakage.

    Split Construction Protocol:
    1. Clean the official 'test' split and deterministically sample test_size (1,000) pairs.
    2. Register all test source sentences in a seen set.
    3. Clean the official 'train' split, eliminating any pairs whose source exists in test.
    4. Carve exactly val_size (1,000) pairs from the cleaned train pool.
    5. Register all validation source sentences in the seen set.
    6. Sample exactly train_size (50,000) pairs from the remaining pool.
    """
    random.seed(seed)

    # 1. Process and carve Test Set from original 'test' split
    raw_test = raw_splits.get("test", [])
    clean_test_pool = []
    seen_test_pairs: Set[Tuple[str, str]] = set()

    for item in raw_test:
        pair = (item["en"], item["hi"])
        if pair not in seen_test_pairs and clean_sentence_pair(
            item["en"], item["hi"], max_words, min_ratio, max_ratio, min_devanagari_alpha_ratio
        ):
            clean_test_pool.append(item)
            seen_test_pairs.add(pair)

    if len(clean_test_pool) < test_size:
        raise ValueError(
            f"Cleaned test pool ({len(clean_test_pool)}) has fewer pairs than requested {test_size}."
        )

    # Deterministic shuffle with seed
    random.shuffle(clean_test_pool)
    selected_test = clean_test_pool[:test_size]

    # Store normalized source strings to prevent any train/val leakage
    reserved_sources: Set[str] = {item["en"].strip().lower() for item in selected_test}
    logger.info("Selected %d test pairs from %d cleaned test candidates", len(selected_test), len(clean_test_pool))

    # 2. Process and carve Validation & Train Sets from original 'train' split
    raw_train = raw_splits.get("train", [])
    clean_train_pool = []
    seen_train_pairs: Set[Tuple[str, str]] = set()

    for item in raw_train:
        src_norm = item["en"].strip().lower()
        pair = (item["en"], item["hi"])
        # Strictly skip if exact pair seen or if source appears in test set
        if src_norm in reserved_sources or pair in seen_train_pairs:
            continue
        if clean_sentence_pair(
            item["en"], item["hi"], max_words, min_ratio, max_ratio, min_devanagari_alpha_ratio
        ):
            clean_train_pool.append(item)
            seen_train_pairs.add(pair)

    logger.info("Found %d clean candidate pairs in train split disjoint from test", len(clean_train_pool))

    if len(clean_train_pool) < (train_size + val_size):
        raise ValueError(
            f"Clean train pool ({len(clean_train_pool)}) is smaller than required train+val ({train_size + val_size})."
        )

    # Deterministic shuffle
    random.shuffle(clean_train_pool)

    # Carve validation set
    selected_val = clean_train_pool[:val_size]
    for item in selected_val:
        reserved_sources.add(item["en"].strip().lower())

    # Carve training set (guaranteed disjoint from test and val)
    candidate_train = clean_train_pool[val_size:]
    selected_train = []
    for item in candidate_train:
        src_norm = item["en"].strip().lower()
        if src_norm not in reserved_sources:
            selected_train.append(item)
            if len(selected_train) == train_size:
                break

    if len(selected_train) < train_size:
        raise ValueError(f"Could only obtain {len(selected_train)} train pairs disjoint from val/test.")

    logger.info(
        "Splits created successfully: train=%d, val=%d, test=%d (Zero source overlap guaranteed)",
        len(selected_train),
        len(selected_val),
        len(selected_test),
    )

    # Add numeric IDs
    def assign_ids(items: List[Dict[str, str]], prefix: str) -> List[Dict[str, Any]]:
        return [{"id": f"{prefix}_{idx:06d}", "en": it["en"], "hi": it["hi"]} for idx, it in enumerate(items)]

    final_train = assign_ids(selected_train, "train")
    final_val = assign_ids(selected_val, "val")
    final_test = assign_ids(selected_test, "test")

    # Compute comprehensive dataset statistics
    en_train_lens = [len(x["en"].split()) for x in final_train]
    hi_train_lens = [len(x["hi"].split()) for x in final_train]
    ratios = [h / e for e, h in zip(en_train_lens, hi_train_lens) if e > 0]
    ratios.sort()
    median_ratio = ratios[len(ratios) // 2] if ratios else 1.0

    stats = {
        "dataset_name": "cfilt/iitb-english-hindi",
        "random_seed": seed,
        "splits": {
            "train_count": len(final_train),
            "val_count": len(final_val),
            "test_count": len(final_test),
        },
        "filtering": {
            "max_words_per_sentence": max_words,
            "min_length_ratio": min_ratio,
            "max_length_ratio": max_ratio,
            "min_devanagari_alpha_ratio": min_devanagari_alpha_ratio,
        },
        "train_statistics": {
            "avg_en_words": round(sum(en_train_lens) / len(en_train_lens), 2),
            "avg_hi_words": round(sum(hi_train_lens) / len(hi_train_lens), 2),
            "median_en_words": sorted(en_train_lens)[len(en_train_lens) // 2],
            "median_hi_words": sorted(hi_train_lens)[len(hi_train_lens) // 2],
            "median_hi_en_word_ratio": round(median_ratio, 4),
        },
    }

    return final_train, final_val, final_test, stats


def save_splits_as_jsonl(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    stats: Dict[str, Any],
    output_dir: str = "data/splits",
) -> None:
    """Save dataset splits into jsonl files and export dataset_stats.json."""
    os.makedirs(output_dir, exist_ok=True)

    def write_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Saved %d rows to %s", len(data), path)

    write_jsonl(os.path.join(output_dir, "train.jsonl"), train_data)
    write_jsonl(os.path.join(output_dir, "val.jsonl"), val_data)
    write_jsonl(os.path.join(output_dir, "test.jsonl"), test_data)

    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("Saved dataset statistics to %s", stats_path)


def load_split_from_jsonl(split_name: str, splits_dir: str = "data/splits") -> List[Dict[str, Any]]:
    """Load a specific jsonl split from disk (e.g. 'train', 'val', or 'test')."""
    file_path = os.path.join(splits_dir, f"{split_name}.jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Split file not found: {file_path}. Run 00_data_prep.ipynb first.")

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
