"""Evaluation metrics, parameter accounting, and timing benchmarks for Machine Translation.

Computes sacreBLEU, chrF, chrF++, counts parameters, profiles GPU latency,
and exports standardized JSON and CSV artifacts.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import sacrebleu
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_sacrebleu(hypotheses: List[str], references: List[str]) -> Tuple[float, str]:
    """Compute corpus-level BLEU score with standard 13a tokenization and return full signature."""
    try:
        score = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="13a")
        sig = "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp"
        try:
            if hasattr(score, "signature"):
                sig = str(score.signature)
            elif hasattr(score, "format"):
                sig = str(score.format())
        except Exception:
            pass
        return round(float(score.score), 2), sig
    except Exception as e:
        logger.warning("Error in compute_sacrebleu: %s", e)
        return 0.0, "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp"


def compute_chrf(
    hypotheses: List[str], references: List[str]
) -> Tuple[float, float, str]:
    """Compute corpus chrF (character n-grams) and chrF++ (character + word n-grams)."""
    try:
        score_chrf = sacrebleu.corpus_chrf(hypotheses, [references])
        score_pp = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
        sig = "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no"
        try:
            if hasattr(score_chrf, "signature"):
                sig = str(score_chrf.signature)
            elif hasattr(score_chrf, "format"):
                sig = str(score_chrf.format())
        except Exception:
            pass
        return round(float(score_chrf.score), 2), round(float(score_pp.score), 2), sig
    except Exception as e:
        logger.warning("Error in compute_chrf: %s", e)
        return 0.0, 0.0, "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Calculate the total and trainable parameter counts of any PyTorch model.

    Distinguishes frozen base parameters from active LoRA adapter parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }


class CumulativeTimer:
    """Timer supporting cumulative elapsed duration across Google Colab session disconnects.

    Preserves total training duration when resuming from intermediate checkpoints.
    """

    def __init__(self, initial_seconds: float = 0.0):
        self.cumulative_seconds = float(initial_seconds)
        self.start_time: Optional[float] = None
        self.is_running = False

    def start(self) -> None:
        if not self.is_running:
            self.start_time = time.perf_counter()
            self.is_running = True

    def stop(self) -> float:
        if self.is_running and self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            self.cumulative_seconds += elapsed
            self.is_running = False
            self.start_time = None
        return self.cumulative_seconds

    def elapsed(self) -> float:
        total = self.cumulative_seconds
        if self.is_running and self.start_time is not None:
            total += time.perf_counter() - self.start_time
        return total


def get_gpu_device_name() -> str:
    """Detect and return the active GPU hardware descriptor, or CPU if unavailable."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def save_predictions_csv(
    ids: List[Union[str, int]],
    sources: List[str],
    references: List[str],
    hypotheses: List[str],
    output_path: str,
) -> None:
    """Save test predictions into standard CSV with columns: id, source, reference, hypothesis."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame({
        "id": ids,
        "source": sources,
        "reference": references,
        "hypothesis": hypotheses,
    })
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved %d predictions to %s", len(df), output_path)


def save_metrics_json(
    model_name: str,
    architecture_type: str,
    adaptation_method: str,
    total_params: int,
    trainable_params: int,
    train_data_size: int,
    train_time_sec: float,
    inference_time_total_sec: float,
    num_test_samples: int,
    bleu: float,
    bleu_signature: str,
    chrf: float,
    chrf_pp: float,
    chrf_signature: str,
    decoding_settings: Dict[str, Any],
    batch_size: int,
    hyperparameters: Dict[str, Any],
    output_path: str,
    cleaned_outputs_count: int = 0,
) -> Dict[str, Any]:
    """Serialize full evaluation benchmark metrics into standardized JSON schema.

    Strict schema used by all 4 models and make_report.py.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    inference_per_sentence = (
        round(inference_time_total_sec / num_test_samples, 6) if num_test_samples > 0 else 0.0
    )

    metrics_payload = {
        "model_name": model_name,
        "architecture_type": architecture_type,  # 'encoder_decoder' or 'decoder_only'
        "adaptation_method": adaptation_method,  # 'from_scratch', 'full_finetune', 'qlora_4bit', 'zero_shot'
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_data_size": train_data_size,
        "train_time_sec": round(train_time_sec, 2),
        "inference_time_total_sec": round(inference_time_total_sec, 3),
        "inference_time_per_sentence_sec": inference_per_sentence,
        "inference_time_per_sentence_ms": round(inference_per_sentence * 1000.0, 2),
        "num_test_samples": num_test_samples,
        "bleu": bleu,
        "bleu_signature": bleu_signature,
        "chrf": chrf,
        "chrf_pp": chrf_pp,
        "chrf_signature": chrf_signature,
        "batch_size": batch_size,
        "decoding_settings": decoding_settings,
        "cleaned_outputs_count": cleaned_outputs_count,
        "gpu": get_gpu_device_name(),
        "hyperparameters": hyperparameters,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved standardized evaluation metrics to %s", output_path)
    return metrics_payload
