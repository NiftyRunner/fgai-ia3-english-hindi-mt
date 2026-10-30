"""Verification script to validate experimental outputs, schema conformance, and row counts.

Run this script after completing notebook executions in Google Colab:
    python scripts/verify_outputs.py
"""

import json
import os
import sys
from typing import List, Tuple

import pandas as pd

REQUIRED_METRIC_KEYS = [
    "model_name",
    "architecture_type",
    "total_params",
    "trainable_params",
    "train_time_sec",
    "inference_time_total_sec",
    "inference_time_per_sentence_sec",
    "bleu",
    "bleu_signature",
    "chrf",
    "chrf_pp",
    "batch_size",
    "gpu",
    "hyperparameters",
]

CORE_MODELS = [
    "transformer_scratch",
    "mt5_small",
    "llama_qlora",
    "gemma_qlora",
]


def check_prediction_file(file_path: str, expected_rows: int = 1000) -> Tuple[bool, str]:
    if not os.path.exists(file_path):
        return False, f"Missing prediction file: {file_path}"
    try:
        df = pd.read_csv(file_path)
        required_cols = {"id", "source", "reference", "hypothesis"}
        if not required_cols.issubset(set(df.columns)):
            return False, f"{file_path} missing required columns: {required_cols - set(df.columns)}"
        if len(df) != expected_rows:
            return False, f"{file_path} row count mismatch: found {len(df)}, expected {expected_rows}"
        return True, f"OK ({len(df)} rows)"
    except Exception as e:
        return False, f"Error reading {file_path}: {e}"


def check_metrics_file(file_path: str) -> Tuple[bool, str]:
    if not os.path.exists(file_path):
        return False, f"Missing metrics file: {file_path}"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        missing_keys = [k for k in REQUIRED_METRIC_KEYS if k not in data]
        if missing_keys:
            return False, f"{file_path} missing keys: {missing_keys}"
        return True, f"OK (BLEU={data.get('bleu')}, chrF={data.get('chrf')})"
    except Exception as e:
        return False, f"Error parsing {file_path}: {e}"


def check_temperature_files(model_name: str, expected_rows: int = 300) -> List[Tuple[bool, str]]:
    results = []
    # 1. Summary JSON
    json_path = f"results/temperature/temp_{model_name}.json"
    if not os.path.exists(json_path):
        results.append((False, f"Missing temperature summary: {json_path}"))
    else:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                t_data = json.load(f)
            expected_keys = {"greedy", "0.2", "0.7", "1.0"}
            if not expected_keys.issubset(set(t_data.keys())):
                results.append((False, f"{json_path} missing temperature keys"))
            else:
                results.append((True, f"{json_path} OK"))
        except Exception as e:
            results.append((False, f"Error in {json_path}: {e}"))

    # 2. Per-sentence samples CSV
    csv_path = f"results/temperature/temp_{model_name}_samples.csv"
    if not os.path.exists(csv_path):
        results.append((False, f"Missing temperature samples CSV: {csv_path}"))
    else:
        try:
            df = pd.read_csv(csv_path)
            if len(df) != expected_rows:
                results.append((False, f"{csv_path} row count mismatch: found {len(df)}, expected {expected_rows}"))
            else:
                results.append((True, f"{csv_path} OK ({len(df)} rows)"))
        except Exception as e:
            results.append((False, f"Error in {csv_path}: {e}"))

    return results


def main() -> int:
    print("=" * 70)
    print("Verifying Experiment Artifacts & Output Schemas")
    print("=" * 70)

    all_passed = True

    # 1. Check Predictions CSVs
    print("\n[1/4] Checking Test Set Predictions (Expected: 1,000 rows)...")
    for model in CORE_MODELS:
        pred_file = f"results/predictions/predictions_{model}.csv"
        ok, msg = check_prediction_file(pred_file, 1000)
        status = "PASSED" if ok else "FAILED"
        print(f"  [{status}] {model}: {msg}")
        if not ok:
            all_passed = False

    # 2. Check Metrics JSONs
    print("\n[2/4] Checking Metrics JSON Schema & Signatures...")
    for model in CORE_MODELS:
        metric_file = f"results/metrics/metrics_{model}.json"
        ok, msg = check_metrics_file(metric_file)
        status = "PASSED" if ok else "FAILED"
        print(f"  [{status}] {model}: {msg}")
        if not ok:
            all_passed = False

    # 3. Check Temperature Files
    print("\n[3/4] Checking Temperature Sweeps (JSON + 300-row Samples CSV)...")
    for model in CORE_MODELS:
        t_results = check_temperature_files(model, 300)
        for ok, msg in t_results:
            status = "PASSED" if ok else "FAILED"
            print(f"  [{status}] {model}: {msg}")
            if not ok:
                all_passed = False

    # 4. Check Ethical Probe Outputs
    print("\n[4/4] Checking Responsible AI Probe Artifacts...")
    ethics_file = "results/ethics/probe_outputs_annotated.csv"
    if not os.path.exists(ethics_file):
        print(f"  [FAILED] Missing ethical probe outputs: {ethics_file}")
        all_passed = False
    else:
        df_eth = pd.read_csv(ethics_file)
        print(f"  [PASSED] {ethics_file} OK ({len(df_eth)} total probe & spot-check items)")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CHECKS PASSED: Your results are complete and ready for submission!")
        return 0
    else:
        print("SOME CHECKS FAILED: Pending files detected. Ensure all notebooks finished on Colab.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
