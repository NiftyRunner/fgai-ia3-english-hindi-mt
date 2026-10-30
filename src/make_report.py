"""Generates evaluation summary tables and injects them into README.md from results/ files only.

Adheres strictly to the NO-FABRICATION rule: If result files do not exist yet,
cleanly formats placeholders without inventing any numbers.
"""

import glob
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_all_metrics(metrics_dir: str = "results/metrics") -> List[Dict[str, Any]]:
    """Load all JSON metric files from results/metrics/."""
    pattern = os.path.join(metrics_dir, "metrics_*.json")
    files = glob.glob(pattern)
    records = []
    for f in sorted(files):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                data["_file"] = os.path.basename(f)
                records.append(data)
        except Exception as e:
            logger.warning("Could not read %s: %s", f, e)
    return records


def build_comparison_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Transform list of metrics records into a clean comparison DataFrame."""
    if not records:
        return pd.DataFrame()

    rows = []
    for r in records:
        rows.append({
            "Model": r.get("model_name", "Unknown"),
            "Architecture": r.get("architecture_type", "Unknown"),
            "Adaptation": r.get("adaptation_method", "Unknown"),
            "Total Params": f"{r.get('total_params', 0):,}",
            "Trainable Params": f"{r.get('trainable_params', 0):,}",
            "Train Pairs": f"{r.get('train_data_size', 0):,}",
            "BLEU": r.get("bleu", 0.0),
            "chrF": r.get("chrf", 0.0),
            "chrF++": r.get("chrf_pp", 0.0),
            "Train Time (s)": round(r.get("train_time_sec", 0.0), 1),
            "Inference (ms/sent)": r.get("inference_time_per_sentence_ms", 0.0),
        })

    df = pd.DataFrame(rows)
    return df


def load_temperature_metrics(temp_dir: str = "results/temperature") -> Dict[str, Dict[str, Any]]:
    """Load temperature sweep JSON files from results/temperature/."""
    pattern = os.path.join(temp_dir, "temp_*.json")
    files = glob.glob(pattern)
    data: Dict[str, Dict[str, Any]] = {}
    for f in sorted(files):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                model_key = os.path.basename(f).replace("temp_", "").replace(".json", "")
                data[model_key] = json.load(fp)
        except Exception as e:
            logger.warning("Could not read %s: %s", f, e)
    return data


def generate_temperature_markdown(temp_data: Dict[str, Dict[str, Any]]) -> str:
    """Format temperature sweep results into a comparative markdown table."""
    if not temp_data:
        return "*Temperature evaluation results pending. Run `notebooks/05_temperature_analysis.ipynb`.*"

    rows = []
    for model_name, temps in temp_data.items():
        row = {"Model": model_name}
        for t_key in ["greedy", "0.2", "0.7", "1.0"]:
            val = temps.get(t_key, {})
            bleu = val.get("bleu", "-")
            chrf = val.get("chrf", "-")
            row[f"T={t_key} (BLEU/chrF)"] = f"{bleu} / {chrf}" if bleu != "-" else "-"
        rows.append(row)

    df_temp = pd.DataFrame(rows)
    return df_temp.to_markdown(index=False)


def generate_results_markdown_block(
    records: List[Dict[str, Any]],
    temp_data: Dict[str, Dict[str, Any]],
) -> str:
    """Construct markdown content for injection between README markers."""
    if not records:
        return (
            "### Quantitative Results\n\n"
            "> [!NOTE]\n"
            "> **Pending Execution**: No metrics JSON files found in `results/metrics/`. "
            "Run the notebooks in Google Colab to populate this section.\n\n"
            "| Model | Architecture | Adaptation | Total Params | Trainable Params | BLEU | chrF | Train Time (s) | Inference (ms/sent) |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
            "| Basic Transformer | Encoder-Decoder | Scratch | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` |\n"
            "| mT5-small | Encoder-Decoder | Full FT | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` |\n"
            "| LLaMA-3.2-1B-Instruct | Decoder-Only | QLoRA (4-bit) | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` |\n"
            "| Gemma-3/2-1B/2B-It | Decoder-Only | QLoRA (4-bit) | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` | `<<TO BE FILLED>>` |\n"
        )

    df_comp = build_comparison_dataframe(records)
    table_md = df_comp.to_markdown(index=False)

    # Programmatic winner identification (No hardcoding)
    best_bleu_row = df_comp.loc[df_comp["BLEU"].idxmax()]
    best_chrf_row = df_comp.loc[df_comp["chrF"].idxmax()]

    winner_text = (
        f"**Programmatic Model Summary**:\n"
        f"- **Highest BLEU**: `{best_bleu_row['Model']}` with score **{best_bleu_row['BLEU']}**\n"
        f"- **Highest chrF**: `{best_chrf_row['Model']}` with score **{best_chrf_row['chrF']}**\n"
    )

    temp_table_md = generate_temperature_markdown(temp_data)

    block = (
        "### Quantitative Performance Comparison\n\n"
        "Evaluated on identical 1,000-sentence test set (`data/splits/test.jsonl`) with sacreBLEU 13a tokenization:\n\n"
        f"{table_md}\n\n"
        f"{winner_text}\n\n"
        "### Sampling Temperature Analysis (300-Sentence Slice)\n\n"
        f"{temp_table_md}\n\n"
        "Visual plots generated:\n"
        "- [BLEU vs Temperature](results/plots/temperature_vs_bleu.png)\n"
        "- [chrF vs Temperature](results/plots/temperature_vs_chrf.png)\n"
    )
    return block


def update_readme(
    readme_path: str = "README.md",
    metrics_dir: str = "results/metrics",
    temp_dir: str = "results/temperature",
) -> None:
    """Inject generated tables into README.md between RESULTS markers."""
    if not os.path.exists(readme_path):
        logger.error("README.md not found at %s", readme_path)
        return

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    start_marker = "<!-- RESULTS_START -->"
    end_marker = "<!-- RESULTS_END -->"

    if start_marker not in content or end_marker not in content:
        logger.warning("Markers <!-- RESULTS_START --> and <!-- RESULTS_END --> not found in %s", readme_path)
        return

    records = load_all_metrics(metrics_dir)
    temp_data = load_temperature_metrics(temp_dir)
    block_md = generate_results_markdown_block(records, temp_data)

    pattern = re.compile(f"{re.escape(start_marker)}.*?{re.escape(end_marker)}", re.DOTALL)
    new_content = pattern.sub(f"{start_marker}\n\n{block_md}\n\n{end_marker}", content)

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    logger.info("Successfully updated %s with latest results block.", readme_path)


if __name__ == "__main__":
    records = load_all_metrics()
    if records:
        df = build_comparison_dataframe(records)
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/comparison_table.csv", index=False)
        with open("results/comparison_table.md", "w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False))
        logger.info("Saved comparison tables to results/comparison_table.{csv,md}")
    update_readme()
