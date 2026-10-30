# Google Colab T4 Execution Guide

This guide walks you through executing the English $\to$ Hindi Machine Translation notebooks on a **free Google Colab T4 GPU (16 GB VRAM, ~12 GB System RAM)**.

---

## 1. Prerequisites & Account Setup

### A. Gated Model Access on Hugging Face
Both LLaMA-3.2 and Gemma are gated models. You must accept their license agreements:
1. Log in to [Hugging Face](https://huggingface.co).
2. Visit [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and submit the access request form.
3. Visit [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) and [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) and accept the terms.
4. Generate a User Access Token with `read` permissions at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### B. Configure Colab Secrets
In Google Colab:
1. Click the **Key icon** (Secrets) on the left sidebar.
2. Click **Add new secret**.
3. Name: `HF_TOKEN`
4. Value: Paste your Hugging Face read token.
5. Toggle **Notebook access** to ON.

---

## 2. Hardware Configuration
Always ensure your runtime is assigned a GPU:
1. Go to **Runtime $\to$ Change runtime type**.
2. Hardware accelerator: **T4 GPU**.
3. GPU class: **Standard**.
4. Click **Save**.

Verify GPU allocation inside any notebook cell:
```python
!nvidia-smi
```
*Note: The T4 has Compute Capability 7.5. It supports `float16` and `int8`/`int4`, but does NOT have hardware acceleration for `bfloat16` and cannot run FlashAttention-2.*

---

## 3. Google Drive Persistence & Resumability

Colab free sessions may disconnect after inactivity or timeouts. All notebooks persist state directly to Google Drive so you never lose progress:

### Root Drive Directory
Every notebook uses:
```python
DRIVE_DIR = "/content/drive/MyDrive/mt-fgai-assignment3"
```
The first cell will mount Drive and clone/pull this GitHub repo:
```python
from google.colab import drive
import os

drive.mount('/content/drive')
os.makedirs(DRIVE_DIR, exist_ok=True)
```

### Resume Behavior by Component:
1. **Model A (`01_transformer_scratch.ipynb`)**:
   - Saves `last_checkpoint.pt` at the end of each epoch containing model weights, optimizer state, scheduler state, scaler state, epoch number, and `cumulative_train_time_sec`.
   - If disconnected, simply re-run the notebook. It detects `last_checkpoint.pt` on Drive and resumes from the exact epoch with cumulative time intact.
2. **Models B, C, D (`02_mt5`, `03_llama`, `04_gemma`)**:
   - Save step checkpoints every 200 steps to Drive with `save_total_limit=2`.
   - Training is called with `resume_from_checkpoint=True`.
   - If disconnected, running the notebook picks up the latest step checkpoint automatically.
3. **Inference & Evaluations**:
   - Before running test set inference or temperature sweeps, notebooks check if `results/metrics/metrics_<model>.json` already exists. If present, expensive inference is skipped and results are loaded directly.

---

## 4. Notebook Execution Order & Approximate Runtimes

> [!IMPORTANT]
> Execute the notebooks strictly in order (00 through 07) because later notebooks depend on the output files produced by earlier ones.

| Order | Notebook | Purpose | Approx. Runtime (T4) | Primary Output |
| :---: | :--- | :--- | :---: | :--- |
| **00** | `00_data_prep.ipynb` | Cleans dataset, creates fixed splits, computes length stats | 3 – 6 min | `data/splits/*.jsonl`, `dataset_stats.json` |
| **01** | `01_transformer_scratch.ipynb` | Trains joint SentencePiece (16k) & Transformer from scratch | 30 – 45 min | `predictions_transformer_scratch.csv`, `metrics_transformer_scratch.json` |
| **02** | `02_mt5_finetune.ipynb` | Fine-tunes `google/mt5-small` in fp32 | 50 – 75 min | `predictions_mt5_small.csv`, `metrics_mt5_small.json` |
| **03** | `03_llama_qlora.ipynb` | Zero-shot eval + QLoRA fine-tunes LLaMA-3.2-1B on 20k subset | 50 – 70 min | `predictions_llama_qlora.csv`, `metrics_llama_qlora.json` |
| **04** | `04_gemma_qlora.ipynb` | Zero-shot eval + QLoRA fine-tunes Gemma with fallback | 50 – 70 min | `predictions_gemma_qlora.csv`, `metrics_gemma_qlora.json` |
| **05** | `05_temperature_analysis.ipynb` | Sequential temperature sweep ($T=0.2, 0.7, 1.0$) on 300 test slice | 25 – 35 min | `temp_<model>.json`, `temp_<model>_samples.csv`, plots |
| **06** | `06_comparison.ipynb` | Generates master comparison tables, trade-off charts, tokenizer stats | 2 – 3 min | `results/plots/*.png`, `comparison_table.csv` |
| **07** | `07_responsible_ai_probe.ipynb` | Runs 40-sentence ethical probe + heuristic anomaly checks | 8 – 12 min | `probe_outputs_annotated.csv` |

---

## 5. Synchronizing Results Back to GitHub

After running the notebooks on Google Colab, copy the generated results from your Google Drive folder back into your local repository clone:

```bash
# In your local repo root:
# Copy results from Drive mount or download results/ from Google Drive UI
python scripts/verify_outputs.py
python src/make_report.py
```
This updates `README.md` with real calculated tables and plots!

Then commit and push:
```bash
git add data/splits/ results/ README.md
git commit -m "Add experiment results, metrics, and ethical probe outputs"
git push origin main
```
*Note: Model checkpoint weights (`*.pt`, `checkpoint-*`) are automatically ignored by `.gitignore` and must stay in Google Drive.*
