# University Assignment Submission Template

Use this document to prepare the Word document (`YourName_RegNum_FGAI_Assignment3.docx`) required by **Step 3** of the assignment specification before converting to PDF and uploading to the LMS.

---

## Cover Page Details

| Field | Submission Details |
| :--- | :--- |
| **Student Name** | `[Insert Your Full Name]` |
| **Registration Number** | `[Insert Your Registration Number]` |
| **Course Code & Title** | ICT4403 - Foundations of Generative AI |
| **Assignment Number** | Assignment 3 (Autoregressive & Encoder-Decoder MT Systems) |
| **Title of Developed System** | **A Comparative Study of Encoder-Decoder and Autoregressive Decoder-Only Architectures for English-to-Hindi Machine Translation** |
| **Dataset Name** | IIT Bombay English-Hindi Parallel Corpus (`cfilt/iitb-english-hindi`) |
| **Dataset Source Link** | [https://huggingface.co/datasets/cfilt/iitb-english-hindi](https://huggingface.co/datasets/cfilt/iitb-english-hindi) |
| **GitHub Repository Link** | `[Insert Your GitHub Repository URL here after uploading]` |

---

## Suggested Report Sections to Paste into the Submission Document

1. **Executive Summary**:
   - A brief overview of the translation task (English $\to$ Hindi).
   - Comparison of 4 models: Basic PyTorch Transformer (from scratch), mT5-small (full fine-tuning), LLaMA-3.2-1B-Instruct (4-bit QLoRA), and Gemma-3-1b-it / Gemma-2-2b-it (4-bit QLoRA).
2. **Dataset Preprocessing & Split Statistics**:
   - Paste the table generated in `00_data_prep.ipynb` and stored in `data/splits/dataset_stats.json`.
3. **Model Performance Summary Table**:
   - Paste the master comparison table produced by `src/make_report.py` (BLEU, chrF, total/trainable params, training time, inference latency).
4. **Sampling Temperature Observations**:
   - Insert the plots from `results/plots/temperature_vs_bleu.png` and `results/plots/temperature_vs_chrf.png`.
   - Discuss the trade-off between greedy search fidelity and sampling creativity.
5. **Responsible AI & Ethical Analysis**:
   - Summarize observations from `results/ethics/probe_outputs_annotated.csv`.
   - Discuss gender bias, cultural term handling, hallucination heuristics, and mitigation strategies.
6. **Conclusion**:
   - Identify the best model based on BLEU/chrF and inference efficiency.
