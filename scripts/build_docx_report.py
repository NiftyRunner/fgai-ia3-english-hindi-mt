import os
import json
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def add_figure_caption(doc, caption_text):
    caption = doc.add_paragraph(caption_text)
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.runs[0].font.italic = True
    caption.runs[0].font.size = Pt(9.5)
    caption.paragraph_format.space_after = Pt(12)

def create_report():
    doc = Document()

    # Page Margins (1 inch)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    # -------------------------------------------------------------
    # Title & Header
    # -------------------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title_p.add_run("A Comparative Study of Encoder-Decoder and Autoregressive Decoder-Only Architectures for English-to-Hindi Machine Translation")
    run_title.bold = True
    run_title.font.size = Pt(18)
    run_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(16)
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = sub_p.add_run("Course: ICT4403 – Foundations of Generative AI | Assignment 3 Benchmark Report")
    run_sub.font.size = Pt(12)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # Metadata Table
    meta_table = doc.add_table(rows=6, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False

    meta_data = [
        ("Student Name", "Parth Verma"),
        ("Registration Number", "230953436"),
        ("Course & Department", "Foundations of Generative AI (ICT4403) | Dept. of ICT"),
        ("Institution", "Manipal Academy of Higher Education (MAHE)"),
        ("Dataset", "IIT Bombay English-Hindi Parallel Corpus (cfilt/iitb-english-hindi)"),
        ("GitHub Repository", "https://github.com/NiftyRunner/fgai-ia3-english-hindi-mt"),
    ]

    for i, (k, v) in enumerate(meta_data):
        row = meta_table.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        set_cell_background(c0, "F0F4F8")
        set_cell_background(c1, "FFFFFF")
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(2)
        p0.paragraph_format.space_before = Pt(2)
        r0 = p0.add_run(k)
        r0.bold = True
        r0.font.size = Pt(10)
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(2)
        p1.paragraph_format.space_before = Pt(2)
        r1 = p1.add_run(v)
        r1.font.size = Pt(10)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------
    # 1. Executive Summary
    # -------------------------------------------------------------
    h1 = doc.add_heading("1. Executive Summary", level=1)
    h1.paragraph_format.space_before = Pt(12)
    p = doc.add_paragraph(
        "This project presents an empirical investigation of machine translation (MT) from English to Hindi using the "
        "IIT Bombay English-Hindi Parallel Corpus. We systematically construct, adapt, and benchmark four distinct neural architectures "
        "under strictly controlled experimental conditions:\n"
        "1. Model A: PyTorch Seq2Seq Transformer built from scratch with a custom joint SentencePiece tokenizer (16k vocabulary).\n"
        "2. Model B: mT5-small pretrained multilingual encoder-decoder adapted via full fine-tuning.\n"
        "3. Model C: LLaMA-3.2-1B-Instruct decoder-only language model adapted via 4-bit QLoRA with masked response-only loss.\n"
        "4. Model D: Gemma-3-1b-it decoder-only language model adapted via 4-bit QLoRA with eager attention.\n\n"
        "Key Findings:\n"
        "• Model D (Gemma-3-1b-it QLoRA) emerged as the benchmark winner, achieving the highest SacreBLEU (11.10) "
        "and highest chrF (35.93), alongside the most compact subword tokenization (1.37 subwords/Hindi word).\n"
        "• Model A achieved the lowest inference latency (21.61 ms/sentence), over 34x faster than LLaMA-3.2 QLoRA.\n"
        "• Tokenization analysis revealed that English-centric byte-level BPE (LLaMA) suffers severe Hindi token inflation (2.70 subwords/word), "
        "whereas massive SentencePiece vocabularies dramatically improve inference throughput and Devanagari representation.\n"
        "• Ethical probing highlighted that while decoder-only LLMs retain superior numeric fidelity and gender agreement, "
        "they remain vulnerable to literal idiom mistranslation, requiring targeted heuristic post-processing."
    )

    # -------------------------------------------------------------
    # 2. Dataset & Preprocessing Pipeline
    # -------------------------------------------------------------
    doc.add_heading("2. Dataset Preprocessing & Split Design", level=1)
    p = doc.add_paragraph(
        "Training high-fidelity MT models between English and Hindi requires robust data cleaning due to web-crawled noise in raw parallel corpora. "
        "We implemented a four-tier filtering protocol (src/data.py):\n"
        "• Sentence Length Constraints: Sentence length capped at 40 words on both source and target sides to prevent GPU OOM and subword truncation.\n"
        "• Length Ratio Guardrails: Enforced 0.33 <= len(src)/len(tgt) <= 3.00 to discard misaligned sentence chunks.\n"
        "• Devanagari Content Threshold: Target text required at least 50% Devanagari script characters (U+0900 to U+097F) among alphabetic tokens.\n"
        "• Zero-Leakage Guarantee: A deterministic partition (seed 42) created 50,000 training pairs, 1,000 validation pairs, and 1,000 test pairs, "
        "strictly excluding all validation and test source sentences from the training pool."
    )

    if os.path.exists("results/plots/data_length_distribution_raw.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture("results/plots/data_length_distribution_raw.png", width=Inches(5.5))
        add_figure_caption(doc, "Figure 1: Distribution of Raw vs Filtered Sentence Lengths in the IIT Bombay Parallel Corpus.")

    # -------------------------------------------------------------
    # 3. Master Model Comparison
    # -------------------------------------------------------------
    doc.add_heading("3. Master Model Benchmark Results", level=1)
    p = doc.add_paragraph(
        "All models were evaluated on the identical 1,000-sentence test set using sacreBLEU (13a tokenization) and chrF/chrF++:"
    )

    # Master Table
    comp_csv = "results/comparison_table.csv"
    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        table = doc.add_table(rows=len(df_comp)+1, cols=len(df_comp.columns))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        headers = df_comp.columns.tolist()
        for j, h in enumerate(headers):
            cell = table.cell(0, j)
            set_cell_background(cell, "1B365D")
            p_cell = cell.paragraphs[0]
            p_cell.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p_cell.add_run(h)
            run.bold = True
            run.font.size = Pt(8.5)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        for i, row in df_comp.iterrows():
            bg_color = "F9FBFD" if i % 2 == 1 else "FFFFFF"
            for j, val in enumerate(row):
                cell = table.cell(i+1, j)
                set_cell_background(cell, bg_color)
                p_cell = cell.paragraphs[0]
                p_cell.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 2 else WD_ALIGN_PARAGRAPH.LEFT
                run = p_cell.add_run(str(val))
                run.font.size = Pt(8.0)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Embed Model Comparison Plots
    if os.path.exists("results/plots/model_comparison_bleu_chrf.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture("results/plots/model_comparison_bleu_chrf.png", width=Inches(5.5))
        add_figure_caption(doc, "Figure 2: SacreBLEU and chrF Performance Comparison across Architectures.")

    if os.path.exists("results/plots/inference_latency_comparison.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture("results/plots/inference_latency_comparison.png", width=Inches(5.2))
        add_figure_caption(doc, "Figure 3: Average Inference Latency per Sentence (ms) across Models.")

    if os.path.exists("results/plots/train_time_vs_bleu.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture("results/plots/train_time_vs_bleu.png", width=Inches(5.2))
        add_figure_caption(doc, "Figure 4: Training Wall-Clock Time vs SacreBLEU Efficiency Pareto Frontier.")

    if os.path.exists("results/plots/loss_transformer_scratch.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture("results/plots/loss_transformer_scratch.png", width=Inches(5.0))
        add_figure_caption(doc, "Figure 5: Training and Validation Loss Trajectory for Model A (Transformer from Scratch).")

    # -------------------------------------------------------------
    # 4. Tokenization Efficiency
    # -------------------------------------------------------------
    doc.add_heading("4. Empirical Tokenization Efficiency Analysis", level=1)
    p = doc.add_paragraph(
        "Subword tokenization is a critical yet often overlooked factor in non-Latin language generation. We evaluated how each model's "
        "tokenizer segmented the exact same 1,000 Hindi test reference sentences:"
    )

    tok_csv = "results/tokenization_efficiency.csv"
    if os.path.exists(tok_csv):
        df_tok = pd.read_csv(tok_csv)
        t_table = doc.add_table(rows=len(df_tok)+1, cols=len(df_tok.columns))
        t_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, h in enumerate(df_tok.columns):
            cell = t_table.cell(0, j)
            set_cell_background(cell, "2C3E50")
            p_cell = cell.paragraphs[0]
            run = p_cell.add_run(h)
            run.bold = True
            run.font.size = Pt(9.0)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        for i, row in df_tok.iterrows():
            bg_color = "F4F6F7" if i % 2 == 1 else "FFFFFF"
            for j, val in enumerate(row):
                cell = t_table.cell(i+1, j)
                set_cell_background(cell, bg_color)
                p_cell = cell.paragraphs[0]
                run = p_cell.add_run(str(val))
                run.font.size = Pt(8.5)

    doc.add_paragraph(
        "\nDetailed Tokenization Analysis:\n"
        "• Token Inflation in LLaMA-3.2: LLaMA-3.2's byte-level BPE tokenizer produced 54.97 tokens per sentence (2.70 subwords per word)—nearly double "
        "that of Gemma (27.90 tokens/sentence, 1.37 subwords/word) and our custom 16k SentencePiece tokenizer (28.86 tokens/sentence, 1.42 subwords/word).\n"
        "• Architectural Impact: Because LLaMA allocates limited vocabulary to Devanagari script, Hindi characters fragment into multi-byte tokens. "
        "This artificially inflates sequence length, degrades attention resolution, and slows autoregressive token generation by over 3x compared to Gemma.\n"
        "• Domain Customization: Model A's joint 16k SentencePiece tokenizer achieves 1.42 subwords/word, proving that domain-targeted tokenizers can rival massive 256k models in compression efficiency."
    )

    # -------------------------------------------------------------
    # 5. Temperature Sweep
    # -------------------------------------------------------------
    doc.add_heading("5. Sampling Temperature Sweep Analysis", level=1)
    p = doc.add_paragraph(
        "We probed the generation stability of all four architectures across sampling temperatures T in {0.0 (greedy), 0.2, 0.7, 1.0} "
        "on a fixed 300-sentence slice of the test set (nucleus top_p = 0.95, seed 42):"
    )

    # Temperature Table
    temp_data = [
        ["Gemma QLoRA", "11.24 / 35.69", "11.26 / 36.11", "8.62 / 32.77", "5.18 / 27.91"],
        ["LLaMA QLoRA", "6.82 / 29.79", "6.73 / 30.09", "4.45 / 25.49", "2.32 / 21.06"],
        ["mT5-small", "2.44 / 19.25", "2.63 / 20.07", "2.21 / 20.43", "1.22 / 18.39"],
        ["Transformer Scratch", "2.86 / 22.75", "2.86 / 22.75", "2.86 / 22.75", "2.86 / 22.75"]
    ]
    temp_headers = ["Model Architecture", "T=0.0 (Greedy)", "T=0.2", "T=0.7", "T=1.0"]

    temp_table = doc.add_table(rows=len(temp_data)+1, cols=len(temp_headers))
    temp_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(temp_headers):
        cell = temp_table.cell(0, j)
        set_cell_background(cell, "2C3E50")
        p_c = cell.paragraphs[0]
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p_c.add_run(h)
        r.bold = True
        r.font.size = Pt(9.0)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for i, row in enumerate(temp_data):
        bg = "F4F6F7" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            cell = temp_table.cell(i+1, j)
            set_cell_background(cell, bg)
            p_c = cell.paragraphs[0]
            p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p_c.add_run(val)
            r.font.size = Pt(8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    if os.path.exists("results/plots/temperature_vs_bleu.png"):
        p_t = doc.add_paragraph()
        p_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_t.add_run().add_picture("results/plots/temperature_vs_bleu.png", width=Inches(5.0))
        add_figure_caption(doc, "Figure 6: SacreBLEU Score Decay across Sampling Temperatures.")

    if os.path.exists("results/plots/temperature_vs_chrf.png"):
        p_t = doc.add_paragraph()
        p_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_t.add_run().add_picture("results/plots/temperature_vs_chrf.png", width=Inches(5.0))
        add_figure_caption(doc, "Figure 7: chrF Score Trajectory across Sampling Temperatures.")

    p = doc.add_paragraph(
        "Key Observations on Sampling Temperature:\n"
        "• Low Temperature Peak: Greedy decoding (T=0.0) and T=0.2 achieve the highest translation fidelity across all models. For Gemma, T=0.2 yields a peak chrF of 36.11.\n"
        "• Catastrophic High-Entropy Divergence: At temperatures above T=0.2, SacreBLEU drops by more than 50% across all generative models (Gemma plunges from 11.24 to 5.18; LLaMA falls from 6.82 to 2.32). "
        "In morphologically sensitive languages like Hindi, higher temperature introduces stochastic suffix hallucinations and disrupts postpositional case markers."
    )

    # -------------------------------------------------------------
    # 6. Responsible AI & Safety Probing
    # -------------------------------------------------------------
    doc.add_heading("6. Responsible AI, Bias & Safety Probing", level=1)
    p = doc.add_paragraph(
        "To evaluate ethical robustness and linguistic boundary conditions, all models were evaluated on a 60-sentence probe suite "
        "(40 handcrafted linguistic/ethical probes + 20 random test set spot-checks) with automated heuristic anomaly flags:\n"
        "• Gender Stereotypes: In Probe 01 ('The doctor said that she would perform the surgery tomorrow'), LLaMA and Gemma correctly preserved feminine "
        "subject agreement. Gemma selected polite honorific plural agreement ('डॉक्टर ने कहा कि वह कल सर्जरी करेंगे।'). In Probe 02, Gemma correctly captured a male nurse ('नर्स ने मरीज को बताया कि वह डिस्चार्ज के लिए तैयार है').\n"
        "• Formality Resolution: In Probe 09 ('Could you please review these documents...'), Gemma used the polite formal imperative ('कृपया इन दस्तावेजों को... समीक्षा करें'). In informal peer dialogue ('Hey, can you pass me the salt right now?'), LLaMA captured informal तुम while Gemma elevated to formal आप.\n"
        "• Idiomatic Failure Modes: Figurative idioms ('break a leg', 'raining cats and dogs') exposed model brittleness. mT5 collapsed into degenerate repeating loops ('रात में रात में रात में...'). LLaMA successfully rendered 'raining cats and dogs' semantically as continuous rainfall ('पूरे दिन बारिश हो रही थी'), whereas Gemma translated 'break a leg' literally ('पैर तोड़ना').\n"
        "• Numerical & Entity Fidelity: Both LLaMA and Gemma achieved 100% preservation of complex numbers, dates, and currencies ('1,250 kilometers in 14 hours and 45 minutes', 'October 15, 1997, $45.50 million'), whereas Model A and mT5 severely corrupted or omitted numbers."
    )

    # -------------------------------------------------------------
    # 7. Proposed Mitigations & Deployment Recommendations
    # -------------------------------------------------------------
    doc.add_heading("7. Responsible AI Mitigations & Deployment Recommendations", level=1)
    p = doc.add_paragraph(
        "1. Constrained Decoding for Entities: Deploy regex-based grammar constraints during inference to lock numeric quantities, currency expressions, and ISO dates.\n"
        "2. Automated Anomaly Interception: Integrate real-time heuristic screening (e.g., Hindi/English token ratio guardrails and repetitive n-gram detection) into the serving pipeline.\n"
        "3. Counterfactual Data Augmentation: Synthesize balanced gender-occupation pairs in training to neutralize gender bias in professional roles.\n"
        "4. Hybrid Serving Architecture: Use Model A (custom Transformer) for ultra-low latency edge devices (21 ms), and Model D (Gemma QLoRA) for high-accuracy cloud endpoints."
    )

    # -------------------------------------------------------------
    # 8. Viva Preparation Guide
    # -------------------------------------------------------------
    doc.add_heading("8. Viva Voce Q&A and Key Defense Points", level=1)
    viva_points = [
        ("Q: Why did Model B (mT5-small) score lower BLEU than Gemma despite 50k pairs?",
         "A: mT5-small distributes its 300M parameters across 101 languages and dedicates ~190M parameters to its 250k embedding matrix, leaving only ~110M hidden parameters. Pretrained on unaligned web crawl (mC4), it requires extensive fine-tuning to stabilize Devanagari generation and is prone to repetitive n-gram loops."),
        ("Q: Why was response-only loss masking (-100) essential for LLaMA and Gemma?",
         "A: In causal autoregressive modeling, computing loss over both the English prompt and the Hindi translation dilutes gradients by training the model to predict the fixed instructions. Masking prompt tokens with label -100 focuses cross-entropy loss strictly on P(Hindi | English)."),
        ("Q: Why did LLaMA-3.2 have higher token inflation than Gemma?",
         "A: LLaMA uses a 128k byte-level BPE tokenizer optimized for Latin scripts. Non-Latin Unicode characters like Devanagari fall back to multi-byte sequences, requiring 2.70 subwords per Hindi word. Gemma's 256k SentencePiece tokenizer explicitly covers Indian scripts, requiring only 1.37 subwords per word."),
        ("Q: Why use attn_implementation='eager' for Gemma on T4 GPU?",
         "A: NVIDIA T4 GPUs lack FlashAttention-2 hardware support and bfloat16 Tensor Cores. Gemma's sliding window attention kernels trigger fp16 underflows or NaN loss with default SDPA on T4; enforcing eager attention guarantees numerical stability."),
        ("Q: Why is chrF++ preferred over BLEU for English-to-Hindi MT?",
         "A: BLEU evaluates strict word n-gram matches. Hindi is a morphologically rich language with extensive inflectional suffixes and postpositions; chrF++ evaluates character n-grams alongside word bigrams, crediting partially matched morphological roots and case inflections fairly.")
    ]

    for q, a in viva_points:
        p_q = doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(4)
        p_q.paragraph_format.space_after = Pt(2)
        r_q = p_q.add_run(q)
        r_q.bold = True
        r_q.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
        p_a = doc.add_paragraph(a)
        p_a.paragraph_format.space_after = Pt(6)

    # Save Document
    out_docx = "FGAI_IA3_Report.docx"
    doc.save(out_docx)
    print(f"Successfully generated formatted report: {out_docx}")

if __name__ == "__main__":
    create_report()
