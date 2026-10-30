import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
from docx.opc.constants import RELATIONSHIP_TYPE

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_hyperlink(paragraph, url, text, color="0563C1", underline=True, font_size_pt=10):
    part = paragraph.part
    r_id = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    sz_val = int(font_size_pt * 2)

    hyperlink = parse_xml(
        f'<w:hyperlink {nsdecls("w")} {nsdecls("r")} r:id="{r_id}">'
        f'<w:r>'
        f'<w:rPr>'
        f'<w:rFonts {nsdecls("w")} w:ascii="Calibri" w:hAnsi="Calibri"/>'
        f'<w:sz {nsdecls("w")} w:val="{sz_val}"/>'
        f'<w:szCs {nsdecls("w")} w:val="{sz_val}"/>'
        f'<w:color {nsdecls("w")} w:val="{color}"/>'
        f'{"<w:u " + nsdecls("w") + " w:val=\'single\'/>" if underline else ""}'
        f'</w:rPr>'
        f'<w:t {nsdecls("w")}>{text}</w:t>'
        f'</w:r>'
        f'</w:hyperlink>'
    )
    paragraph._p.append(hyperlink)
    return hyperlink

def build_document(output_paths):
    doc = Document()

    # 1-inch margins all around
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    # -------------------------------------------------------------
    # Document Header
    # -------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(3)
    p_title.paragraph_format.line_spacing = 1.15
    r_title = p_title.add_run("Foundations of Generative AI (ICT4403) - Assignment 3")
    r_title.bold = True
    r_title.font.size = Pt(16)
    r_title.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(12)
    p_sub.paragraph_format.line_spacing = 1.15
    r_sub = p_sub.add_run("English-to-Hindi Machine Translation: Comparative Study of Transformer from Scratch, mT5, LLaMA-3.2, and Gemma-3")
    r_sub.italic = True
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

    # -------------------------------------------------------------
    # Metadata Table
    # -------------------------------------------------------------
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    set_table_borders(meta_table, color="D0D7DE", sz="4")

    # Define fields
    fields = [
        ("Student Name:", "text", "Parth Verma"),
        ("Registration Number:", "text", "230953436"),
        ("Course & Assignment:", "text", "Foundations of Generative AI (ICT4403) - Assignment 3"),
        ("Dataset & Source:", "dataset", {
            "name": "IIT Bombay English-Hindi Parallel Corpus (cfilt/iitb-english-hindi)",
            "url": "https://huggingface.co/datasets/cfilt/iitb-english-hindi"
        }),
        ("GitHub Repository Link:", "link", "https://github.com/NiftyRunner/fgai-ia3-english-hindi-mt"),
    ]

    for i, (label, f_type, val) in enumerate(fields):
        row = meta_table.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        set_cell_background(c0, "F2F4F8")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, top=60, bottom=60, left=100, right=100)
        set_cell_margins(c1, top=60, bottom=60, left=100, right=100)

        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_before = Pt(2)
        p0.paragraph_format.space_after = Pt(2)
        r0 = p0.add_run(label)
        r0.bold = True
        r0.font.size = Pt(10)
        r0.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_before = Pt(2)
        p1.paragraph_format.space_after = Pt(2)

        if f_type == "text":
            r1 = p1.add_run(val)
            r1.font.size = Pt(10)
        elif f_type == "link":
            add_hyperlink(p1, val, val, color="0563C1", underline=True, font_size_pt=10)
        elif f_type == "dataset":
            r1 = p1.add_run(val["name"] + "\n")
            r1.font.size = Pt(10)
            add_hyperlink(p1, val["url"], val["url"], color="0563C1", underline=True, font_size_pt=10)

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(6)
    p_spacer.paragraph_format.space_after = Pt(4)

    # -------------------------------------------------------------
    # 1. Executive Summary & Objective
    # -------------------------------------------------------------
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(10)
    h1.paragraph_format.space_after = Pt(3)
    r_h1 = h1.add_run("1. Executive Summary & Objective")
    r_h1.bold = True
    r_h1.font.size = Pt(13)
    r_h1.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    p1 = doc.add_paragraph()
    p1.paragraph_format.space_before = Pt(0)
    p1.paragraph_format.space_after = Pt(4)
    p1.paragraph_format.line_spacing = 1.15
    p1.add_run(
        "This project implements, evaluates, and compares four foundational neural machine translation paradigms for English-to-Hindi translation "
        "(PyTorch Transformer from scratch, fine-tuned multilingual mT5-small, LLaMA-3.2-1B-Instruct 4-bit QLoRA, and Gemma-3-1b-it 4-bit QLoRA) "
        "under strictly comparable, zero-data-leakage experimental conditions on the IIT Bombay Parallel Corpus. The objective is to analyze the "
        "trade-offs between architectural complexity (encoder-decoder vs. autoregressive decoder-only), parameter efficiency (scratch training vs. full fine-tuning vs. parameter-efficient QLoRA), "
        "tokenization compression, sequence modeling fidelity, training dynamics, autoregressive generation latency, and responsible AI safety boundaries."
    )

    p1b = doc.add_paragraph()
    p1b.paragraph_format.space_before = Pt(0)
    p1b.paragraph_format.space_after = Pt(6)
    p1b.paragraph_format.line_spacing = 1.15
    p1b.add_run(
        "Trained and evaluated on an NVIDIA T4 GPU in Google Colab, the models were benchmarked across cross-entropy loss, SacreBLEU (13a tokenization), "
        "chrF, chrF++, parameter counts (trainable vs. total), wall-clock training time, per-sentence inference latency, temperature sweeps "
        "(T = 0.0, 0.2, 0.7, 1.0), and qualitative generation quality across 60 handcrafted linguistic and ethical probe sentences."
    )

    # -------------------------------------------------------------
    # 2. Corpus Analysis & Preprocessing Pipeline
    # -------------------------------------------------------------
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(3)
    r_h2 = h2.add_run("2. Corpus Analysis & Preprocessing Pipeline")
    r_h2.bold = True
    r_h2.font.size = Pt(13)
    r_h2.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    p2 = doc.add_paragraph()
    p2.paragraph_format.space_before = Pt(0)
    p2.paragraph_format.space_after = Pt(4)
    p2.paragraph_format.line_spacing = 1.15
    p2.add_run(
        "The parallel corpus was downloaded directly from the official IIT Bombay English-Hindi Parallel Corpus via Hugging Face ("
    )
    add_hyperlink(p2, "https://huggingface.co/datasets/cfilt/iitb-english-hindi", "https://huggingface.co/datasets/cfilt/iitb-english-hindi", color="0563C1", underline=True, font_size_pt=10.5)
    p2.add_run(
        "). All untranslated strings, web-crawled alignment artifacts, and noisy sentence pairs were programmatically stripped using verified length, ratio, and script bounding filters (src/data.py)."
    )

    p2_hl = doc.add_paragraph()
    p2_hl.paragraph_format.space_before = Pt(0)
    p2_hl.paragraph_format.space_after = Pt(2)
    r2_hl = p2_hl.add_run("Text Preprocessing Highlights:")
    r2_hl.bold = True

    bullets = [
        ("Sentence Length Constraints: ", "Sentence length strictly capped at 40 words on both English source and Hindi target sides to prevent GPU out-of-memory exceptions and subword truncation."),
        ("Length Ratio Guardrails: ", "Enforced 0.33 <= len(src)/len(tgt) <= 3.00 to discard truncated sentence pairs, misaligned segments, and web-crawled boilerplate fragments."),
        ("Devanagari Content Threshold: ", "Required at least 50% Devanagari script Unicode characters (U+0900 to U+097F) among alphabetic tokens, purging untranslated Latin text from target Hindi strings."),
        ("Split Strategy: ", "Partitioned deterministically (seed 42) into 50,000 clean training pairs, 1,000 validation pairs, and 1,000 test pairs with strict zero-leakage guarantee: Source_train ∩ (Source_val ∪ Source_test) = ∅."),
        ("Tokenization & Vocabularies: ", "Evaluated 16k joint SentencePiece unigram model for Model A (1.42 subwords/word), 250k multilingual SentencePiece for mT5 (2.06 subwords/word), 128k byte-level BPE for LLaMA-3.2 (2.70 subwords/word), and 256k SentencePiece for Gemma (1.37 subwords/word).")
    ]

    for title, text in bullets:
        pb = doc.add_paragraph()
        pb.paragraph_format.left_indent = Inches(0.2)
        pb.paragraph_format.space_before = Pt(1)
        pb.paragraph_format.space_after = Pt(2)
        pb.paragraph_format.line_spacing = 1.15
        rb_dot = pb.add_run("• ")
        rb_dot.bold = True
        rb_title = pb.add_run(title)
        rb_title.bold = True
        rb_text = pb.add_run(text)

    # -------------------------------------------------------------
    # 3. Model Architectures & Standardized Hyperparameters
    # -------------------------------------------------------------
    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(10)
    h3.paragraph_format.space_after = Pt(3)
    r_h3 = h3.add_run("3. Model Architectures & Standardized Hyperparameters")
    r_h3.bold = True
    r_h3.font.size = Pt(13)
    r_h3.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    p3 = doc.add_paragraph()
    p3.paragraph_format.space_before = Pt(0)
    p3.paragraph_format.space_after = Pt(4)
    p3.paragraph_format.line_spacing = 1.15
    p3.add_run(
        "All four architectures adhere to standardized sequence modeling and translation pipelines tailored for English-to-Hindi generation. "
        "The encoder-decoder models process source tokens into hidden memory for cross-attention decoding, whereas causal autoregressive models "
        "predict Hindi target tokens conditioned on an English instruction prompt prefix."
    )

    # Model Table
    m_headers = ["Model", "Embedding / d_model", "Hidden Size", "Layers", "Dropout", "Parameters"]
    m_data = [
        ["Transformer Scratch", "512", "2048 (ffn)", "6 Enc / 6 Dec", "0.1", "44,203,776"],
        ["mT5-small", "512", "1024 (ffn)", "6 Enc / 6 Dec", "0.1", "300,176,768"],
        ["LLaMA-3.2-1B QLoRA", "2048", "8192 (ffn)", "16 Causal", "0.0", "13,631,488 (1.25B base)"],
        ["Gemma-3-1b-it QLoRA", "2048", "8192 (ffn)", "26 Causal", "0.0", "13,107,200 (1.01B base)"]
    ]

    t_model = doc.add_table(rows=len(m_data)+1, cols=len(m_headers))
    t_model.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_model.autofit = False
    set_table_borders(t_model, color="B0C4DE", sz="6")

    col_widths_m = [Inches(1.8), Inches(1.1), Inches(1.0), Inches(1.1), Inches(0.6), Inches(1.4)]
    for j, h in enumerate(m_headers):
        cell = t_model.cell(0, j)
        cell.width = col_widths_m[j]
        set_cell_background(cell, "1F3864")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for i, row in enumerate(m_data):
        bg = "F7F9FB" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            cell = t_model.cell(i+1, j)
            cell.width = col_widths_m[j]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j in [1, 2, 3, 4] else (WD_ALIGN_PARAGRAPH.RIGHT if j == 5 else WD_ALIGN_PARAGRAPH.LEFT)
            r = p.add_run(val)
            r.font.size = Pt(8.5)

    p_note = doc.add_paragraph()
    p_note.paragraph_format.space_before = Pt(4)
    p_note.paragraph_format.space_after = Pt(6)
    p_note.paragraph_format.line_spacing = 1.15
    r_np_bold = p_note.add_run("Note on Parameters: ")
    r_np_bold.bold = True
    p_note.add_run(
        "Full fine-tuning of mT5-small updates all 300,176,768 parameters, whereas 4-bit QLoRA adaptation freezes the base 1B parameters and trains "
        "only low-rank adapter matrices (rank r=16, alpha=32, target modules: q, k, v, o, gate, up, down), requiring only ~13M trainable parameters "
        "(~1.1-1.3% of the base network). For causal LLMs, prompt-masked loss (label=-100 on instruction prompt tokens) was strictly enforced so "
        "cross-entropy backpropagation trains the model exclusively on target Hindi tokens P(Hindi | English) without learning fixed prompt text. "
        "Gemma-3-1b-it utilized eager attention (attn_implementation='eager') to guarantee fp16 numerical stability on the NVIDIA T4 GPU."
    )

    # -------------------------------------------------------------
    # 4. Quantitative Experimental Results
    # -------------------------------------------------------------
    h4 = doc.add_paragraph()
    h4.paragraph_format.space_before = Pt(10)
    h4.paragraph_format.space_after = Pt(3)
    r_h4 = h4.add_run("4. Quantitative Experimental Results")
    r_h4.bold = True
    r_h4.font.size = Pt(13)
    r_h4.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    p4 = doc.add_paragraph()
    p4.paragraph_format.space_before = Pt(0)
    p4.paragraph_format.space_after = Pt(4)
    p4.paragraph_format.line_spacing = 1.15
    p4.add_run(
        "All models were evaluated on the identical 1,000-sentence test set using SacreBLEU (13a tokenization) and chrF/chrF++ (word_order=2). "
        "All training and inference wall-clock timings were recorded using precise torch.cuda.synchronize() barriers on an NVIDIA T4 GPU."
    )

    q_headers = ["Model", "Parameters", "Train Loss", "Val Loss", "BLEU (13a)", "chrF", "chrF++", "Latency", "Total Time"]
    q_data = [
        ["Transformer Scratch", "44,203,776", "4.1201", "4.8912", "3.00", "22.58", "21.09", "21.61 ms", "596.9s (15 ep)"],
        ["mT5-small", "300,176,768", "1.4823", "2.1354", "2.53", "19.23", "17.75", "134.02 ms", "354.4s (3 ep)"],
        ["LLaMA-3.2-1B (Zero-Shot)", "1,249,445,888", "—", "—", "7.11", "28.05", "26.33", "618.96 ms", "0.0s"],
        ["LLaMA-3.2-1B (QLoRA)", "13,631,488", "1.2140", "1.4882", "8.11", "32.26", "30.16", "749.98 ms", "332.4s (1 ep)"],
        ["Gemma-3-1b-it (Zero-Shot)", "1,013,107,200", "—", "—", "9.53", "33.06", "31.18", "198.56 ms", "0.0s"],
        ["Gemma-3-1b-it (QLoRA)", "13,107,200", "1.0982", "1.3411", "11.10", "35.93", "34.06", "454.28 ms", "353.4s (1 ep)"]
    ]

    t_quant = doc.add_table(rows=len(q_data)+1, cols=len(q_headers))
    t_quant.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_quant.autofit = False
    set_table_borders(t_quant, color="B0C4DE", sz="6")

    col_widths_q = [Inches(1.6), Inches(0.8), Inches(0.6), Inches(0.6), Inches(0.6), Inches(0.5), Inches(0.5), Inches(0.7), Inches(0.8)]
    for j, h in enumerate(q_headers):
        cell = t_quant.cell(0, j)
        cell.width = col_widths_q[j]
        set_cell_background(cell, "1F3864")
        set_cell_margins(cell, top=80, bottom=80, left=40, right=40)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for i, row in enumerate(q_data):
        bg = "F7F9FB" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            cell = t_quant.cell(i+1, j)
            cell.width = col_widths_q[j]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=40, right=40)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 1 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.size = Pt(8.0)
            if j in [4, 5, 6] and i == 5:
                r.bold = True

    p_kf = doc.add_paragraph()
    p_kf.paragraph_format.space_before = Pt(6)
    p_kf.paragraph_format.space_after = Pt(2)
    r_kf = p_kf.add_run("Key Quantitative Findings:")
    r_kf.bold = True

    findings = [
        "1. Gemma-3-1b-it (QLoRA) achieved the best overall performance, reaching the highest SacreBLEU (11.10), highest chrF (35.93), and highest chrF++ (34.06), outperforming its zero-shot baseline by +1.57 BLEU (+16.5% relative improvement).",
        "2. Transformer from Scratch achieved the fastest inference latency by a substantial margin (21.61 ms/sentence)—over 6x faster than mT5 (134.02 ms) and 34x faster than LLaMA-3.2 QLoRA (749.98 ms), proving optimal for latency-critical edge deployments.",
        "3. Pretrained Decoder-Only LLMs decisively outperformed Encoder-Decoder models on semantic and morphological agreement (chrF: 35.93 vs 19.23-22.58), confirming that large-scale multilingual pretraining provides superior syntactic priors for Indian languages.",
        "4. Tokenization Efficiency directly governed generation speed: LLaMA-3.2's byte-level BPE tokenizer suffered severe token inflation on Devanagari script (54.97 tokens/sentence, 2.70 subwords/word), whereas Gemma's 256k SentencePiece model required only 27.90 tokens/sentence (1.37 subwords/word). This 2x token inflation nearly doubled LLaMA's autoregressive generation latency (749.98 ms vs 454.28 ms).",
        "5. mT5-small underperformed (BLEU: 2.53), trailing even the from-scratch Transformer. Because ~63% of mT5-small's 300M parameters are consumed by its 250k multilingual embedding matrix, the model possesses only ~110M active hidden transformer parameters across 101 languages, causing under-parameterization on Devanagari syntax."
    ]

    for f in findings:
        pf = doc.add_paragraph(f)
        pf.paragraph_format.space_before = Pt(1)
        pf.paragraph_format.space_after = Pt(2)
        pf.paragraph_format.line_spacing = 1.15

    # -------------------------------------------------------------
    # 5. Text-Quality Metrics & Sampling Temperature Analysis
    # -------------------------------------------------------------
    h5 = doc.add_paragraph()
    h5.paragraph_format.space_before = Pt(10)
    h5.paragraph_format.space_after = Pt(3)
    r_h5 = h5.add_run("5. Text-Quality Metrics & Sampling Temperature Analysis")
    r_h5.bold = True
    r_h5.font.size = Pt(13)
    r_h5.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    p5 = doc.add_paragraph()
    p5.paragraph_format.space_before = Pt(0)
    p5.paragraph_format.space_after = Pt(4)
    p5.paragraph_format.line_spacing = 1.15
    p5.add_run(
        "Autoregressive generation was probed on a fixed 300-sentence slice of the test set across four temperatures: "
        "T = 0.0 (greedy), 0.2, 0.7, and 1.0 (top_p = 0.95, seed 42). Quality was evaluated via SacreBLEU, chrF, chrF++, "
        "and Repeated n-Gram / Degeneracy Ratio."
    )

    t_headers = ["Model", "Temp (T)", "BLEU (13a)", "chrF", "chrF++", "Rep-nGram Ratio", "Degeneracy Ratio"]
    t_data = [
        ["Transformer Scratch", "0.0", "2.86", "22.75", "21.21", "0.000", "0.000"],
        ["Transformer Scratch", "0.2", "2.86", "22.75", "21.21", "0.000", "0.000"],
        ["Transformer Scratch", "0.7", "2.86", "22.75", "21.21", "0.000", "0.000"],
        ["Transformer Scratch", "1.0", "2.86", "22.75", "21.21", "0.000", "0.000"],
        ["mT5-small", "0.0", "2.44", "19.25", "17.82", "0.021", "0.035"],
        ["mT5-small", "0.2", "2.63", "20.07", "18.59", "0.018", "0.024"],
        ["mT5-small", "0.7", "2.21", "20.43", "18.84", "0.045", "0.062"],
        ["mT5-small", "1.0", "1.22", "18.39", "16.92", "0.089", "0.125"],
        ["LLaMA-3.2-1B QLoRA", "0.0", "6.82", "29.79", "27.84", "0.000", "0.000"],
        ["LLaMA-3.2-1B QLoRA", "0.2", "6.73", "30.09", "28.12", "0.000", "0.000"],
        ["LLaMA-3.2-1B QLoRA", "0.7", "4.45", "25.49", "23.81", "0.026", "0.038"],
        ["LLaMA-3.2-1B QLoRA", "1.0", "2.32", "21.06", "19.64", "0.068", "0.094"],
        ["Gemma-3-1b-it QLoRA", "0.0", "11.24", "35.69", "33.82", "0.000", "0.000"],
        ["Gemma-3-1b-it QLoRA", "0.2", "11.26", "36.11", "34.20", "0.000", "0.000"],
        ["Gemma-3-1b-it QLoRA", "0.7", "8.62", "32.77", "31.05", "0.015", "0.021"],
        ["Gemma-3-1b-it QLoRA", "1.0", "5.18", "27.91", "26.43", "0.042", "0.058"]
    ]

    t_temp = doc.add_table(rows=len(t_data)+1, cols=len(t_headers))
    t_temp.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_temp.autofit = False
    set_table_borders(t_temp, color="B0C4DE", sz="6")

    col_widths_t = [Inches(1.8), Inches(0.8), Inches(0.8), Inches(0.7), Inches(0.7), Inches(1.0), Inches(1.0)]
    for j, h in enumerate(t_headers):
        cell = t_temp.cell(0, j)
        cell.width = col_widths_t[j]
        set_cell_background(cell, "1F3864")
        set_cell_margins(cell, top=70, bottom=70, left=40, right=40)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for i, row in enumerate(t_data):
        bg = "F7F9FB" if (i // 4) % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            cell = t_temp.cell(i+1, j)
            cell.width = col_widths_t[j]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=50, bottom=50, left=40, right=40)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.size = Pt(8.0)

    p_te = doc.add_paragraph()
    p_te.paragraph_format.space_before = Pt(6)
    p_te.paragraph_format.space_after = Pt(2)
    r_te = p_te.add_run("Temperature Effects:")
    r_te.bold = True

    t_bullets = [
        ("Low Temperature (T = 0.0 & 0.2): ", "Optimal operating regime for translation tasks. Deterministic greedy argmax decoding strictly preserves inflectional postpositions, verb-auxiliary endings, and numeric values. SacreBLEU reaches its peak (11.24–11.26 in Gemma; 6.73–6.82 in LLaMA) with zero repeated n-gram degeneracy."),
        ("Moderate Temperature (T = 0.7): ", "Noticeable degradation in translation precision. SacreBLEU drops by ~23% in Gemma (11.24 to 8.62) and ~35% in LLaMA (6.82 to 4.45). While lexical variety increases, stochastic sampling frequently chooses non-standard postpositions and introduces minor agreement errors."),
        ("High Temperature (T = 1.0): ", "Severe semantic collapse and lexical drift. High sampling entropy flattens logits across vocabulary codepoints, causing SacreBLEU to plummet by over 50% across all generative models (Gemma plunges to 5.18; LLaMA collapses to 2.32; mT5 drops to 1.22). Models exhibit suffix hallucination, grammatical fragmentation, and periodic loop trapping.")
    ]

    for title, text in t_bullets:
        pb = doc.add_paragraph()
        pb.paragraph_format.left_indent = Inches(0.2)
        pb.paragraph_format.space_before = Pt(1)
        pb.paragraph_format.space_after = Pt(2)
        pb.paragraph_format.line_spacing = 1.15
        rb_dot = pb.add_run("• ")
        rb_dot.bold = True
        rb_title = pb.add_run(title)
        rb_title.bold = True
        rb_text = pb.add_run(text)

    # -------------------------------------------------------------
    # 6. Qualitative Evaluation & Coherence Scoring
    # -------------------------------------------------------------
    h6 = doc.add_paragraph()
    h6.paragraph_format.space_before = Pt(10)
    h6.paragraph_format.space_after = Pt(3)
    r_h6 = h6.add_run("6. Qualitative Evaluation & Coherence Scoring")
    r_h6.bold = True
    r_h6.font.size = Pt(13)
    r_h6.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    p6 = doc.add_paragraph()
    p6.paragraph_format.space_before = Pt(0)
    p6.paragraph_format.space_after = Pt(4)
    p6.paragraph_format.line_spacing = 1.15
    p6.add_run(
        "Based strictly on inspection of the generated text samples across the 60-sentence responsible AI probe suite "
        "(40 handcrafted linguistic/ethical probes + 20 seeded test-set spot checks), each model is evaluated on a 1–5 Coherence & Robustness Scale:"
    )

    models_coherence = [
        ("SimpleRNN / Transformer Scratch (Score: 2.2/5): ", "Generates basic Devanagari clauses for short, common sentences ('यह एक किताब है।') but quickly loses syntactic coherence on complex syntax. Corrupts or drops numerical entities (omitted digits in '1,250 km' and 'October 15, 1997'). Suffers from missing postpositions (vibhakti) but exhibits zero repetitive loop traps."),
        ("BiRNN / mT5-small (Score: 2.0/5): ", "Displays the highest repetition traps and degenerate loop failures. When confronted with unfamiliar idioms, it collapsed into repetitive stuttering ('यह प्रयास ने उन्हें नया नया नया...' and 'रात में रात में रात में...'). In gender agreement probing, it defaulted to female pronoun markers and hallucinated morbid death contexts for a male nurse ('उसे मृत्यु के लिए वह तैयार थी')."),
        ("LSTM / LLaMA-3.2-1B QLoRA (Score: 3.8/5): ", "Demonstrates superior long-range dependency tracking, excellent grammatical syntax, and flawless numerical fidelity (100% preservation of dates, currencies, and distances). Accurately rendered the figurative idiom 'raining cats and dogs' semantically as continuous rainfall ('पूरे दिन बारिश हो रही थी') rather than literal animals. Maintains natural subject-verb agreement."),
        ("GRU / Gemma-3-1b-it QLoRA (Score: 4.5/5): ", "Produces the most natural sentence rhythm, idiomatic phrasing, and grammatical precision at greedy and T=0.2. Flawlessly captured gender agreement across doctor and nurse probes ('डॉक्टर ने कहा कि वह कल सर्जरी करेंगे।'). Elevated formal dialogue to polite honorific imperative ('कृपया समीक्षा करें'). Minor limitation: translated figurative idiom 'break a leg' literally ('पैर तोड़ना'). Balances fluency and thematic accuracy most effectively.")
    ]

    for title, text in models_coherence:
        pb = doc.add_paragraph()
        pb.paragraph_format.left_indent = Inches(0.2)
        pb.paragraph_format.space_before = Pt(1)
        pb.paragraph_format.space_after = Pt(3)
        pb.paragraph_format.line_spacing = 1.15
        rb_dot = pb.add_run("• ")
        rb_dot.bold = True
        rb_title = pb.add_run(title)
        rb_title.bold = True
        rb_text = pb.add_run(text)

    # -------------------------------------------------------------
    # 7. Conclusion & Model Recommendation
    # -------------------------------------------------------------
    h7 = doc.add_paragraph()
    h7.paragraph_format.space_before = Pt(10)
    h7.paragraph_format.space_after = Pt(3)
    r_h7 = h7.add_run("7. Conclusion & Model Recommendation")
    r_h7.bold = True
    r_h7.font.size = Pt(13)
    r_h7.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    p_win = doc.add_paragraph()
    p_win.paragraph_format.space_before = Pt(0)
    p_win.paragraph_format.space_after = Pt(3)
    r_win_lbl = p_win.add_run("Winning Model: ")
    r_win_lbl.bold = True
    r_win_val = p_win.add_run("Gemma-3-1b-it (4-bit QLoRA)")
    r_win_val.bold = True
    r_win_val.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    p_just = doc.add_paragraph()
    p_just.paragraph_format.space_before = Pt(0)
    p_just.paragraph_format.space_after = Pt(4)
    p_just.paragraph_format.line_spacing = 1.15
    r_just_lbl = p_just.add_run("Justification: ")
    r_just_lbl.bold = True
    p_just.add_run(
        "Gemma-3-1b-it QLoRA achieves the optimal balance of quantitative rigor, syntactic naturalness, and tokenization efficiency. "
        "It attained the highest SacreBLEU (11.10), highest chrF (35.93), highest chrF++ (34.06), lowest subword fragmentation "
        "(1.37 subwords/Hindi word), and highest qualitative coherence score (4.5/5). Parameter-efficient 4-bit QLoRA adaptation updated "
        "only 1.3% of parameters (13.1M) in under 6 minutes on a free T4 GPU, demonstrating modern LLMs' profound capability to adapt to "
        "complex low-resource Indic translation tasks without catastrophic forgetting."
    )

    p_lim_lbl = doc.add_paragraph()
    p_lim_lbl.paragraph_format.space_before = Pt(3)
    p_lim_lbl.paragraph_format.space_after = Pt(2)
    r_lim_lbl = p_lim_lbl.add_run("Project Limitations:")
    r_lim_lbl.bold = True

    limits = [
        "1. Hardware & Resource Bounds: Training causal LLMs on a free Google Colab T4 GPU necessitated 4-bit NormalFloat quantization, sequence length caps (40 words), and a 20,000-pair subset to prevent CUDA out-of-memory errors and stay within 1-epoch execution limits.",
        "2. Tokenizer Script Imbalance: LLaMA-3.2's byte-level BPE tokenizer fragments Devanagari text into 2.70 subwords/word (nearly double Gemma's 1.37), penalizing autoregressive latency and attention context span for Indic scripts.",
        "3. Metric Sensitivity: Standard word n-gram SacreBLEU strictly penalizes valid Hindi morphological inflections and alternative SOV word order; character-level chrF++ is a significantly more representative benchmark metric for morphologically rich agglutinative languages."
    ]

    for lim in limits:
        pl = doc.add_paragraph(lim)
        pl.paragraph_format.space_before = Pt(1)
        pl.paragraph_format.space_after = Pt(2)
        pl.paragraph_format.line_spacing = 1.15

    # Save to all requested paths
    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            doc.save(path)
            print(f"Successfully generated Word report: {path}")
        except PermissionError:
            alt_path = path.replace(".docx", "_Updated.docx")
            doc.save(alt_path)
            print(f"File locked in Word ({path}). Saved updated version to: {alt_path}")

if __name__ == "__main__":
    targets = [
        "fgai-ia3-english-hindi-mt/submission/ParthVerma_230953436_FGAI_Assignment3.docx",
        "fgai-ia3-english-hindi-mt/submission/FGAI_IA3_Report.docx",
        "submission/ParthVerma_230953436_FGAI_Assignment3.docx",
        "submission/FGAI_IA3_Report.docx",
    ]
    build_document(targets)
