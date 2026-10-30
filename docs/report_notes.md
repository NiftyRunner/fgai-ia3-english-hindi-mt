# Academic Report Notes & Viva Defense Preparation

This document provides theoretical background, architectural justifications, and viva defense Q&A for your project.

---

## 1. Architectural Justifications (Why These Choices?)

### A. Model A: Basic Transformer from Scratch
* **Why $d_{\text{model}}=256, 4 \text{ heads}, 3+3 \text{ layers}, d_{\text{ff}}=1024$?**
  Training a standard Transformer base ($d_{\text{model}}=512, 6+6 \text{ layers}$, ~65M parameters) from scratch on only 50,000 parallel sentence pairs leads to severe overfitting and poor generalization. A compact configuration (~10–12M parameters) matches the information capacity of a 50k dataset while training comfortably within ~35 minutes on a T4 GPU.
* **Why Joint SentencePiece (16,000 Vocab)?**
  Training a single joint BPE/Unigram model on both English and Hindi allows shared subword representations for numerals, named entities, punctuation, and loanwords, while bounding the embedding table size so it does not dominate the total parameter count.
* **Why Tied Embeddings?**
  Weight tying between the target input embeddings and the final output projection matrix reduces parameters significantly and regularizes the representation space.
* **Why `torch.amp.GradScaler('cuda')`?**
  Automatic mixed precision (fp16) cuts tensor memory by nearly half and leverages T4 Tensor Cores for matrix multiplications, speeding up training 2–3x without numerical instability.

---

### B. Model B: mT5-small Fine-Tuning
* **Why Pretrained Encoder-Decoder?**
  mT5 is pretrained on the mC4 multilingual corpus (101 languages) with a masked language modeling (span corruption) objective. The encoder learns rich multilingual bidirectional representations, while the decoder cross-attends to the encoded source.
* **CRITICAL VIVA POINT: Why `fp32` instead of `fp16` on T4?**
  The T5 architecture uses RMSNorm and does not apply additive biases in attention layers. Furthermore, mT5 has a massive vocabulary of 250,112 tokens. In `fp16`, the maximum representable exponent is $65,504$ and values below $2^{-14} \approx 6.1 \times 10^{-5}$ flush to zero. When computing softmax across 250k vocabulary logits, intermediate activations routinely underflow, producing zero gradients or NaN loss. While modern GPUs use `bfloat16` (which retains the dynamic range of `fp32`), the free T4 GPU lacks `bfloat16` hardware acceleration. Therefore, mT5 MUST be fine-tuned in `fp32` on T4.

---

### C. Models C & D: Decoder-Only LLMs (LLaMA-3.2-1B and Gemma)
* **How Does a Decoder-Only Model Perform Translation?**
  Unlike encoder-decoder models which separate source encoding from target generation, a decoder-only model treats translation as next-token prediction conditioned on a prompt. The source sentence and translation instruction form the prefix, and the Hindi translation is generated autoregressively.
* **Why 4-bit NF4 QLoRA?**
  NormalFloat4 (NF4) is an information-theoretically optimal quantile quantization for normally distributed weights. Loading a 1B–2B model in 4-bit shrinks weight memory from 2–4 GB down to ~0.8–1.5 GB. LoRA ($r=16, \alpha=32$) injects low-rank adapter matrices into all linear projections (`q, k, v, o, gate, up, down`). Only the LoRA parameters (~1–2% of total) are updated, enabling fine-tuning on a single 16 GB GPU with zero degradation compared to 16-bit full tuning.
* **Why Response-Only Loss Masking?**
  In causal LM training, standard cross-entropy loss would penalize the model for failing to predict the English prompt tokens. By masking all prompt token positions with label `-100`, the loss is computed strictly on $P(\text{Hindi} \mid \text{English})$, forcing the model to concentrate gradient updates entirely on translation fidelity.
* **Why Left-Padding During Inference?**
  For causal decoder-only generation in batches, tokens are appended to the right. If inputs have variable lengths and are padded on the right, the model's position IDs and generation heads misalign with the padding tokens, generating garbage. Left-padding guarantees that the last token of every sequence is the final prompt token before generation starts.

---

## 2. Common Viva Examination Questions & Answers

### Q1: Why is BLEU alone insufficient for evaluating Hindi translations?
**Answer**: BLEU relies on exact n-gram matching against a single reference. Hindi is a morphologically rich language with extensive inflectional suffixes (vibhakti), free word order flexibility (SOV default, but SVO and OSV are often grammatically permissible), and high synonym diversity. A translation may be fluent and accurate but score low on BLEU due to surface-form mismatch. Therefore, we report **chrF / chrF++** (character n-gram F-score), which correlates significantly better with human judgment for morphologically complex and non-Latin scripts.

### Q2: What is the difference between Unigram and BPE tokenization?
**Answer**: Byte-Pair Encoding (BPE) begins with individual characters and greedily merges the most frequent pairs until the target vocabulary size is reached. Unigram tokenization begins with a massive vocabulary and iteratively removes subwords that contribute least to the likelihood of the training corpus according to a unigram language model. SentencePiece supports both; Unigram often produces superior segmentation for morphologically rich languages.

### Q3: How do encoder-decoder and decoder-only architectures compare for MT?
**Answer**:
* **Encoder-Decoder**:
  * *Strengths*: Explicit bidirectional source representation, cross-attention directly attends to all source tokens, highly compact model sizes (300M parameter mT5 achieves strong BLEU), dedicated translation conditioning.
  * *Weaknesses*: Fixed task framing; cannot easily adapt to open-ended dialogue or complex reasoning without separate heads.
* **Decoder-Only**:
  * *Strengths*: Massive zero-shot transfer ability, instruction-following capabilities, shared autoregressive pretraining across billions of tokens.
  * *Weaknesses*: KV-cache overhead scales with prompt length, unidirectional attention over the prompt (unless attention mask is modified), higher latency per token on single sentences.

### Q4: Why did we test multiple sampling temperatures?
**Answer**: Temperature $T$ scales the logits before the softmax operation:
$$P(y_t = w) = \frac{\exp(z_w / T)}{\sum_{v} \exp(z_v / T)}$$
* At $T \to 0$ (or greedy decoding), the model deterministically picks the highest-probability token. For machine translation, where precision and faithfulness are paramount, greedy search typically yields the highest BLEU/chrF.
* At higher temperatures ($T=0.7, 1.0$), the probability distribution flattens, increasing lexical diversity and synonym variety, but also increasing the probability of hallucinations, grammatical errors, and omission of numbers or named entities.
