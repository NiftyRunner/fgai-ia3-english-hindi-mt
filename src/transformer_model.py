"""Model A: Basic PyTorch Encoder-Decoder Transformer trained from scratch.

Features:
- d_model=256, nhead=4, 3 encoder + 3 decoder layers, dim_feedforward=1024, dropout=0.1
- Sinusoidal positional embeddings
- Tied target embeddings and output projection
- Proper causal and padding masks
- Modern torch.amp.GradScaler('cuda') mixed-precision training
- Complete epoch checkpointing with cumulative training time tracking
"""

import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.metrics import CumulativeTimer


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding as introduced in 'Attention Is All You Need'."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model) for batch_first=True
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """Token embedding layer with scaling by sqrt(d_model)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)


class Seq2SeqTransformer(nn.Module):
    """Encoder-Decoder Transformer with tied embeddings and batch_first=True."""

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Shared/Joint embedding representation
        self.src_tok_emb = TokenEmbedding(vocab_size, d_model)
        self.tgt_tok_emb = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output linear projection
        self.generator = nn.Linear(d_model, vocab_size)

        # Weight tying: ties target embedding weights to generator output weights
        # Reduces parameter count and improves generalization on small datasets
        self.generator.weight = self.tgt_tok_emb.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform weight initialization for Transformer parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Embed and add positional encoding
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)),
            mask=src_mask,
        )

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory=memory,
            tgt_mask=tgt_mask,
        )


def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Generate causal upper-triangular boolean mask preventing attention to future tokens."""
    mask = torch.triu(torch.ones((sz, sz), device=device, dtype=torch.bool), diagonal=1)
    return mask


def create_masks(
    src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct causal and padding masks for Transformer forward pass.

    - tgt_mask: boolean causal mask where True positions are blocked
    - src_padding_mask: True where token is pad_idx
    - tgt_padding_mask: True where token is pad_idx
    """
    device = src.device
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = None

    src_padding_mask = src == pad_idx
    tgt_padding_mask = tgt == pad_idx
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class ParallelDataset(Dataset):
    """PyTorch Dataset for parallel English-Hindi sentence pairs."""

    def __init__(self, data: List[Dict[str, Any]], tokenizer: Any, max_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        src_ids = self.tokenizer.encode(item["en"], add_bos=True, add_eos=True)[: self.max_len]
        tgt_ids = self.tokenizer.encode(item["hi"], add_bos=True, add_eos=True)[: self.max_len]
        return {
            "id": item.get("id", idx),
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "en": item["en"],
            "hi": item["hi"],
        }


def collate_fn_pad(batch: List[Dict[str, Any]], pad_idx: int = 0) -> Dict[str, Any]:
    """Pads variable-length sequences within a batch to uniform batch maximum."""
    src_list = [torch.tensor(b["src_ids"], dtype=torch.long) for b in batch]
    tgt_list = [torch.tensor(b["tgt_ids"], dtype=torch.long) for b in batch]

    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "ids": [b["id"] for b in batch],
        "en": [b["en"] for b in batch],
        "hi": [b["hi"] for b in batch],
    }


def train_one_epoch(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float = 1.0,
    pad_idx: int = 0,
) -> float:
    """Execute one training epoch with fp16 mixed precision and gradient clipping."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        # Teacher forcing inputs and targets
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_masks(src, tgt_input, pad_idx)

        optimizer.zero_grad()

        # Mixed precision context using modern torch.amp
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
            logits = model(
                src=src,
                tgt=tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_padding_mask=src_pad_mask,
                tgt_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_expected.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_loss(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_idx: int = 0,
) -> float:
    """Compute average cross-entropy loss over validation dataloader."""
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_masks(src, tgt_input, pad_idx)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
            logits = model(
                src=src,
                tgt=tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_padding_mask=src_pad_mask,
                tgt_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_expected.reshape(-1))

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def batched_greedy_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    max_len: int = 128,
    bos_idx: int = 2,
    eos_idx: int = 3,
    pad_idx: int = 0,
) -> torch.Tensor:
    """Generate translations token by token using batched greedy decoding.

    Why batched greedy decoding for viva:
    Autoregressive decoding generates each token conditioned on all previously generated tokens.
    Greedy decoding chooses argmax at each step. Batching accelerates inference on T4 by
    evaluating full batches simultaneously with causal masking.
    """
    model.eval()
    device = src.device
    batch_size = src.shape[0]

    src_pad_mask = src == pad_idx
    memory = model.transformer.encoder(
        model.positional_encoding(model.src_tok_emb(src)),
        src_key_padding_mask=src_pad_mask,
    )

    # Initialize decoder with BOS token for all sequences
    ys = torch.full((batch_size, 1), fill_value=bos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.shape[1], device)
        out = model.transformer.decoder(
            model.positional_encoding(model.tgt_tok_emb(ys)),
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        # Project last token representation to vocabulary
        prob = model.generator(out[:, -1, :])
        next_word = torch.argmax(prob, dim=-1)

        # Append next word
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

        # Check for sequences that have generated EOS
        finished = finished | (next_word == eos_idx)
        if finished.all():
            break

    return ys
