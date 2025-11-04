#!/usr/bin/env python3
"""
train_transformer.py
A from-scratch Encoder-Decoder Transformer on tiny_shakespeare.
Features:
- Manual implementation of Attention, FFN, Residual+LayerNorm, PositionalEncoding
- GPU training (if available)
- Validation set + perplexity computation
- Early stopping
- Training curve plot + text generation
Author: ChatGPT (for assignment)
"""

import os, math, random, time, argparse, requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

# -------------------------
# Logging setup
# -------------------------
def setup_logger(log_path="results/train.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("Logger initialized. Writing to %s", log_path)


# -------------------------
# Utility
# -------------------------
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def subsequent_mask(size):
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask

# -------------------------
# Positional Encoding (fixed)
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = self._build_pe(max_len, d_model)
        self.register_buffer("pe", pe)

    def _build_pe(self, max_len, d_model, device=None, dtype=torch.float32):
        position = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            self.pe = self._build_pe(seq_len, self.d_model, x.device, x.dtype)
            self.register_buffer("pe", self.pe)
        return x + self.pe[:seq_len].unsqueeze(0)

# -------------------------
# Layers
# -------------------------
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        b, s, _ = x.size()
        x = x.view(b, s, self.n_heads, self.d_head).transpose(1,2)
        return x

    def combine_heads(self, x):
        b, h, s, dh = x.size()
        return x.transpose(1,2).contiguous().view(b, s, h*dh)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.split_heads(self.q_lin(q))
        k = self.split_heads(self.k_lin(k))
        v = self.split_heads(self.v_lin(v))
        if mask is not None and mask.dim()==2:
            mask = mask.unsqueeze(0).unsqueeze(1).expand(B, self.n_heads, mask.size(0), mask.size(1))
        out, _ = self.attn(q, k, v, mask)
        out = self.combine_heads(out)
        return self.out_lin(out)

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFFN(d_model, d_ff, dropout)
        self.ln1, self.ln2 = LayerNorm(d_model), LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.ln2(x + self.dropout(self.ff(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFFN(d_model, d_ff, dropout)
        self.ln1, self.ln2, self.ln3 = LayerNorm(d_model), LayerNorm(d_model), LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        x = self.ln1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.ln2(x + self.dropout(self.cross_attn(x, memory, memory, memory_mask)))
        x = self.ln3(x + self.dropout(self.ff(x)))
        return x

# -------------------------
# Encoder / Decoder / Transformer
# -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln = LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.pos(self.emb(x) * math.sqrt(self.emb.embedding_dim))
        for layer in self.layers:
            x = layer(x, mask)
        return self.ln(x)

class Decoder(nn.Module):
    def __init__(self, vocab, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln = LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        x = self.pos(self.emb(x) * math.sqrt(self.emb.embedding_dim))
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.out(self.ln(x))

class Transformer(nn.Module):
    def __init__(self, vocab, d_model, n_heads, d_ff, n_layers_enc, n_layers_dec, dropout):
        super().__init__()
        self.encoder = Encoder(vocab, d_model, n_layers_enc, n_heads, d_ff, dropout)
        self.decoder = Decoder(vocab, d_model, n_layers_dec, n_heads, d_ff, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encoder(src, src_mask)
        return self.decoder(tgt, memory, tgt_mask, None)

# -------------------------
# Data Loading
# -------------------------
def download_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    os.makedirs("data", exist_ok=True)
    path = "data/tiny_shakespeare.txt"
    if not os.path.exists(path):
        print("[DATA] Downloading...")
        text = requests.get(url).text
        open(path, "w").write(text)
    else:
        text = open(path).read()
    return text

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    return stoi, itos

def encode(text, stoi): return [stoi[c] for c in text]
def decode(ids, itos): return "".join(itos[i] for i in ids)

# -------------------------
# Training / Validation / Perplexity
# -------------------------
def get_batch(data, batch_size, block_size, device):
    n = len(data)
    ix = torch.randint(0, n - block_size - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix])
    return x.to(device), y.to(device)

def compute_loss(model, data, config, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(50):
            x, y = get_batch(data, config['batch_size'], config['block_size'], device)
            src, tgt_in = x[:, :-1], x[:, :-1]
            tgt_mask = subsequent_mask(tgt_in.size(1)).to(device)
            logits = model(src, tgt_in, None, tgt_mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y[:, :-1].reshape(-1))
            losses.append(loss.item())
    mean_loss = sum(losses)/len(losses)
    return mean_loss, math.exp(mean_loss)

def train(config):
    setup_logger("results/train.log")
    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    config['device'] = device
    print(f"[DEVICE] {device}")

    # Load and split data
    text = download_data()
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    vocab = len(stoi)

    model = Transformer(vocab, config['d_model'], config['n_heads'], config['d_ff'],
                        config['n_layers_enc'], config['n_layers_dec'], config['dropout']).to(device)
    opt = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    print(f"[MODEL] Params = {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    best_val = float('inf')
    patience = 10
    bad_epochs = 0
    train_losses, val_losses = [], []

    for epoch in range(config['epochs']):
        model.train()
        losses = []
        for _ in range(max(1, len(train_data)//(config['batch_size']*config['block_size']))):
            x, y = get_batch(train_data, config['batch_size'], config['block_size'], device)
            src, tgt_in = x[:, :-1], x[:, :-1]
            tgt_mask = subsequent_mask(tgt_in.size(1)).to(device)
            logits = model(src, tgt_in, None, tgt_mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y[:, :-1].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            opt.step()
            losses.append(loss.item())

        mean_train = sum(losses)/len(losses)
        val_loss, val_ppl = compute_loss(model, val_data, config, device)
        train_losses.append(mean_train)
        val_losses.append(val_loss)

        log_msg = f"Epoch {epoch+1:03d}: train={mean_train:.4f}, val={val_loss:.4f}, ppl={val_ppl:.2f}"
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "results/best_model.pt")
            bad_epochs = 0
            log_msg += " [BEST]"
        else:
            bad_epochs += 1
            log_msg += f" (no improve {bad_epochs}/{patience})"
        logging.info(log_msg)

        if bad_epochs >= patience:
            logging.info("[EARLY STOP] Validation did not improve for %d epochs.", patience)
            break

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/loss_curve.png")
    print("[PLOT] saved -> results/loss_curve.png")

    # load best model
    model.load_state_dict(torch.load("results/best_model.pt"))
    sample = generate(model, "ROMEO: ", stoi, itos, config, 400)
    print("\n==== GENERATED TEXT ====\n", sample)

@torch.no_grad()
def generate(model, prompt, stoi, itos, config, max_new_tokens=200):
    model.eval()
    device = config['device']
    src = torch.tensor([[stoi[c] for c in prompt]], device=device)
    memory = model.encoder(src)
    out = src.clone()
    for _ in range(max_new_tokens):
        tgt_mask = subsequent_mask(out.size(1)).to(device)
        logits = model.decoder(out, memory, tgt_mask)
        next_id = torch.multinomial(F.softmax(logits[0, -1, :], dim=-1), 1).item()
        out = torch.cat([out, torch.tensor([[next_id]], device=device)], dim=1)
    return "".join(itos[i.item()] for i in out[0])

if __name__ == "__main__":
    cfg = dict(
        epochs=800, batch_size=64, block_size=128, d_model=256,
        n_heads=8, d_ff=1024, n_layers_enc=3, n_layers_dec=3,
        dropout=0.1, lr=3e-4, weight_decay=1e-2,
        grad_clip=1.0, seed=1337
    )
    os.makedirs("results", exist_ok=True)
    train(cfg)
