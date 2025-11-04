# LLM-homework
大模型基础与应用-中期作业

This project implements a full **Encoder–Decoder Transformer** architecture **from scratch** (no `transformers` library),
trained on the **Tiny Shakespeare** character-level dataset.

### Features
- Pure PyTorch implementation (no `transformers`)
- Encoder + Decoder with Multi-Head Attention, FFN, LayerNorm, Positional Encoding
- GPU support
- Train/Validation split, Perplexity computation
- Early stopping
- Training loss curve plot and generated sample text

---

## 🧰 Requirements

Install dependencies:
```bash
conda create-n transformer
pip install -r requirements.txt
