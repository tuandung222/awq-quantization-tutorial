# AWQ Quantization Tutorial: Qwen3-4B-Instruct

A comprehensive, beginner-friendly tutorial project on **Activation-aware Weight Quantization (AWQ)** — from mathematical foundations to hands-on model quantization and inference.

We walk through quantizing **[Qwen/Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct)** to 4-bit INT4 using [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), and show how to load the quantized checkpoint for fast inference with HuggingFace Transformers and vLLM.

---

## 📖 Table of Contents

| # | Document | Description |
|---|----------|-------------|
| ⭐ | **[Visualizing Quantization](visualizing_quantization/)** | **Start Here:** Beginner-friendly visual guide & intuition building (Notebooks + Analogies) |
| 1 | [AWQ Theory Deep-Dive](docs/01_awq_theory.md) | Mathematical foundations, quantization basics, and the full AWQ algorithm explained |
| 2 | [AWQ vs Other Methods](docs/02_awq_vs_others.md) | Comparison with GPTQ, RTN, SmoothQuant, and GGUF |
| 3 | [Quantization Guide](tutorials/03_quantize_qwen3_awq.md) | Step-by-step guide to quantize Qwen3-4B-Instruct with AWQ |
| 4 | [Inference Guide](tutorials/04_inference_awq.md) | Loading AWQ checkpoints and running inference |

### 🔬 Jupyter Notebooks

| Notebook | Description |
|----------|-------------|
| [quantize_qwen3_awq.ipynb](notebooks/quantize_qwen3_awq.ipynb) | Hands-on quantization notebook with progress tracking |
| [inference_awq.ipynb](notebooks/inference_awq.ipynb) | Inference demo with benchmarks |

### 🛠️ CLI Scripts

| Script | Description |
|--------|-------------|
| [quantize.py](scripts/quantize.py) | Standalone quantization CLI tool |
| [inference.py](scripts/inference.py) | Standalone inference CLI tool |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with Compute Capability ≥ 7.5 (e.g., A100, H100)
- CUDA 11.8+

### Installation

```bash
# Clone this repository
git clone https://github.com/tuandung222/awq-quantization-tutorial.git
cd awq-quantization-tutorial

# Install dependencies
pip install -r requirements.txt
```

### Quantize Qwen3-4B-Instruct (one command)

```bash
python scripts/quantize.py \
    --model_path Qwen/Qwen3-4B-Instruct \
    --output_path ./qwen3-4b-instruct-awq \
    --w_bit 4 \
    --q_group_size 128
```

### Run Inference

```bash
python scripts/inference.py \
    --model_path ./qwen3-4b-instruct-awq \
    --prompt "Explain quantum computing in simple terms"
```

---

## 📊 Hardware Requirements

| Task | Minimum GPU VRAM | Recommended |
|------|-------------------|-------------|
| Quantization (4B model) | 16 GB | 24+ GB (A100/H100) |
| Inference (AWQ INT4) | ~3 GB | 8+ GB |
| Inference (FP16 original) | ~8 GB | 16+ GB |

> This tutorial was developed and tested on a machine with **4× NVIDIA H100 GPUs**.

---

## 📚 Learning Path

If you're new to quantization, we recommend following the documents in order:

1. **Start with theory** → [01_awq_theory.md](docs/01_awq_theory.md) to understand the math
2. **Compare methods** → [02_awq_vs_others.md](docs/02_awq_vs_others.md) to understand why AWQ
3. **Quantize a model** → [03_quantize_qwen3_awq.md](tutorials/03_quantize_qwen3_awq.md) for hands-on practice
4. **Run inference** → [04_inference_awq.md](tutorials/04_inference_awq.md) to deploy your quantized model

---

## 🔗 References

- **AWQ Paper**: [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) (MLSys 2024 Best Paper)
- **AutoAWQ**: [github.com/casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- **Qwen3**: [Qwen/Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct)

## License

This tutorial project is released under the MIT License. The Qwen3 model is subject to its own license terms.
