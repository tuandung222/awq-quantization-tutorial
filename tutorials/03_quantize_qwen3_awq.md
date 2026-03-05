# Step-by-Step: Quantize Qwen3-4B-Instruct with AWQ

> **Goal**: Convert `Qwen/Qwen3-4B-Instruct` from FP16 (~8 GB) to AWQ INT4 (~2.5 GB) using AutoAWQ.  
> **Time**: ~10-15 minutes on a single H100 GPU.  
> **Prerequisites**: NVIDIA GPU with ≥16 GB VRAM, CUDA 11.8+, Python 3.10+

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Understanding the Configuration](#2-understanding-the-configuration)
3. [Download the Model](#3-download-the-model)
4. [Prepare Calibration Data](#4-prepare-calibration-data)
5. [Run AWQ Quantization](#5-run-awq-quantization)
6. [Verify the Quantized Model](#6-verify-the-quantized-model)
7. [Upload to HuggingFace Hub](#7-upload-to-huggingface-hub-optional)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Environment Setup

### Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv awq-env
source awq-env/bin/activate

# Install PyTorch (adjust CUDA version as needed)
pip install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install AutoAWQ and related packages
pip install autoawq>=0.2.7
pip install transformers>=4.51.0 accelerate datasets

# Verify installation
python -c "import awq; print('AutoAWQ version:', awq.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

### Verify GPU

```bash
nvidia-smi
```

You should see your GPU(s) listed. For quantizing a 4B parameter model, a single GPU with ≥16 GB VRAM is sufficient.

---

## 2. Understanding the Configuration

AWQ quantization uses these key parameters:

```python
quant_config = {
    "zero_point": True,      # Use asymmetric quantization (with zero-point)
    "q_group_size": 128,     # Number of weights sharing one scale/zero-point
    "w_bit": 4,              # Quantization bit-width (4-bit)
    "version": "GEMM"        # Kernel version: GEMM (batched) or GEMV (single)
}
```

### Parameter Explanation

| Parameter | Value | Why |
|-----------|-------|-----|
| `zero_point` | `True` | Asymmetric quantization captures weight distributions better |
| `q_group_size` | `128` | Groups of 128 weights share one scale factor. Smaller = more accurate but more overhead |
| `w_bit` | `4` | 4-bit quantization — best balance of compression (3.86×) and accuracy |
| `version` | `"GEMM"` | General Matrix Multiply kernel — best for batched inference and vLLM compatibility |

---

## 3. Download the Model

AutoAWQ handles model downloading automatically from HuggingFace Hub. But you can also pre-download:

```bash
# Option 1: Let AutoAWQ download during quantization (easiest)
# Just use "Qwen/Qwen3-4B-Instruct" as the model path

# Option 2: Pre-download with huggingface-cli
huggingface-cli download Qwen/Qwen3-4B-Instruct --local-dir ./qwen3-4b-instruct-fp16
```

---

## 4. Prepare Calibration Data

AWQ needs a small calibration dataset to collect activation statistics. WikiText-2 is the standard choice.

```python
from datasets import load_dataset

# Load WikiText-2
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Preview
print(f"Dataset size: {len(dataset)} samples")
print(f"Sample text: {dataset[100]['text'][:200]}")
```

AutoAWQ handles this internally — you just need to specify the dataset name. But understanding what happens:

1. **Loads** 128 random text samples from the calibration dataset
2. **Tokenizes** them with the model's tokenizer
3. **Runs forward pass** through the model to collect activation statistics
4. **Computes** per-channel mean absolute activation values: `x̄_j = mean(|X[:, j]|)`

---

## 5. Run AWQ Quantization

### Full Quantization Script

```python
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Configuration
MODEL_ID = "Qwen/Qwen3-4B-Instruct"
OUTPUT_DIR = "./qwen3-4b-instruct-awq"

QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Step 1: Load model and tokenizer
print("Loading model...")
model = AutoAWQForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"           # Automatically distribute across GPUs
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Step 2: Quantize
print("Starting AWQ quantization...")
model.quantize(
    tokenizer,
    quant_config=QUANT_CONFIG,
    calib_data="wikitext",      # Use WikiText-2 for calibration
    # Alternative: pass your own calibration data as a list of strings
    # calib_data=["sample text 1", "sample text 2", ...]
)

# Step 3: Save quantized model
print(f"Saving quantized model to {OUTPUT_DIR}...")
model.save_quantized(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done! Quantized model saved.")
```

### What Happens During Quantization

```
Loading model...
  → Downloads ~8 GB FP16 model from HuggingFace
  → Loads into GPU memory

Starting AWQ quantization...
  → Fetches WikiText-2 calibration dataset
  → Tokenizes 128 samples  
  → For each transformer layer (36 layers for Qwen3-4B):
      → Forward pass to get input activations
      → Compute per-channel activation statistics
      → Grid search for optimal α (20 values)
      → Apply scaling transformation
      → Quantize weights to INT4
  → Total: ~10-15 minutes on H100

Saving quantized model...
  → Saves INT4 weights + FP16 scales/zero-points
  → Saves tokenizer and config
  → Output: ~2.5 GB folder
```

### Using the CLI Script

We provide a ready-made CLI script:

```bash
python scripts/quantize.py \
    --model_path Qwen/Qwen3-4B-Instruct \
    --output_path ./qwen3-4b-instruct-awq \
    --w_bit 4 \
    --q_group_size 128 \
    --version GEMM \
    --calib_data wikitext \
    --calib_size 128
```

---

## 6. Verify the Quantized Model

### Check Model Size

```bash
# Compare sizes
echo "Original model (FP16):"
du -sh ./qwen3-4b-instruct-fp16/

echo "Quantized model (AWQ INT4):"
du -sh ./qwen3-4b-instruct-awq/
```

Expected: ~8 GB → ~2.5 GB (approximately 3.2× compression with metadata overhead).

### Check Model Files

```bash
ls -la ./qwen3-4b-instruct-awq/
```

You should see:
```
model.safetensors        # Quantized weights (~2.3 GB)
config.json              # Model config (with quantization_config)
tokenizer.json           # Tokenizer
tokenizer_config.json    # Tokenizer config
generation_config.json   # Generation defaults
quant_config.json        # AWQ quantization config
```

### Quick Inference Test

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load quantized model
model = AutoAWQForCausalLM.from_quantized(
    "./qwen3-4b-instruct-awq",
    fuse_layers=True         # Fuse QKV and MLP layers for faster inference
)
tokenizer = AutoTokenizer.from_pretrained("./qwen3-4b-instruct-awq")

# Test generation
prompt = "What is the capital of France?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"Response: {response}")
```

---

## 7. Upload to HuggingFace Hub (Optional)

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("your-username/Qwen3-4B-Instruct-AWQ", exist_ok=True)

# Upload all files
api.upload_folder(
    folder_path="./qwen3-4b-instruct-awq",
    repo_id="your-username/Qwen3-4B-Instruct-AWQ",
    commit_message="Upload AWQ quantized Qwen3-4B-Instruct"
)

print("Model uploaded to HuggingFace Hub!")
```

---

## 8. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Use `device_map="auto"` for multi-GPU, or reduce `calib_size` |
| `ImportError: No module named 'awq'` | `pip install autoawq` |
| `Kernel not found` | Install with kernels: `pip install autoawq[kernels]` |
| `Slow quantization` | Check if model is on GPU: `nvidia-smi` during quantization |
| `Wrong tokenizer` | Always use `trust_remote_code=True` for Qwen models |
| `Version mismatch` | Ensure `torch`, `transformers`, and `autoawq` versions are compatible |

### Memory Requirements

| Model Size | FP16 Load | During Quantization | Quantized Output |
|------------|-----------|--------------------:|-----------------|
| 4B | ~8 GB | ~12-14 GB peak | ~2.5 GB |
| 7B | ~14 GB | ~20-24 GB peak | ~4 GB |
| 14B | ~28 GB | ~40-48 GB peak | ~8 GB |

> With 4× H100 (80 GB each), you have 320 GB total — more than enough for any model!

---

**← Previous**: [02 — AWQ vs Other Methods](../docs/02_awq_vs_others.md)  
**Next →**: [04 — Inference with AWQ](04_inference_awq.md)
