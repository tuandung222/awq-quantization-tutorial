# AWQ vs Other Quantization Methods

> A practical comparison of popular LLM quantization techniques to help you choose the right method.

---

## Overview

| Method | Type | Bits | Requires Training? | Calibration Data | Speed | Key Idea |
|--------|------|------|---------------------|------------------|-------|----------|
| **RTN** | PTQ, weight-only | 4-8 | ❌ | None | ⚡ Instant | Simple round-to-nearest |
| **GPTQ** | PTQ, weight-only | 2-8 | ❌ | ~128 samples | 🕐 Moderate | Layer-wise reconstruction via Hessian |
| **AWQ** | PTQ, weight-only | 4 | ❌ | ~128 samples | ⚡ Fast | Activation-aware scaling |
| **SmoothQuant** | PTQ, W+A | 8 | ❌ | ~128 samples | ⚡ Fast | Smooth activation outliers |
| **GGUF** | PTQ, weight-only | 2-8 | ❌ | Varies | ⚡ Fast | Multiple quant schemes for CPU/GPU |
| **QAT** | Training-based | 4-8 | ✅ | Full dataset | 🐌 Slow | Quantization-aware fine-tuning |

---

## Detailed Comparison

### 1. RTN (Round-to-Nearest)

**How it works**: The simplest approach — compute scale and zero-point per group, round each weight to the nearest integer.

```
Q(w) = round(w / Δ) · Δ
```

| Pros | Cons |
|------|------|
| ✅ No calibration data needed | ❌ Significant accuracy loss at 4-bit |
| ✅ Instantaneous | ❌ Treats all weights equally |
| ✅ Simple implementation | ❌ Not suitable for production |

**When to use**: Quick baseline, 8-bit quantization where accuracy isn't critical.

---

### 2. GPTQ (GPT Quantization)

**How it works**: Quantizes weights one column at a time, using a Hessian-based reconstruction to minimize the layer output error. After quantizing one weight, it adjusts the remaining un-quantized weights to compensate for the error.

**Key formula** (Optimal Brain Quantizer):
$$
\hat{w}_q = \text{argmin}_{q} \frac{(\text{quant}(w_q) - w_q)^2}{[H^{-1}]_{qq}}
$$

Where `H` is the Hessian matrix of the layer (computed from calibration activations).

| Pros | Cons |
|------|------|
| ✅ Often achieves excellent perplexity | ❌ Slower quantization (Hessian computation) |
| ✅ Works well at 3-4 bits | ❌ Order-dependent (sequential column processing) |
| ✅ Mature ecosystem (ExLlamaV2) | ❌ Can be sensitive to calibration data |
| ✅ Supports 2-3 bit quantization | ❌ More complex implementation |

**When to use**: When you need the absolute best perplexity, especially at very low bit-widths (2-3 bit).

---

### 3. AWQ (Activation-aware Weight Quantization)

**How it works**: Identifies salient weight channels via activation statistics, applies per-channel scaling to protect them, then quantizes. Uses grid search over a single hyperparameter α.

**Key formula**:
$$
s_j = \left(\frac{|\bar{x}_j|}{\max_k |\bar{x}_k|}\right)^{\alpha}, \quad \mathbf{W'} = \mathbf{W} \cdot \text{diag}(\mathbf{s})^{-1}
$$

| Pros | Cons |
|------|------|
| ✅ Fast quantization (minutes) | ❌ Primarily 4-bit only |
| ✅ Excellent generalization across models | ❌ AutoAWQ library is deprecated (still works) |
| ✅ No backpropagation needed | ❌ Slightly lower accuracy than GPTQ on some models |
| ✅ Robust to calibration data choice | |
| ✅ Hardware-friendly (uniform INT4) | |
| ✅ Native HuggingFace Transformers support | |
| ✅ MLSys 2024 Best Paper Award | |

**When to use**: General-purpose 4-bit quantization with the best speed/quality trade-off. Excellent default choice.

---

### 4. SmoothQuant

**How it works**: Smooths the activation outliers by migrating the quantization difficulty from activations to weights. Applies a per-channel smoothing factor that divides activations and multiplies weights.

$$
\mathbf{Y} = (\mathbf{X} \cdot \text{diag}(\mathbf{s})^{-1}) \cdot (\text{diag}(\mathbf{s}) \cdot \mathbf{W}) = \hat{\mathbf{X}} \cdot \hat{\mathbf{W}}
$$

| Pros | Cons |
|------|------|
| ✅ Enables W8A8 quantization | ❌ Primarily INT8, not INT4 |
| ✅ True acceleration (INT8 GEMM kernels) | ❌ Requires activation quantization |
| ✅ Works well for encoder models | ❌ Less effective for very large models |

**When to use**: When you need INT8×INT8 acceleration with both weight and activation quantization.

> **Note**: AWQ and SmoothQuant share a similar mathematical intuition (per-channel scaling), but apply it differently — AWQ scales weights for weight-only INT4, while SmoothQuant smooths activations for W8A8.

---

### 5. GGUF (llama.cpp)

**How it works**: GGUF is a file format (successor to GGML) used by llama.cpp. It supports many quantization types (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, etc.) with different precision/size trade-offs. Uses a "k-quant" scheme with importance-based mixed precision across layers.

| Pros | Cons |
|------|------|
| ✅ Runs on CPU (no GPU required) | ❌ Not as fast as GPU-optimized formats |
| ✅ Many quantization levels (2-8 bit) | ❌ CPU-focused (GPU support improving) |
| ✅ Huge community (Ollama, LM Studio) | ❌ Different ecosystem from HuggingFace |
| ✅ Mixed-precision across layers | ❌ Conversion from HuggingFace format needed |

**When to use**: Running models on CPU, local deployment, compatibility with llama.cpp ecosystem.

---

### 6. QAT (Quantization-Aware Training)

**How it works**: Inserts fake quantization operations during training/fine-tuning, allowing the model to learn to be robust to quantization errors.

| Pros | Cons |
|------|------|
| ✅ Best accuracy at any bit-width | ❌ Requires full training loop |
| ✅ Model adapts to quantization | ❌ Very expensive (GPU hours/days) |
| ✅ Works at extreme low-bits (1-2 bit) | ❌ Needs training data and infrastructure |

**When to use**: When accuracy is paramount and you have the compute budget for training.

---

## Benchmark Comparison

### Perplexity on WikiText-2 (lower is better)

| Model | FP16 | RTN-4bit | GPTQ-4bit | AWQ-4bit |
|-------|------|----------|-----------|----------|
| Llama-2-7B | 5.47 | 6.29 | 5.63 | 5.60 |
| Llama-2-13B | 4.88 | 5.42 | 4.99 | 4.97 |
| Llama-2-70B | 3.32 | 3.75 | 3.41 | 3.40 |
| Mistral-7B | 5.25 | 5.89 | 5.38 | 5.37 |

> Source: AWQ paper (Lin et al., 2024) and community benchmarks. Values are approximate.

### Quantization Speed (7B model on A100)

| Method | Time |
|--------|------|
| RTN | <1 minute |
| AWQ | ~10 minutes |
| GPTQ | ~30-60 minutes |
| QAT | Hours-Days |

### Inference Speed (tokens/sec, 7B model, A100)

| Format | Batch=1 | Batch=8 |
|--------|---------|---------|
| FP16 | ~50 t/s | ~200 t/s |
| AWQ-4bit | ~100 t/s | ~400 t/s |
| GPTQ-4bit | ~95 t/s | ~380 t/s |
| GGUF-Q4_K_M (GPU) | ~85 t/s | ~320 t/s |

> Values are approximate and depend on specific hardware, model, and inference engine.

---

## Decision Flowchart

```
Start
  │
  ├─ Need maximum accuracy at INT4?
  │   ├─ Yes → GPTQ or AWQ (try both, pick lower perplexity)
  │   └─ No ─┐
  │           │
  ├─ Need to run on CPU?
  │   ├─ Yes → GGUF (llama.cpp)
  │   └─ No ─┐
  │           │
  ├─ Need INT8 with activation quantization?
  │   ├─ Yes → SmoothQuant
  │   └─ No ─┐
  │           │
  ├─ Need fastest quantization process?
  │   ├─ Yes → AWQ (or RTN for quick testing)
  │   └─ No ─┐
  │           │
  ├─ Have compute budget for training?
  │   ├─ Yes → QAT
  │   └─ No → AWQ (best general-purpose choice)
  │
  └─ Default recommendation: AWQ
```

---

## Summary

**AWQ is the recommended default** for most use cases because it offers:
- Near-GPTQ accuracy with faster quantization
- Excellent generalization (works well across model families)
- Native HuggingFace + vLLM support
- Hardware-friendly uniform INT4 format
- MLSys 2024 Best Paper — well-researched and validated

Choose GPTQ when you need every fraction of perplexity, GGUF for CPU deployment, and QAT when you have training budget and need extreme compression.

---

**← Previous**: [01 — AWQ Theory Deep-Dive](01_awq_theory.md)  
**Next →**: [03 — Quantizing Qwen3-4B-Instruct](../tutorials/03_quantize_qwen3_awq.md)
