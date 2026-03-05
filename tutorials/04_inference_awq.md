# Inference with AWQ Quantized Models

> **Goal**: Load your AWQ-quantized Qwen3-4B-Instruct and run inference using three different methods.  
> **Prerequisites**: A quantized model checkpoint (from the [quantization tutorial](03_quantize_qwen3_awq.md))

---

## Table of Contents

1. [Method 1 — AutoAWQ Native](#method-1--autoawq-native)
2. [Method 2 — HuggingFace Transformers](#method-2--huggingface-transformers)
3. [Method 3 — vLLM Server](#method-3--vllm-server)
4. [Performance Comparison](#performance-comparison)
5. [Chat Template Usage](#chat-template-usage)
6. [Batch Inference](#batch-inference)

---

## Method 1 — AutoAWQ Native

The most direct approach using AutoAWQ's own loading mechanism.

### Basic Usage

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

MODEL_PATH = "./qwen3-4b-instruct-awq"
# Or from HuggingFace: "Qwen/Qwen3-4B-AWQ"

# Load model (fuse_layers=True for optimized inference)
model = AutoAWQForCausalLM.from_quantized(
    MODEL_PATH,
    fuse_layers=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Prepare input
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain the theory of relativity in 3 sentences."}
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# Decode (skip the input tokens)
response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)
print(response)
```

### Key Parameters for `from_quantized()`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fuse_layers` | `True` | Fuse QKV projections and MLP layers for faster inference |
| `device_map` | `"auto"` | Automatically distribute model across GPUs |
| `max_memory` | `None` | Dict specifying max memory per device |
| `offload_folder` | `None` | Folder for CPU offloading (for large models) |

---

## Method 2 — HuggingFace Transformers

Since Transformers v4.35+, AWQ models are natively supported. This is the easiest method if you're already using the HuggingFace ecosystem.

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "./qwen3-4b-instruct-awq"

# Load — Transformers auto-detects AWQ quantization from config.json
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Prepare input with chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a Python function to find prime numbers."}
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)
print(response)
```

### How Transformers Detects AWQ

When you load an AWQ model, `config.json` contains:

```json
{
    "quantization_config": {
        "quant_method": "awq",
        "zero_point": true,
        "group_size": 128,
        "bits": 4,
        "version": "gemm"
    }
}
```

Transformers reads this and automatically:
1. Replaces standard `nn.Linear` layers with `WQLinear_GEMM` (AWQ-optimized)
2. Loads the INT4 packed weights, scales, and zero-points
3. Handles dequantization during forward pass

### Using with `pipeline()`

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

messages = [{"role": "user", "content": "What is machine learning?"}]
output = pipe(messages, max_new_tokens=256, temperature=0.7)
print(output[0]["generated_text"][-1]["content"])
```

---

## Method 3 — vLLM Server

vLLM provides the highest throughput for serving AWQ models in production.

### Install vLLM

```bash
pip install vllm>=0.6.1
```

### Option A: Python API

```python
from vllm import LLM, SamplingParams

MODEL_PATH = "./qwen3-4b-instruct-awq"

# Initialize vLLM engine
llm = LLM(
    model=MODEL_PATH,
    quantization="awq",         # Explicitly specify AWQ
    dtype="float16",
    gpu_memory_utilization=0.8, # Use 80% of GPU memory
    max_model_len=4096          # Maximum sequence length
)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

# Generate
prompts = [
    "Explain quantum computing:",
    "Write a haiku about AI:",
    "What is the meaning of life?"
]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Response: {output.outputs[0].text}\n")
```

### Option B: OpenAI-Compatible API Server

Start the server:

```bash
vllm serve ./qwen3-4b-instruct-awq \
    --quantization awq \
    --dtype float16 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096
```

Then query with any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="./qwen3-4b-instruct-awq",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain AWQ quantization in simple terms."}
    ],
    max_tokens=256,
    temperature=0.7
)

print(response.choices[0].message.content)
```

Or with `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "./qwen3-4b-instruct-awq",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }'
```

---

## Performance Comparison

### Memory Usage (Qwen3-4B-Instruct)

| Format | GPU VRAM | Compression |
|--------|----------|-------------|
| FP16 (original) | ~8.0 GB | 1× |
| AWQ INT4 | ~2.5 GB | 3.2× |

### Inference Speed (approximate, single H100)

| Method | Tokens/sec (batch=1) | Tokens/sec (batch=8) |
|--------|---------------------|---------------------|
| FP16 + Transformers | ~45 | ~180 |
| AWQ + AutoAWQ | ~85 | ~340 |
| AWQ + Transformers | ~80 | ~320 |
| AWQ + vLLM | ~110 | ~500+ |

> vLLM achieves the highest throughput due to PagedAttention and continuous batching.

---

## Chat Template Usage

Qwen3 models use a specific chat template. Here's how to format messages correctly:

```python
# Single turn
messages = [
    {"role": "user", "content": "Hello, who are you?"}
]

# Multi-turn
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a fibonacci function in Python"},
    {"role": "assistant", "content": "Here's a fibonacci function:\n```python\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n```"},
    {"role": "user", "content": "Now make it iterative"}
]

# Apply template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True   # Adds the assistant's response prefix
)
```

---

## Batch Inference

For processing multiple prompts efficiently:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_quantized(
    "./qwen3-4b-instruct-awq",
    fuse_layers=True
)
tokenizer = AutoTokenizer.from_pretrained("./qwen3-4b-instruct-awq")
tokenizer.pad_token = tokenizer.eos_token

# Prepare batch
prompts = [
    "What is gravity?",
    "Explain photosynthesis.",
    "How does a computer work?",
]

all_messages = [
    [{"role": "user", "content": p}] for p in prompts
]

texts = [
    tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    for m in all_messages
]

inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True
)

for i, output in enumerate(outputs):
    response = tokenizer.decode(
        output[inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    print(f"Q: {prompts[i]}")
    print(f"A: {response}\n")
```

---

**← Previous**: [03 — Quantizing Qwen3-4B-Instruct](03_quantize_qwen3_awq.md)  
**Next →**: Try the [Jupyter Notebooks](../notebooks/) for interactive examples!
