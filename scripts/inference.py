#!/usr/bin/env python3
"""
AWQ Inference Script for Quantized Models

Usage:
    python inference.py --model_path ./qwen3-4b-instruct-awq --prompt "Hello, who are you?"

For all options:
    python inference.py --help
"""

import argparse
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with an AWQ quantized model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple prompt
  python inference.py --model_path ./qwen3-4b-awq --prompt "What is AI?"

  # With system prompt
  python inference.py --model_path ./qwen3-4b-awq \\
      --prompt "Write a Python sort function" \\
      --system "You are a coding assistant."

  # Custom generation parameters
  python inference.py --model_path ./qwen3-4b-awq \\
      --prompt "Tell me a story" \\
      --max_tokens 512 --temperature 0.9

  # Using HuggingFace Transformers backend
  python inference.py --model_path ./qwen3-4b-awq \\
      --prompt "Explain gravity" --backend transformers
        """,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to AWQ quantized model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User prompt text",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Optional system prompt",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling (default: 0.9)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="autoawq",
        choices=["autoawq", "transformers"],
        help="Inference backend (default: autoawq)",
    )
    parser.add_argument(
        "--no_fuse",
        action="store_true",
        help="Disable layer fusion (AutoAWQ backend only)",
    )

    return parser.parse_args()


def inference_autoawq(args):
    """Run inference using AutoAWQ native backend."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"Loading model with AutoAWQ backend...")
    start_load = time.time()

    model = AutoAWQForCausalLM.from_quantized(
        args.model_path,
        fuse_layers=not args.no_fuse,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.1f}s")

    # Build messages
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    # Tokenize
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Generate
    print(f"Generating (max {args.max_tokens} tokens)...")
    start_gen = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
        )

    gen_time = time.time() - start_gen
    output_len = outputs.shape[1] - input_len

    # Decode
    response = tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    )

    return response, {
        "load_time": load_time,
        "gen_time": gen_time,
        "input_tokens": input_len,
        "output_tokens": output_len,
        "tokens_per_sec": output_len / gen_time if gen_time > 0 else 0,
    }


def inference_transformers(args):
    """Run inference using HuggingFace Transformers backend."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model with Transformers backend...")
    start_load = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.1f}s")

    # Build messages
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    # Tokenize
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Generate
    print(f"Generating (max {args.max_tokens} tokens)...")
    start_gen = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
        )

    gen_time = time.time() - start_gen
    output_len = outputs.shape[1] - input_len

    # Decode
    response = tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    )

    return response, {
        "load_time": load_time,
        "gen_time": gen_time,
        "input_tokens": input_len,
        "output_tokens": output_len,
        "tokens_per_sec": output_len / gen_time if gen_time > 0 else 0,
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("AWQ Inference")
    print("=" * 60)
    print(f"  Model:       {args.model_path}")
    print(f"  Backend:     {args.backend}")
    print(f"  Prompt:      {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    if args.system:
        print(f"  System:      {args.system[:80]}{'...' if len(args.system) > 80 else ''}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p:       {args.top_p}")
    print("=" * 60)

    # Run inference
    if args.backend == "autoawq":
        response, stats = inference_autoawq(args)
    else:
        response, stats = inference_transformers(args)

    # Print results
    print("\n" + "=" * 60)
    print("Response:")
    print("=" * 60)
    print(response)
    print("\n" + "-" * 60)
    print("Statistics:")
    print(f"  Model load time:  {stats['load_time']:.1f}s")
    print(f"  Generation time:  {stats['gen_time']:.1f}s")
    print(f"  Input tokens:     {stats['input_tokens']}")
    print(f"  Output tokens:    {stats['output_tokens']}")
    print(f"  Throughput:       {stats['tokens_per_sec']:.1f} tokens/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
