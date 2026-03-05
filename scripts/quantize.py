#!/usr/bin/env python3
"""
AWQ Quantization Script for Qwen3-4B-Instruct

Usage:
    python quantize.py --model_path Qwen/Qwen3-4B-Instruct --output_path ./qwen3-4b-instruct-awq

For all options:
    python quantize.py --help
"""

import argparse
import os
import time
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize a HuggingFace model to AWQ INT4 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic quantization
  python quantize.py --model_path Qwen/Qwen3-4B-Instruct --output_path ./qwen3-4b-awq

  # Custom calibration
  python quantize.py --model_path Qwen/Qwen3-4B-Instruct --output_path ./qwen3-4b-awq \\
      --calib_data wikitext --calib_size 256

  # GEMV version (optimized for single-token generation)
  python quantize.py --model_path Qwen/Qwen3-4B-Instruct --output_path ./qwen3-4b-awq \\
      --version GEMV
        """,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace model ID or local path (e.g., 'Qwen/Qwen3-4B-Instruct')",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--w_bit",
        type=int,
        default=4,
        choices=[4],
        help="Quantization bit-width (default: 4)",
    )
    parser.add_argument(
        "--q_group_size",
        type=int,
        default=128,
        help="Group size for quantization (default: 128)",
    )
    parser.add_argument(
        "--zero_point",
        type=bool,
        default=True,
        help="Use asymmetric quantization with zero-point (default: True)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="GEMM",
        choices=["GEMM", "GEMV"],
        help="Kernel version: GEMM (batched) or GEMV (single-token) (default: GEMM)",
    )
    parser.add_argument(
        "--calib_data",
        type=str,
        default="wikitext",
        help="Calibration dataset name or path (default: 'wikitext')",
    )
    parser.add_argument(
        "--calib_size",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Model dtype for loading (default: float16)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Import here to avoid slow import when just checking --help
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    # Print configuration
    print("=" * 60)
    print("AWQ Quantization Configuration")
    print("=" * 60)
    print(f"  Model:           {args.model_path}")
    print(f"  Output:          {args.output_path}")
    print(f"  Bit-width:       {args.w_bit}")
    print(f"  Group size:      {args.q_group_size}")
    print(f"  Zero-point:      {args.zero_point}")
    print(f"  Kernel version:  {args.version}")
    print(f"  Calibration:     {args.calib_data} ({args.calib_size} samples)")
    print(f"  Dtype:           {args.dtype}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:             {torch.cuda.get_device_name(0)}")
        print(
            f"  GPU memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
        )
    print("=" * 60)

    # Quantization config
    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version,
    }

    # Load model
    print("\n[1/3] Loading model...")
    start_time = time.time()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    load_time = time.time() - start_time
    print(f"    Model loaded in {load_time:.1f}s")

    # Quantize
    print("\n[2/3] Running AWQ quantization...")
    start_time = time.time()

    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=args.calib_data,
    )

    quant_time = time.time() - start_time
    print(f"    Quantization completed in {quant_time:.1f}s")

    # Save
    print(f"\n[3/3] Saving quantized model to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_quantized(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    # Report
    total_size = sum(
        os.path.getsize(os.path.join(args.output_path, f))
        for f in os.listdir(args.output_path)
        if os.path.isfile(os.path.join(args.output_path, f))
    )

    print("\n" + "=" * 60)
    print("Quantization Complete!")
    print("=" * 60)
    print(f"  Output directory:  {args.output_path}")
    print(f"  Total size:        {total_size / 1e9:.2f} GB")
    print(f"  Total time:        {load_time + quant_time:.1f}s")
    print(f"  Files:")
    for f in sorted(os.listdir(args.output_path)):
        fpath = os.path.join(args.output_path, f)
        if os.path.isfile(fpath):
            fsize = os.path.getsize(fpath)
            print(f"    {f:40s} {fsize / 1e6:>8.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
