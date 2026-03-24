"""Convert any HuggingFace model to JANG format.

Usage:
    pip install "jang[mlx]"
    python convert_model.py /path/to/source/model -p JANG_4S -o /path/to/output

Profiles:
    JANG_2S  — Smallest (CRITICAL=6, COMPRESS=2) ~2.5 bpw
    JANG_2L  — Small (CRITICAL=8, COMPRESS=2) ~2.1 bpw
    JANG_3M  — Medium-low (CRITICAL=8, COMPRESS=3) ~3.0 bpw
    JANG_4S  — Medium (CRITICAL=6, COMPRESS=4) ~4.2 bpw
    JANG_4M  — Standard (CRITICAL=8, COMPRESS=4) ~4.1 bpw
    JANG_6M  — High quality (CRITICAL=8, COMPRESS=6) ~6.0 bpw

All profiles auto-detect: MoE experts, MLA attention, bfloat16 needs,
FP8 source models, VLM vision encoders, gate routing precision.
"""
import sys

if __name__ == "__main__":
    from jang_tools.convert import convert_model
    import argparse

    parser = argparse.ArgumentParser(description="Convert model to JANG format")
    parser.add_argument("model_path", help="Source model path")
    parser.add_argument("-p", "--profile", default="JANG_4M", help="JANG profile")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-m", "--method", default="mse", choices=["mse", "rtn"],
                       help="Quantization method")
    args = parser.parse_args()

    convert_model(
        model_path=args.model_path,
        output_dir=args.output,
        profile=args.profile,
        quantization_method=args.method,
    )
