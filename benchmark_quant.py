#!/usr/bin/env python3
"""Benchmark MLX quantization levels for Qwen3-TTS models.

Tests all quantized variants (4bit, 5bit, 6bit, 8bit, bf16) within each
model family to find the speed/memory sweet spot.

Usage:
    # Run speak-mode families (no ref audio needed)
    python benchmark_quant.py --family 0.6B-CustomVoice --runs 3
    python benchmark_quant.py --family 1.7B-CustomVoice --runs 3

    # Run a specific subset of quantisations
    python benchmark_quant.py --family 0.6B-CustomVoice --quants 4bit 8bit

    # Clone mode (needs reference audio)
    python benchmark_quant.py --family 0.6B-Base --ref-audio sample.wav --ref-text "Hello"

    # Save raw data to JSON
    python benchmark_quant.py --family 1.7B-CustomVoice --output-json results.json

    # Run all speak-mode families
    python benchmark_quant.py --runs 3
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

DEFAULT_TEXT = (
    "Hello! This is a benchmark test for Qwen 3 text to speech. "
    "We are comparing different quantization levels to find "
    "the optimal configuration for Apple Silicon."
)

SPEAKER = "Ryan"
LANG_CODE = "en"

ALL_QUANTS = ["4bit", "5bit", "6bit", "8bit", "bf16"]

# Family definitions: name -> (size, suffix, mode)
FAMILIES = {
    "0.6B-CustomVoice": ("0.6B", "CustomVoice", "speak"),
    "0.6B-Base":        ("0.6B", "Base",        "clone"),
    "1.7B-CustomVoice": ("1.7B", "CustomVoice", "speak"),
    "1.7B-Base":        ("1.7B", "Base",        "clone"),
    "1.7B-VoiceDesign": ("1.7B", "VoiceDesign", "design"),
}

SPEAK_FAMILIES = [k for k, v in FAMILIES.items() if v[2] == "speak"]


def model_id(size, suffix, quant):
    """Build the HuggingFace model ID for an MLX community model."""
    return f"mlx-community/Qwen3-TTS-12Hz-{size}-{suffix}-{quant}"


def _get_peak_memory():
    """Return peak Metal GPU memory in bytes, or None if unavailable."""
    try:
        import mlx.core as mx
        return mx.metal.get_peak_memory()
    except (ImportError, AttributeError):
        return None


def _reset_peak_memory():
    """Reset peak Metal GPU memory counter."""
    try:
        import mlx.core as mx
        mx.metal.reset_peak_memory()
    except (ImportError, AttributeError):
        pass


def benchmark_variant(model_id_str, mode, text, runs, ref_audio=None, ref_text=None):
    """Benchmark a single MLX model variant.

    Returns a dict with load_time, first_run, avg_generation,
    audio_duration, rtf, and peak_memory_mb.
    """
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.generate import generate_audio
    import soundfile as sf

    out_dir = tempfile.mkdtemp(prefix="qwen_tts_qbench_")
    out_file = os.path.join(out_dir, "audio_000.wav")

    # --- Load ---
    _reset_peak_memory()
    t0 = time.perf_counter()
    model = load_model(model_id_str)
    load_time = time.perf_counter() - t0

    # Build generation kwargs
    gen_kwargs = dict(
        text=text, model=model,
        voice=SPEAKER.lower(), lang_code=LANG_CODE,
        output_path=out_dir, verbose=False,
    )
    if mode == "clone":
        gen_kwargs["ref_audio"] = ref_audio
        if ref_text:
            gen_kwargs["ref_text"] = ref_text
    elif mode == "design":
        gen_kwargs["instruct"] = "A warm, clear male voice with a natural tone"

    # --- First generation (warm-up) ---
    _reset_peak_memory()
    t0 = time.perf_counter()
    generate_audio(**gen_kwargs)
    first_run = time.perf_counter() - t0

    data, sr = sf.read(out_file)
    audio_duration = len(data) / sr

    # --- Timed runs ---
    gen_times = []
    _reset_peak_memory()
    for i in range(runs):
        t0 = time.perf_counter()
        generate_audio(**gen_kwargs)
        elapsed = time.perf_counter() - t0
        gen_times.append(elapsed)
        print(f"    Run {i + 1}/{runs}: {elapsed:.2f}s", flush=True)

    avg_gen = sum(gen_times) / len(gen_times)
    peak_mem = _get_peak_memory()
    peak_mem_mb = peak_mem / (1024 * 1024) if peak_mem is not None else None

    # Cleanup
    shutil.rmtree(out_dir, ignore_errors=True)
    del model

    return {
        "model_id": model_id_str,
        "load_time": load_time,
        "first_run": first_run,
        "avg_generation": avg_gen,
        "audio_duration": audio_duration,
        "rtf": avg_gen / audio_duration if audio_duration > 0 else float("inf"),
        "peak_memory_mb": peak_mem_mb,
    }


def print_family_results(family_name, quants, results):
    """Print a per-family results table and highlight the winner."""
    size, suffix, mode = FAMILIES[family_name]
    print()
    print("=" * 90)
    print(f"  {family_name}  (mode: {mode})")
    print("=" * 90)

    has_mem = any(r.get("peak_memory_mb") is not None for r in results)

    header = (
        f"  {'Quant':<8} {'Load':>8} {'1st Gen':>8} "
        f"{'Avg Gen':>8} {'Audio':>7} {'RTF':>6}"
    )
    if has_mem:
        header += f" {'Peak Mem':>10}"
    print(header)

    units = (
        f"  {'':8} {'(s)':>8} {'(s)':>8} "
        f"{'(s)':>8} {'(s)':>7} {'':>6}"
    )
    if has_mem:
        units += f" {'(MB)':>10}"
    print(units)
    print("  " + "-" * (86 if has_mem else 76))

    for quant, r in zip(quants, results):
        if r is None:
            print(f"  {quant:<8} {'FAILED':>8}")
            continue
        line = (
            f"  {quant:<8} "
            f"{r['load_time']:>8.2f} "
            f"{r['first_run']:>8.2f} "
            f"{r['avg_generation']:>8.2f} "
            f"{r['audio_duration']:>7.1f} "
            f"{r['rtf']:>6.2f}"
        )
        if has_mem and r.get("peak_memory_mb") is not None:
            line += f" {r['peak_memory_mb']:>10.0f}"
        print(line)

    print("  " + "-" * (86 if has_mem else 76))

    valid = [(q, r) for q, r in zip(quants, results) if r is not None]
    if valid:
        winner_q, winner_r = min(valid, key=lambda x: x[1]["rtf"])
        print()
        print(f"  >>> Winner: {winner_q}  (RTF {winner_r['rtf']:.2f}, "
              f"avg gen {winner_r['avg_generation']:.2f}s)")
        if winner_r.get("peak_memory_mb") is not None:
            print(f"      Peak memory: {winner_r['peak_memory_mb']:.0f} MB")
        print(f"      Model: {winner_r['model_id']}")
    print()


def print_summary(all_results):
    """Print a final summary with recommended MLX_MODELS dict."""
    print()
    print("=" * 90)
    print("  RECOMMENDED MLX_MODELS (best RTF per family)")
    print("=" * 90)
    print()

    mode_map = {"CustomVoice": "speak", "Base": "clone", "VoiceDesign": "design"}
    recommendations = []

    for family_name, family_results in all_results.items():
        valid = [(q, r) for q, r in family_results if r is not None]
        if not valid:
            continue
        winner_q, winner_r = min(valid, key=lambda x: x[1]["rtf"])
        size, suffix, mode = FAMILIES[family_name]
        recommendations.append((size, mode, winner_r["model_id"]))

    print("MLX_MODELS = {")
    for size, mode, mid in sorted(recommendations):
        print(f'    ("{size}", "{mode:6s}"): "{mid}",')
    print("}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MLX quantization levels for Qwen3-TTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--family", default=None, choices=list(FAMILIES.keys()),
        help="Model family to benchmark. Default: all speak-mode families.",
    )
    parser.add_argument(
        "--quants", nargs="*", default=None,
        help=f"Quantisation levels to test. Choices: {', '.join(ALL_QUANTS)}. "
             "Default: all.",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of timed generation runs (excluding warm-up).",
    )
    parser.add_argument(
        "--text", default=DEFAULT_TEXT,
        help="Text to synthesise for the benchmark.",
    )
    parser.add_argument(
        "--ref-audio", default=None,
        help="Reference audio file (required for clone-mode families).",
    )
    parser.add_argument(
        "--ref-text", default=None,
        help="Transcript of reference audio (used with clone-mode families).",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Save raw benchmark data to a JSON file.",
    )
    args = parser.parse_args()

    # Validate MLX is available
    try:
        import mlx_audio  # noqa: F401
    except ImportError:
        sys.exit(
            "Error: mlx-audio not installed. Install with:\n"
            '  pip install "qwen-tts-cli[mlx]"'
        )

    quants = args.quants or ALL_QUANTS
    for q in quants:
        if q not in ALL_QUANTS:
            sys.exit(f"Error: Unknown quantisation '{q}'. Choices: {', '.join(ALL_QUANTS)}")

    # Determine which families to run
    if args.family:
        families_to_run = [args.family]
    else:
        families_to_run = SPEAK_FAMILIES

    # Validate clone-mode requirements
    for fam in families_to_run:
        _, _, mode = FAMILIES[fam]
        if mode == "clone" and not args.ref_audio:
            sys.exit(
                f"Error: Family '{fam}' is clone mode and requires --ref-audio.\n"
                f"  python benchmark_quant.py --family {fam} --ref-audio sample.wav --ref-text \"...\""
            )

    print("=" * 90)
    print("  Qwen3-TTS MLX Quantization Benchmark")
    print("=" * 90)
    print(f"  Families:  {', '.join(families_to_run)}")
    print(f"  Quants:    {', '.join(quants)}")
    print(f"  Runs:      {args.runs} (+ 1 warm-up)")
    print(f"  Text:      {args.text[:60]}{'...' if len(args.text) > 60 else ''}")
    print()

    all_results = {}  # family -> [(quant, result_or_None), ...]

    for fam in families_to_run:
        size, suffix, mode = FAMILIES[fam]
        print(f"[{fam}]")
        family_results = []

        for quant in quants:
            mid = model_id(size, suffix, quant)
            print(f"  {quant}: {mid}")
            try:
                result = benchmark_variant(
                    mid, mode, args.text, args.runs,
                    ref_audio=args.ref_audio, ref_text=args.ref_text,
                )
                family_results.append((quant, result))
                print(f"    Avg: {result['avg_generation']:.2f}s, "
                      f"RTF: {result['rtf']:.2f}", flush=True)
            except Exception as e:
                print(f"    Error: {e}")
                family_results.append((quant, None))
            print()

        all_results[fam] = family_results
        quants_ran = [q for q, _ in family_results]
        results_ran = [r for _, r in family_results]
        print_family_results(fam, quants_ran, results_ran)

    # Final summary
    if all_results:
        print_summary(all_results)

    # Save JSON
    if args.output_json:
        json_data = {}
        for fam, family_results in all_results.items():
            json_data[fam] = {
                q: r for q, r in family_results
            }
        with open(args.output_json, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"Raw data saved to {args.output_json}")


if __name__ == "__main__":
    main()
