#!/usr/bin/env python3
"""Benchmark Qwen3-TTS models: Transformers vs MLX on Apple Silicon.

Compares model load time, generation speed, and real-time factor across
all compatible backends and model sizes.

Usage:
    python benchmark.py                   # Run all available benchmarks
    python benchmark.py --runs 5          # Average over 5 generation runs
    python benchmark.py --text "Custom"   # Use custom benchmark text
    python benchmark.py --models mlx-1.7B-8bit tf-0.6B  # Specific models only
"""

import argparse
import os
import time
import sys

DEFAULT_TEXT = (
    "Hello! This is a benchmark test for Qwen 3 text to speech. "
    "We are comparing different model backends and sizes to find "
    "the optimal configuration for Apple Silicon."
)

SPEAKER = "Ryan"
LANGUAGE = "English"

# Registry of benchmarkable models: key -> (backend, model_id_or_alias, label)
MODEL_REGISTRY = {
    "tf-0.6B": ("transformers", "0.6B", "Transformers 0.6B"),
    "tf-1.7B": ("transformers", "1.7B", "Transformers 1.7B"),
    "mlx-0.6B-6bit": ("mlx", "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit", "MLX 0.6B 6-bit"),
    "mlx-1.7B-4bit": ("mlx", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit", "MLX 1.7B 4-bit"),
}


def _has_mlx_audio():
    try:
        import mlx_audio  # noqa: F401
        return True
    except ImportError:
        return False


def _has_transformers():
    try:
        import qwen_tts  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def benchmark_transformers(model_alias, text, runs):
    """Benchmark a transformers-based Qwen3-TTS model."""
    import torch
    from qwen_tts import Qwen3TTSModel

    model_name = f"Qwen/Qwen3-TTS-12Hz-{model_alias}-CustomVoice"
    device = _detect_device()
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    # --- Load ---
    print(f"  Loading {model_name} on {device}...", flush=True)
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(model_name, device_map=device, dtype=dtype)
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # --- First generation (includes JIT / graph compilation overhead) ---
    print("  Warm-up generation...", flush=True)
    t0 = time.perf_counter()
    wavs, sr = model.generate_custom_voice(
        text=text, speaker=SPEAKER, language=LANGUAGE,
    )
    first_run = time.perf_counter() - t0
    audio_duration = len(wavs[0]) / sr
    print(f"  First gen: {first_run:.2f}s → {audio_duration:.1f}s audio")

    # --- Timed runs ---
    gen_times = []
    for i in range(runs):
        t0 = time.perf_counter()
        wavs, sr = model.generate_custom_voice(
            text=text, speaker=SPEAKER, language=LANGUAGE,
        )
        elapsed = time.perf_counter() - t0
        gen_times.append(elapsed)
        print(f"  Run {i + 1}/{runs}: {elapsed:.2f}s", flush=True)

    avg_gen = sum(gen_times) / len(gen_times)

    # Cleanup
    del model
    if device == "mps":
        torch.mps.empty_cache()

    return {
        "label": f"Transformers {model_alias} ({device})",
        "load_time": load_time,
        "first_run": first_run,
        "avg_generation": avg_gen,
        "audio_duration": audio_duration,
        "rtf": avg_gen / audio_duration if audio_duration > 0 else float("inf"),
    }


def benchmark_mlx(model_id, label, text, runs):
    """Benchmark an MLX-based Qwen3-TTS model."""
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.generate import generate_audio
    import soundfile as sf

    out_dir = "/tmp/_qwen_tts_bench"
    out_file = os.path.join(out_dir, "audio_000.wav")

    # --- Load ---
    print(f"  Loading {model_id} (MLX)...", flush=True)
    t0 = time.perf_counter()
    model = load_model(model_id)
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # --- First generation ---
    print("  Warm-up generation...", flush=True)
    t0 = time.perf_counter()
    generate_audio(
        text=text, model=model,
        voice=SPEAKER.lower(), lang_code="en",
        output_path=out_dir, verbose=False,
    )
    first_run = time.perf_counter() - t0

    data, sr = sf.read(out_file)
    audio_duration = len(data) / sr
    print(f"  First gen: {first_run:.2f}s → {audio_duration:.1f}s audio")

    # --- Timed runs ---
    gen_times = []
    for i in range(runs):
        t0 = time.perf_counter()
        generate_audio(
            text=text, model=model,
            voice=SPEAKER.lower(), lang_code="en",
            output_path=out_dir, verbose=False,
        )
        elapsed = time.perf_counter() - t0
        gen_times.append(elapsed)
        print(f"  Run {i + 1}/{runs}: {elapsed:.2f}s", flush=True)

    avg_gen = sum(gen_times) / len(gen_times)

    # Cleanup
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    del model

    return {
        "label": label,
        "load_time": load_time,
        "first_run": first_run,
        "avg_generation": avg_gen,
        "audio_duration": audio_duration,
        "rtf": avg_gen / audio_duration if audio_duration > 0 else float("inf"),
    }


def print_results(results):
    """Print benchmark results as a formatted table."""
    print()
    print("=" * 80)
    print("  BENCHMARK RESULTS")
    print("=" * 80)

    header = (
        f"  {'Model':<32} {'Load':>8} {'1st Gen':>8} "
        f"{'Avg Gen':>8} {'Audio':>7} {'RTF':>6}"
    )
    units = (
        f"  {'':32} {'(s)':>8} {'(s)':>8} "
        f"{'(s)':>8} {'(s)':>7} {'':>6}"
    )
    print(header)
    print(units)
    print("  " + "-" * 76)

    for r in results:
        print(
            f"  {r['label']:<32} "
            f"{r['load_time']:>8.2f} "
            f"{r['first_run']:>8.2f} "
            f"{r['avg_generation']:>8.2f} "
            f"{r['audio_duration']:>7.1f} "
            f"{r['rtf']:>6.2f}"
        )

    print("  " + "-" * 76)
    print()
    print("  RTF = Real-Time Factor (generation time / audio duration)")
    print("  Lower RTF is better. RTF < 1.0 means faster than real-time.")
    print()

    fastest = min(results, key=lambda r: r["avg_generation"])
    slowest = max(results, key=lambda r: r["avg_generation"])
    speedup = slowest["avg_generation"] / fastest["avg_generation"]

    print(f"  Winner: {fastest['label']}")
    print(f"    Avg generation: {fastest['avg_generation']:.2f}s "
          f"(RTF {fastest['rtf']:.2f})")
    print(f"    {speedup:.1f}x faster than {slowest['label']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-TTS models across backends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--text", default=DEFAULT_TEXT,
        help="Text to synthesise for the benchmark.",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of timed generation runs (excluding warm-up).",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Models to benchmark. Choices: {', '.join(MODEL_REGISTRY)}. "
             "Default: all available.",
    )
    args = parser.parse_args()

    selected = args.models or list(MODEL_REGISTRY.keys())
    has_tf = _has_transformers()
    has_mlx = _has_mlx_audio()

    print("=" * 80)
    print("  Qwen3-TTS Benchmark")
    print("=" * 80)
    print(f"  Text:     {args.text[:70]}{'...' if len(args.text) > 70 else ''}")
    print(f"  Speaker:  {SPEAKER} | Language: {LANGUAGE}")
    print(f"  Runs:     {args.runs} (+ 1 warm-up)")
    print(f"  Backends: transformers={'yes' if has_tf else 'NO'}, "
          f"mlx={'yes' if has_mlx else 'NO'}")
    print()

    results = []

    for name in selected:
        if name not in MODEL_REGISTRY:
            print(f"  Unknown model key: '{name}'")
            print(f"  Choices: {', '.join(MODEL_REGISTRY)}")
            continue

        backend, model_arg, label = MODEL_REGISTRY[name]

        if backend == "transformers" and not has_tf:
            print(f"  Skipping {name}: qwen-tts package not installed")
            continue
        if backend == "mlx" and not has_mlx:
            print(f"  Skipping {name}: mlx-audio not installed "
                  "(pip install mlx-audio)")
            continue

        print(f"[{name}]")
        try:
            if backend == "transformers":
                result = benchmark_transformers(model_arg, args.text, args.runs)
            else:
                result = benchmark_mlx(model_arg, label, args.text, args.runs)
            results.append(result)
        except Exception as e:
            print(f"  Error benchmarking {name}: {e}")

        print()

    if results:
        print_results(results)
    else:
        print("No models could be benchmarked.")
        print("Install qwen-tts (transformers backend) and/or "
              "mlx-audio (MLX backend).")
        sys.exit(1)


if __name__ == "__main__":
    main()
