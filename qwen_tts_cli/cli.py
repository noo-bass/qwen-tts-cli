import argparse
import os
import subprocess
import sys

SPEAKERS = {
    "Vivian":   "Bright young female (Chinese)",
    "Serena":   "Warm gentle female (Chinese)",
    "Uncle_Fu": "Seasoned mellow male (Chinese)",
    "Dylan":    "Clear natural male (Chinese, Beijing)",
    "Eric":     "Lively bright male (Chinese, Sichuan)",
    "Ryan":     "Dynamic rhythmic male (English)",
    "Aiden":    "Sunny clear male (English)",
    "Ono_Anna": "Playful light female (Japanese)",
    "Sohee":    "Warm emotional female (Korean)",
}

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

MODEL_ALIASES = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B",
}

MODE_SUFFIXES = {
    "speak":  "CustomVoice",
    "clone":  "Base",
    "design": "VoiceDesign",
}

# MLX model mapping: (size_alias, mode) -> HuggingFace model ID
MLX_MODELS = {
    ("1.7B", "speak"): "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    ("0.6B", "clone"): "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
    ("1.7B", "clone"): "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
}


def _resolve_model(model_arg, mode):
    if "/" in model_arg:
        return model_arg

    base = MODEL_ALIASES.get(model_arg, model_arg)

    if mode == "design" and "0.6B" in base:
        base = MODEL_ALIASES["1.7B"]
        print("Note: Voice design requires the 1.7B model, upgrading automatically.")

    return f"{base}-{MODE_SUFFIXES[mode]}"


def _resolve_mlx_model(model_arg, mode):
    """Resolve a model alias to an MLX community model ID."""
    if "/" in model_arg:
        return model_arg

    key = (model_arg, mode)
    if key in MLX_MODELS:
        return MLX_MODELS[key]

    # Auto-upgrade: only 1.7B CustomVoice exists for MLX speak mode
    if model_arg == "0.6B" and mode == "speak" and ("1.7B", mode) in MLX_MODELS:
        print("Note: MLX backend only has the 1.7B CustomVoice model, upgrading automatically.")
        return MLX_MODELS[("1.7B", mode)]

    available = [f"  {k[0]} ({k[1]} mode)" for k in MLX_MODELS]
    sys.exit(
        f"Error: No MLX model for size '{model_arg}' in '{mode}' mode.\n"
        f"Available MLX models:\n" + "\n".join(available) + "\n"
        f"Or pass a full HuggingFace model ID with -m."
    )


def _detect_device():
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _read_text(args):
    if not args.text:
        if not sys.stdin.isatty():
            return sys.stdin.read().strip()
        return None
    if args.text == ["-"]:
        return sys.stdin.read().strip()
    return " ".join(args.text)


def _load_model(model_name, device):
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    kwargs = dict(device_map=device, dtype=dtype)

    if device.startswith("cuda"):
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

    return Qwen3TTSModel.from_pretrained(model_name, **kwargs)


def _load_model_mlx(model_name):
    from mlx_audio.tts.utils import load_model

    return load_model(model_name)


def _generate(model, mode, text, language, speaker, instruct, clone_audio, ref_text):
    lang = language if language != "Auto" else None

    if mode == "speak":
        kwargs = dict(text=text, language=lang, speaker=speaker)
        if instruct:
            kwargs["instruct"] = instruct
        return model.generate_custom_voice(**kwargs)

    if mode == "clone":
        return model.generate_voice_clone(
            text=text, language=lang,
            ref_audio=clone_audio, ref_text=ref_text,
        )

    return model.generate_voice_design(
        text=text, language=lang, instruct=instruct,
    )


def _generate_mlx(model, mode, text, language, speaker, instruct,
                   clone_audio, ref_text, output_path):
    """Generate audio using the MLX backend. Writes directly to output_path."""
    from mlx_audio.tts.generate import generate_audio

    prefix = os.path.splitext(output_path)[0]
    lang = language if language != "Auto" else None

    kwargs = dict(model=model, text=text, file_prefix=prefix)

    if mode == "speak":
        kwargs["speaker"] = speaker
        if lang:
            kwargs["language"] = lang
        if instruct:
            kwargs["instruct"] = instruct
    elif mode == "clone":
        kwargs["ref_audio"] = clone_audio
        if ref_text:
            kwargs["ref_text"] = ref_text
        if lang:
            kwargs["language"] = lang
    elif mode == "design":
        kwargs["instruct"] = instruct
        if lang:
            kwargs["language"] = lang

    generate_audio(**kwargs)


def _play(path):
    if sys.platform == "darwin":
        subprocess.run(["afplay", path])
    elif sys.platform == "win32":
        os.startfile(path)
    else:
        for cmd in [["aplay"], ["paplay"], ["ffplay", "-nodisp", "-autoexit"]]:
            try:
                subprocess.run([*cmd, path])
                return
            except FileNotFoundError:
                continue
        print("No audio player found. Install aplay, paplay, or ffplay to auto-play.")


def _build_parser():
    from qwen_tts_cli import __version__

    parser = argparse.ArgumentParser(
        prog="qwen-tts",
        description="Generate speech from text using Qwen3-TTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "text", nargs="*",
        help='Text to speak. Use "-" to read from stdin.',
    )
    parser.add_argument("-o", "--output", default="output.wav",
                        help="Output audio file path.")
    parser.add_argument("-m", "--model", default="0.6B",
                        help="Model size (0.6B, 1.7B) or full HuggingFace model ID.")
    parser.add_argument("-b", "--backend", default="transformers",
                        choices=["transformers", "mlx"],
                        help="Inference backend.")
    parser.add_argument("-s", "--speaker", default="Ryan",
                        help=f"Speaker voice. Choices: {', '.join(SPEAKERS)}")
    parser.add_argument("-l", "--language", default="Auto",
                        help=f"Language. Choices: {', '.join(LANGUAGES)}")
    parser.add_argument("-i", "--instruct", default=None,
                        help='Style/emotion instruction, e.g. "Speak in a whisper".')
    parser.add_argument("--device", default=None,
                        help="Device: cuda:0, mps, cpu. Auto-detected if omitted."
                             " (transformers backend only)")
    parser.add_argument("--play", action=argparse.BooleanOptionalAction, default=None,
                        help="Play audio after generation.")

    clone = parser.add_argument_group("voice cloning")
    clone.add_argument("--clone", metavar="AUDIO", default=None,
                       help="Reference audio file for voice cloning (switches to clone mode).")
    clone.add_argument("--ref-text", default=None,
                       help="Transcript of the reference audio.")

    design = parser.add_argument_group("voice design")
    design.add_argument("--design", action="store_true",
                        help="Use voice design mode (describe a voice with --instruct).")

    parser.add_argument("--list-speakers", action="store_true",
                        help="List available speakers and exit.")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")

    return parser


def cli():
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_speakers:
        print("Available speakers:")
        for name, desc in SPEAKERS.items():
            print(f"  {name:12s} {desc}")
        return

    text = _read_text(args)
    if not text:
        parser.error("Please provide text to speak.")

    mode = "clone" if args.clone else "design" if args.design else "speak"

    if mode == "clone" and not args.ref_text:
        parser.error("--ref-text is required when using --clone.")
    if mode == "design" and not args.instruct:
        parser.error("--instruct is required when using --design.")

    speaker = next((k for k in SPEAKERS if k.lower() == args.speaker.lower()), args.speaker)
    backend = args.backend

    if backend == "mlx":
        model_name = _resolve_mlx_model(args.model, mode)
        print(f"Loading {model_name} (MLX backend)...")
        model = _load_model_mlx(model_name)

        print("Generating speech...")
        _generate_mlx(
            model, mode, text, args.language,
            speaker, args.instruct, args.clone, args.ref_text,
            args.output,
        )
    else:
        device = args.device or _detect_device()
        model_name = _resolve_model(args.model, mode)

        print(f"Loading {model_name} on {device}...")
        model = _load_model(model_name, device)

        print("Generating speech...")
        wavs, sr = _generate(
            model, mode, text, args.language,
            speaker, args.instruct, args.clone, args.ref_text,
        )

        import soundfile as sf
        sf.write(args.output, wavs[0], sr)

    print(f"Saved to {args.output}")

    if args.play is not False:
        _play(args.output)


if __name__ == "__main__":
    cli()
