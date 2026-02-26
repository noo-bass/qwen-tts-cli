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

# Mapping from CLI language names to ISO 639-1 codes (used by MLX backend)
LANG_CODES = {
    "Chinese": "zh", "English": "en", "Japanese": "ja", "Korean": "ko",
    "German": "de", "French": "fr", "Russian": "ru", "Portuguese": "pt",
    "Spanish": "es", "Italian": "it",
}

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
    ("0.6B", "speak"):  "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit",
    ("1.7B", "speak"):  "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
    ("0.6B", "clone"):  "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
    ("1.7B", "clone"):  "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    ("1.7B", "design"): "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
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

    # Auto-upgrade: 0.6B has no design model, use 1.7B
    if model_arg == "0.6B" and mode == "design" and ("1.7B", mode) in MLX_MODELS:
        print("Note: Voice design requires the 1.7B model, upgrading automatically.")
        return MLX_MODELS[("1.7B", mode)]

    available = [f"  {k[0]} ({k[1]} mode)" for k in MLX_MODELS]
    sys.exit(
        f"Error: No MLX model for size '{model_arg}' in '{mode}' mode.\n"
        f"Available MLX models:\n" + "\n".join(available) + "\n"
        f"Or pass a full HuggingFace model ID with -m."
    )


def _has_backend(name):
    """Check if a backend is installed without importing it."""
    import importlib.util
    if name == "mlx":
        return importlib.util.find_spec("mlx_audio") is not None
    if name == "transformers":
        return importlib.util.find_spec("qwen_tts") is not None
    return False


def _auto_backend():
    """Pick the best available backend, or exit with a helpful message."""
    has_mlx = _has_backend("mlx")
    has_tf = _has_backend("transformers")

    if has_mlx:
        return "mlx"
    if has_tf:
        return "transformers"

    sys.exit(
        "Error: No backend installed. Install one:\n"
        '  pip install "qwen-tts-cli[mlx]"           # Apple Silicon (recommended)\n'
        '  pip install "qwen-tts-cli[transformers]"   # CUDA / CPU\n'
    )


def _detect_device():
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _read_text(args):
    if args.file:
        try:
            with open(args.file) as f:
                return f.read().strip()
        except FileNotFoundError:
            sys.exit(f"Error: File not found: {args.file}")
    if not args.text:
        if not sys.stdin.isatty():
            return sys.stdin.read().strip()
        return None
    if args.text == ["-"]:
        return sys.stdin.read().strip()
    return " ".join(args.text)


def _load_model(model_name, device):
    import torch
    try:
        from qwen_tts_cli._compat import patch_transformers_compat
        patch_transformers_compat()
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        sys.exit(
            'Error: Transformers backend not installed.\n'
            '  pip install "qwen-tts-cli[transformers]"\n'
        )

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
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError:
        sys.exit(
            'Error: MLX backend not installed.\n'
            '  pip install "qwen-tts-cli[mlx]"\n'
        )
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


def _mlx_gen_kwargs(mode, text, language, speaker, instruct,
                    clone_audio, ref_text):
    """Build kwargs for model.generate() calls (no 'model' key)."""
    lang_code = LANG_CODES.get(language, "en") if language != "Auto" else "en"

    kwargs = dict(
        text=text,
        voice=speaker.lower(),
        lang_code=lang_code,
        verbose=False,
    )

    if mode == "speak":
        if instruct:
            kwargs["instruct"] = instruct
    elif mode == "clone":
        kwargs["ref_audio"] = clone_audio
        if ref_text:
            kwargs["ref_text"] = ref_text
    elif mode == "design":
        kwargs["instruct"] = instruct

    return kwargs


def _generate_mlx(model, mode, text, language, speaker, instruct,
                   clone_audio, ref_text, output_path):
    """Generate audio using the MLX backend. Writes directly to output_path."""
    import shutil
    import tempfile
    from mlx_audio.tts.generate import generate_audio

    kwargs = _mlx_gen_kwargs(mode, text, language, speaker, instruct,
                             clone_audio, ref_text)
    kwargs["model"] = model

    # mlx_audio writes to {output_path}/audio_000.wav (directory-based output)
    tmp_dir = tempfile.mkdtemp(prefix="qwen_tts_mlx_")
    kwargs["output_path"] = tmp_dir

    generate_audio(**kwargs)

    # Move generated file to the user's output path
    generated = os.path.join(tmp_dir, "audio_000.wav")
    shutil.move(generated, output_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _split_sentences(text, chunk_size=1):
    """Split text into sentence chunks, grouping chunk_size sentences together."""
    import re
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]
    # Group sentences into chunks
    return [" ".join(sentences[i:i + chunk_size])
            for i in range(0, len(sentences), chunk_size)]


def _split_paragraphs(text):
    """Split text on blank lines into paragraphs."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _stream_mlx_tokens(model, mode, text, language, speaker, instruct,
                       clone_audio, ref_text, streaming_interval):
    """Stream audio in real-time using token-level streaming."""
    kwargs = _mlx_gen_kwargs(mode, text, language, speaker, instruct,
                             clone_audio, ref_text)
    kwargs["stream"] = True
    kwargs["streaming_interval"] = streaming_interval
    # Treat the whole text as one segment — don't re-split on newlines
    kwargs["split_pattern"] = ""

    lang_code = LANG_CODES.get(language, "en") if language != "Auto" else "en"
    print(f"\033[94mText:\033[0m {text}")
    print(f"\033[94mVoice:\033[0m {speaker.lower()}")
    print(f"\033[94mSpeed:\033[0m 1.0x")
    print(f"\033[94mLanguage:\033[0m {lang_code}")
    print()

    player = _StreamPlayer(sample_rate=model.sample_rate)
    try:
        for result in model.generate(**kwargs):
            player.queue(result.audio)
    finally:
        player.finish()


class _StreamPlayer:
    """Audio player that outputs silence on buffer underrun instead of stopping.

    Unlike mlx_audio's AudioPlayer, this keeps the sound stream running
    continuously, avoiding audio loss from stop/restart cycles between chunks.
    """

    def __init__(self, sample_rate=24000):
        import sounddevice as sd
        from collections import deque
        from threading import Event, Lock

        self._lock = Lock()
        self._buffer = deque()
        self._finished = False
        self._done = Event()

        self._stream = sd.OutputStream(
            samplerate=sample_rate, channels=1,
            callback=self._callback, blocksize=2048,
        )
        self._stream.start()

    def _callback(self, outdata, frames, _time, _status):
        import sounddevice as sd
        filled = 0
        with self._lock:
            while filled < frames and self._buffer:
                chunk = self._buffer[0]
                n = min(frames - filled, len(chunk))
                outdata[filled:filled + n, 0] = chunk[:n]
                filled += n
                if n == len(chunk):
                    self._buffer.popleft()
                else:
                    self._buffer[0] = chunk[n:]

        if filled < frames:
            outdata[filled:] = 0
            if self._finished and not self._buffer:
                self._done.set()
                raise sd.CallbackStop()

    def queue(self, audio):
        import numpy as np
        with self._lock:
            self._buffer.append(np.asarray(audio))

    def finish(self):
        """Signal all audio is queued, wait for playback to complete."""
        import sounddevice as sd
        self._finished = True
        with self._lock:
            if not self._buffer:
                self._done.set()
        self._done.wait()
        sd.sleep(150)
        self._stream.stop()
        self._stream.close()


def _stream_mlx_sentences(model, mode, text, language, speaker, instruct,
                          clone_audio, ref_text, chunks,
                          token_stream=False):
    """Stream audio chunk-by-chunk with pipelined playback."""
    import numpy as np

    lang_code = LANG_CODES.get(language, "en") if language != "Auto" else "en"
    print(f"\033[94mText:\033[0m {text}")
    print(f"\033[94mVoice:\033[0m {speaker.lower()}")
    print(f"\033[94mSpeed:\033[0m 1.0x")
    print(f"\033[94mLanguage:\033[0m {lang_code}")
    print()

    player = _StreamPlayer(sample_rate=model.sample_rate)
    # Short silence between sentences for natural pacing
    silence = np.zeros(int(model.sample_rate * 0.15), dtype=np.float32)

    try:
        for i, chunk in enumerate(chunks):
            kwargs = _mlx_gen_kwargs(mode, chunk, language, speaker,
                                     instruct, clone_audio, ref_text)
            if token_stream:
                kwargs["stream"] = True
                kwargs["streaming_interval"] = 2.0
            print(f"  [{i + 1}/{len(chunks)}] {chunk}")

            for result in model.generate(**kwargs):
                player.queue(result.audio)

            # Add silence gap between chunks (not after the last one)
            if i < len(chunks) - 1:
                player.queue(silence)
    finally:
        player.finish()


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
    parser.add_argument("-f", "--file", default=None,
                        help="Read text from a file instead of the command line.")
    parser.add_argument("-o", "--output", default="output.wav",
                        help="Output audio file path.")
    parser.add_argument("-m", "--model", default="0.6B",
                        help="Model size (0.6B, 1.7B) or full HuggingFace model ID.")
    parser.add_argument("-b", "--backend", default=None,
                        choices=["transformers", "mlx"],
                        help="Inference backend (auto-detected if omitted).")
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

    streaming = parser.add_argument_group("streaming (MLX backend only)")
    streaming.add_argument("--stream", action="store_true",
                           help="Stream audio playback in real-time (token-level). "
                                "Combine with --chunk-sentences or --chunk-paragraphs "
                                "for hybrid mode (token streaming within each chunk).")
    streaming.add_argument("--stream-interval", type=float, default=2.0,
                           help="Seconds of audio per token-level streaming chunk.")
    streaming.add_argument("--chunk-sentences", type=int, default=None, metavar="N",
                           help="Stream in chunks of N sentences.")
    streaming.add_argument("--chunk-paragraphs", action="store_true",
                           help="Stream in paragraph chunks (split on blank lines).")

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
    backend = args.backend or _auto_backend()

    if (args.stream or args.chunk_sentences or args.chunk_paragraphs) and backend != "mlx":
        parser.error("Streaming options require the MLX backend (-b mlx).")

    if backend == "mlx":
        model_name = _resolve_mlx_model(args.model, mode)
        print(f"Loading {model_name} (MLX backend)...")
        model = _load_model_mlx(model_name)

        chunking = args.chunk_sentences or args.chunk_paragraphs
        if args.stream or chunking:
            if chunking:
                if args.chunk_paragraphs:
                    chunks = _split_paragraphs(text)
                    label = "paragraph"
                else:
                    chunks = _split_sentences(text, args.chunk_sentences)
                    label = f"{args.chunk_sentences} sentence(s)"
                hybrid = " + token streaming" if args.stream else ""
                print(f"Streaming speech ({label} chunks{hybrid})...")
                _stream_mlx_sentences(
                    model, mode, text, args.language,
                    speaker, args.instruct, args.clone, args.ref_text,
                    chunks=chunks,
                    token_stream=args.stream,
                )
            else:
                print(f"Streaming speech (token-level, {args.stream_interval}s interval)...")
                _stream_mlx_tokens(
                    model, mode, text, args.language,
                    speaker, args.instruct, args.clone, args.ref_text,
                    args.stream_interval,
                )
            return

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
