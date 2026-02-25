# qwen-tts-cli

Whisper-style CLI for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) text-to-speech. One command, instant speech.

## Install

```bash
# Apple Silicon (recommended for Mac — 6x faster)
pip install "qwen-tts-cli[mlx]"

# CUDA / CPU
pip install "qwen-tts-cli[transformers]"
```

## Usage

```bash
# Just speak
qwen-tts "Hello, world!"

# Choose a speaker and style
qwen-tts "I can't believe it!" --speaker Aiden --instruct "Speak with excitement"

# Save to a specific file
qwen-tts "Good morning." -o greeting.wav

# Use the larger model
qwen-tts "Higher quality voice." --model 1.7B

# Force a specific backend (auto-detected by default)
qwen-tts "Fast on Mac!" --backend mlx

# Clone a voice from a 3-second sample
qwen-tts "Now I sound like someone else." --clone reference.wav --ref-text "Transcript of the reference audio."

# Design a voice from a description
qwen-tts "Hi there!" --design --instruct "A warm, deep male voice with a calm tone"

# Read from stdin
echo "Pipe text in" | qwen-tts -

# List available speakers
qwen-tts --list-speakers
```

## Options

```
positional arguments:
  text                    Text to speak. Use "-" to read from stdin.

options:
  -o, --output FILE       Output audio file (default: output.wav)
  -m, --model SIZE        Model: 0.6B, 1.7B, or full HF ID (default: 0.6B)
  -b, --backend BACKEND   Inference backend: transformers, mlx (default: auto)
  -s, --speaker NAME      Speaker voice (default: Ryan)
  -l, --language LANG     Language (default: Auto)
  -i, --instruct TEXT     Style/emotion instruction
  --device DEVICE         Force device: cuda:0, mps, cpu (default: auto, transformers only)
  --play / --no-play      Play audio after generation (default: on for macOS)
  --list-speakers         List available speakers and exit

voice cloning:
  --clone AUDIO           Reference audio for voice cloning
  --ref-text TEXT         Transcript of reference audio

voice design:
  --design                Design a voice using --instruct description
```

## Speakers

| Speaker   | Description                       | Language         |
|-----------|-----------------------------------|------------------|
| Ryan      | Dynamic rhythmic male             | English          |
| Aiden     | Sunny clear male                  | English          |
| Vivian    | Bright young female               | Chinese          |
| Serena    | Warm gentle female                | Chinese          |
| Uncle_Fu  | Seasoned mellow male              | Chinese          |
| Dylan     | Clear natural male                | Chinese (Beijing)|
| Eric      | Lively bright male                | Chinese (Sichuan)|
| Ono_Anna  | Playful light female              | Japanese         |
| Sohee     | Warm emotional female             | Korean           |

## Supported languages

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.

## Backends

### Transformers (default)

Uses PyTorch + HuggingFace Transformers. Works on all platforms.

| Platform       | Device | Precision  |
|----------------|--------|------------|
| NVIDIA GPU     | cuda   | bfloat16   |
| Apple Silicon  | mps    | float32    |
| CPU            | cpu    | float32    |

### MLX (Apple Silicon)

Uses [mlx-audio](https://github.com/lucasnewman/mlx-audio) with quantized models from [mlx-community](https://huggingface.co/mlx-community) for native Apple Silicon acceleration. All modes (speak, clone, design) are supported.

```bash
qwen-tts "Hello!" --backend mlx
qwen-tts "Hello!" --backend mlx --model 0.6B              # smaller, faster
qwen-tts "Hi!" --backend mlx --design --instruct "warm"   # voice design
```

| Size | Mode | Quant | HuggingFace ID |
|------|------|-------|----------------|
| 0.6B | speak  | 6-bit | `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit` |
| 0.6B | clone  | 4-bit | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit` |
| 1.7B | speak  | 4-bit | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit` |
| 1.7B | clone  | 8-bit | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` |
| 1.7B | design | 8-bit | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` |

Additional quantization variants (4bit, 5bit, 6bit, 8bit, bf16) are available on HuggingFace for all families. Use `benchmark_quant.py` to find the best variant for your hardware.

## Benchmark (Apple Silicon)

Tested on a 16GB M1 MacBook Pro with the same input text (~14s of audio output):

| Model | Load | Avg Gen | RTF |
|-------|------|---------|-----|
| Transformers 0.6B (mps) | 10.6s | 61.4s | 4.36 |
| Transformers 1.7B (mps) | 85.0s | 117.7s | 8.08 |
| **MLX 1.7B 8-bit** | **2.3s** | **10.2s** | **1.00** |

MLX is **6x faster** than the equivalent transformers model while using less memory. RTF (real-time factor) of 1.0 means generation runs at real-time speed.

## License

Apache-2.0 (same as Qwen3-TTS)
