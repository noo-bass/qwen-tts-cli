# qwen-tts-cli

Whisper-style CLI for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) text-to-speech. One command, instant speech.

## Install

```bash
pip install qwen-tts-cli
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
  -s, --speaker NAME      Speaker voice (default: Ryan)
  -l, --language LANG     Language (default: Auto)
  -i, --instruct TEXT     Style/emotion instruction
  --device DEVICE         Force device: cuda:0, mps, cpu (default: auto)
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

## Device support

The CLI auto-detects your hardware:

| Platform       | Device | Precision  |
|----------------|--------|------------|
| NVIDIA GPU     | cuda   | bfloat16   |
| Apple Silicon  | mps    | float32    |
| CPU            | cpu    | float32    |

## License

Apache-2.0 (same as Qwen3-TTS)
