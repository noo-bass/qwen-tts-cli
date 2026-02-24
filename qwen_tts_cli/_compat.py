"""Compatibility patches for qwen-tts with transformers >= 5.0.

The upstream qwen-tts package pins transformers == 4.57.3, but mlx-audio
requires transformers >= 5.0. Two small API changes in transformers 5.0
break qwen-tts at model load time:

1. ROPE_INIT_FUNCTIONS no longer includes a 'default' key.
2. AutoProcessor.from_pretrained() no longer accepts fix_mistral_regex.

This module applies targeted monkey-patches so both packages can coexist
on transformers 5.0+. Call ``patch_transformers_compat()`` before loading
any qwen-tts models.
"""

import importlib.metadata

_patched = False


def _needs_patch() -> bool:
    """Return True if transformers >= 5.0 is installed (where the breaking changes landed)."""
    try:
        version = importlib.metadata.version("transformers")
        major = int(version.split(".")[0])
        return major >= 5
    except Exception:
        return False


def patch_transformers_compat() -> None:
    """Apply compatibility patches if needed. Safe to call multiple times."""
    global _patched
    if _patched or not _needs_patch():
        return

    _patch_rope_init_functions()
    _patch_qwen_tts_from_pretrained()
    _patched = True


def _patch_rope_init_functions() -> None:
    """Add 'default' back to ROPE_INIT_FUNCTIONS for transformers 5.0+."""
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _patch_qwen_tts_from_pretrained() -> None:
    """Remove fix_mistral_regex kwarg from qwen-tts model loading."""
    try:
        import qwen_tts.inference.qwen3_tts_model as _qmod
    except ImportError:
        return

    from transformers import AutoConfig, AutoModel, AutoProcessor
    from qwen_tts.core.models import (
        Qwen3TTSConfig,
        Qwen3TTSForConditionalGeneration,
        Qwen3TTSProcessor,
    )

    @classmethod
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, Qwen3TTSForConditionalGeneration):
            raise TypeError(
                f"AutoModel returned {type(model)}, expected Qwen3TTSForConditionalGeneration."
            )

        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)

        generate_defaults = model.generate_config
        return cls(model=model, processor=processor, generate_defaults=generate_defaults)

    _qmod.Qwen3TTSModel.from_pretrained = _patched_from_pretrained
