"""Option-token readout on a VLM / VLA backbone.

One forward pass, no generation. The prompt ends at the point where the model
would emit its answer; we read the raw logits at that position and keep only
the option-marker tokens. That is the whole mechanism -- everything else in
this file is about making sure the position and the token ids are the right
ones, which is where this kind of readout usually goes wrong silently.

Requires torch + transformers. jev.calibration deliberately does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None


DEFAULT_OPTION_MARKERS = ("A", "B", "C")


@dataclass
class ReadoutResult:
    logits: np.ndarray          # (k,) raw logits at the decision position
    option_token_ids: list[int]
    decision_index: int
    top_token: str | None = None   # what the model would actually have emitted
    meta: dict = field(default_factory=dict)


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "jev.readout needs torch + transformers. "
            "Install with: pip install 'torch' 'transformers>=4.50' pillow"
        )


def resolve_option_token_ids(
    tokenizer,
    markers: tuple[str, ...] = DEFAULT_OPTION_MARKERS,
    leading_space: bool = False,
) -> list[int]:
    """Map each option marker to the single token id the model would emit.

    Raises rather than guessing if a marker is not a single token. A marker
    that splits into two tokens makes the first-token probability mean
    something different for that option than for the others, which silently
    corrupts the comparison -- this is a common way to get a readout that
    "works" but is measuring tokenizer quirks.
    """
    ids: list[int] = []
    for m in markers:
        text = (" " + m) if leading_space else m
        toks = tokenizer.encode(text, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(
                f"option marker {text!r} tokenizes to {len(toks)} tokens ({toks}); "
                "readout requires single-token markers. Try leading_space="
                f"{not leading_space} or different markers."
            )
        ids.append(toks[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"option markers collide in token space: {ids}")
    return ids


class OptionReadout:
    """Single-pass option-token readout over an image-text-to-text model.

    Args:
        model: a loaded AutoModelForImageTextToText (or compatible).
        processor: its AutoProcessor.
        markers: the option letters presented in the prompt.
        leading_space: whether the emitted answer token carries a leading
            space. Depends on the chat template; `autodetect_leading_space`
            settles it empirically.
    """

    def __init__(
        self,
        model,
        processor,
        markers: tuple[str, ...] = DEFAULT_OPTION_MARKERS,
        leading_space: bool = False,
    ):
        _require_torch()
        self.model = model
        self.processor = processor
        self.markers = markers
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.option_token_ids = resolve_option_token_ids(
            self.tokenizer, markers, leading_space
        )
        self.model.eval()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        truncate_layers: int | None = None,
        device: str = "cpu",
        dtype: str = "float32",
        markers: tuple[str, ...] = DEFAULT_OPTION_MARKERS,
        leading_space: bool = False,
    ) -> "OptionReadout":
        """Load a base VLM, optionally truncated to its first N text layers.

        `truncate_layers=16` reproduces exactly what LeRobot's SmolVLA does to
        its backbone (smolvlm_with_expert.py: `text_model.layers[:num_vlm_layers]`,
        default 16 of SmolLM2-360M's 32). The resulting stack is then read out
        through the *untruncated* final norm and lm_head, which SmolVLA keeps
        but never calls. Whether those weights still decode anything from a
        half-depth residual stream is the question E0 exists to answer.
        """
        _require_torch()
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), low_cpu_mem_usage=True
        )
        processor = AutoProcessor.from_pretrained(model_id)
        if truncate_layers is not None:
            truncate_text_layers(model, truncate_layers)
        model.to(device)
        return cls(model, processor, markers=markers, leading_space=leading_space)

    @classmethod
    def from_smolvla(
        cls,
        checkpoint: str = "lerobot/smolvla_base",
        device: str = "cpu",
        markers: tuple[str, ...] = DEFAULT_OPTION_MARKERS,
        leading_space: bool = False,
    ) -> "OptionReadout":
        """Read out through an action-finetuned SmolVLA's own VLM weights.

        SmolVLA freezes lm_head, text_model.norm, and the last two kept text
        layers during action finetuning, so this path is not as degenerate as
        it sounds -- the decoding head and the layers feeding it are the
        original pretrained weights. Everything below them has moved.
        """
        _require_torch()
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(checkpoint)
        wrapper = policy.model.vlm_with_expert
        model = wrapper.vlm.to(device)
        return cls(model, wrapper.processor, markers=markers, leading_space=leading_space)

    # -- the readout -------------------------------------------------------

    def build_prompt(self, question: str, option_texts: list[str], image) -> dict:
        """Chat-template a single-image MCQA turn, stopped before the answer."""
        lines = [question] + [
            f"{m}. {t}" for m, t in zip(self.markers, option_texts, strict=True)
        ]
        content = [{"type": "image"}, {"type": "text", "text": "\n".join(lines)}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return self.processor(text=text, images=[image], return_tensors="pt")

    @torch.no_grad() if torch is not None else (lambda f: f)
    def score(self, image, question: str, option_texts: list[str]) -> ReadoutResult:
        inputs = self.build_prompt(question, option_texts, image)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = self.model(**inputs, use_cache=False)
        # The decision position is the last real token of the prompt. With no
        # padding in a batch of one that is simply -1; assert it so a future
        # batched caller cannot silently read a pad position.
        mask = inputs.get("attention_mask")
        if mask is not None and int(mask[0, -1].item()) != 1:
            raise ValueError(
                "last prompt position is masked; batched/padded input needs "
                "left padding for the decision index to be -1"
            )

        row = out.logits[0, -1, :].float()
        option_logits = row[self.option_token_ids].cpu().numpy()
        top_id = int(row.argmax().item())

        return ReadoutResult(
            logits=option_logits.astype(np.float64),
            option_token_ids=list(self.option_token_ids),
            decision_index=int(out.logits.shape[1] - 1),
            top_token=self.tokenizer.decode([top_id]),
            meta={
                "in_option_mass": float(
                    torch.softmax(row, dim=-1)[self.option_token_ids].sum().item()
                ),
                "n_text_layers": n_text_layers(self.model),
            },
        )

    def autodetect_leading_space(self, image, question: str, option_texts: list[str]):
        """Pick the marker variant the model actually emits.

        Runs the same prompt under both tokenizations and keeps whichever puts
        more probability mass on the option set. If both are near zero the
        model is not answering in the requested format at all, and the readout
        is measuring noise -- which this surfaces instead of hiding.
        """
        best, best_mass = None, -1.0
        for ls in (False, True):
            try:
                ids = resolve_option_token_ids(self.tokenizer, self.markers, ls)
            except ValueError:
                continue
            self.option_token_ids = ids
            mass = self.score(image, question, option_texts).meta["in_option_mass"]
            if mass > best_mass:
                best, best_mass = ls, mass
        if best is None:
            raise ValueError("no usable single-token variant for these markers")
        self.option_token_ids = resolve_option_token_ids(
            self.tokenizer, self.markers, best
        )
        return best, best_mass


# --------------------------------------------------------------------------
# model surgery helpers
# --------------------------------------------------------------------------


def _text_model(model):
    for path in (
        ("model", "text_model"),
        ("model", "language_model"),
        ("language_model", "model"),
    ):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "layers"):
            return obj
    raise AttributeError(
        "could not locate the text decoder stack; pass the module explicitly"
    )


def n_text_layers(model) -> int:
    return len(_text_model(model).layers)


def truncate_text_layers(model, n: int):
    """Keep only the first n text-decoder layers, as SmolVLA does.

    Also updates config.num_hidden_layers so anything that indexes by layer
    count (cache allocation, layer_idx bookkeeping) stays consistent.
    """
    tm = _text_model(model)
    total = len(tm.layers)
    if n > total:
        raise ValueError(f"cannot keep {n} layers, model has {total}")
    tm.layers = tm.layers[:n]
    for cfg in (getattr(tm, "config", None), getattr(model, "config", None)):
        if cfg is None:
            continue
        if hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = n
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
            cfg.text_config.num_hidden_layers = n
    return model
