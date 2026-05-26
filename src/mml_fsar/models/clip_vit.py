"""CLIP ViT-B/16 visual backbone with MML-FSAR temporal adapters."""

from __future__ import annotations

import gzip
import hashlib
import html
import os
import urllib.request
import warnings
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_OPENAI_CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
DEFAULT_CLIP_CACHE_DIR = "~/.cache/clip"
_OPENAI_CLIP_MODEL_URLS = {
    DEFAULT_OPENAI_CLIP_MODEL_ID: (
        "https://openaipublic.azureedge.net/clip/models/"
        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/"
        "ViT-B-16.pt"
    ),
    "ViT-B/16": (
        "https://openaipublic.azureedge.net/clip/models/"
        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/"
        "ViT-B-16.pt"
    ),
    "ViT-B-16": (
        "https://openaipublic.azureedge.net/clip/models/"
        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/"
        "ViT-B-16.pt"
    ),
}

CLIP_INPUT_RESOLUTION = 224
CLIP_PATCH_SIZE = 16
CLIP_VIT_B16_WIDTH = 768
CLIP_VIT_B16_LAYERS = 12
CLIP_VIT_B16_HEADS = 12
CLIP_PROJECTION_DIM = 512
CLIP_CONTEXT_LENGTH = 77
CLIP_VOCAB_SIZE = 49408
CLIP_TEXT_WIDTH = 512
CLIP_TEXT_LAYERS = 12
CLIP_TEXT_HEADS = 8
CLIP_BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"
DEFAULT_NUM_FRAMES = 8
DEFAULT_ADAPTER_MLP_RATIO = 0.75

LOW_STAGE_LEVEL = 0
MID_STAGE_LEVEL = 1
LOW_STAGE_TEMPORAL_SEGMENTS = 4
MID_STAGE_TEMPORAL_SEGMENTS = 2
STAGE_BLOCK_SIZE = 4
LOW_STAGE_FEATURE_LAYER = 4
MID_STAGE_FEATURE_LAYER = 8
TRAINABLE_PARAMETER_PREFIXES = ("fc.",)
TRAINABLE_PARAMETER_PARTS = (".adapter_plus.adapter.",)

try:
    import ftfy
except Exception:  # pragma: no cover - optional text-cleaning dependency
    ftfy = None

try:
    import regex as regex_re
except Exception:  # pragma: no cover - fallback only for import-time smoke tests
    import re as regex_re

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange
except Exception:  # pragma: no cover - depends on the local smoke-test env
    torch = None
    nn = None
    F = None
    rearrange = None
else:
    if not hasattr(torch, "as_tensor") or not hasattr(torch, "nn"):
        torch = None
        nn = None
        F = None
        rearrange = None


def is_torch_available() -> bool:
    return all(dependency is not None for dependency in (torch, nn, F, rearrange))


def require_torch() -> None:
    if not is_torch_available():
        raise ImportError("CLIP ViT backbone requires PyTorch and einops.")


def resolve_clip_checkpoint_path(
    clip_checkpoint_path: str | Path | None = None,
    clip_model_id: str = DEFAULT_OPENAI_CLIP_MODEL_ID,
    download_root: str | Path | None = None,
) -> Path:
    """Resolve a required OpenAI CLIP checkpoint path.

    A configured local checkpoint always wins. Otherwise, the OpenAI CLIP
    ViT-B/16 checkpoint is downloaded into the cache used by the original CLIP
    loader pattern.
    """

    if clip_checkpoint_path is not None:
        checkpoint_path = Path(clip_checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CLIP checkpoint file not found: {checkpoint_path}")
        return checkpoint_path

    if clip_model_id not in _OPENAI_CLIP_MODEL_URLS:
        available = ", ".join(sorted(_OPENAI_CLIP_MODEL_URLS))
        raise ValueError(
            f"Unsupported CLIP model id: {clip_model_id}. Available models: {available}"
        )
    return _download_clip_checkpoint(_OPENAI_CLIP_MODEL_URLS[clip_model_id], download_root)


def _download_clip_checkpoint(url: str, download_root: str | Path | None = None) -> Path:
    root = Path(
        download_root
        or os.environ.get("MML_FSAR_CLIP_ROOT")
        or DEFAULT_CLIP_CACHE_DIR
    ).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / Path(url).name
    expected_sha256 = url.rstrip("/").split("/")[-2]

    if checkpoint_path.exists() and not checkpoint_path.is_file():
        raise RuntimeError(f"{checkpoint_path} exists and is not a regular file.")
    if checkpoint_path.is_file():
        if _sha256(checkpoint_path) == expected_sha256:
            return checkpoint_path
        warnings.warn(
            f"{checkpoint_path} exists, but its SHA256 checksum does not match; "
            "re-downloading the file.",
            stacklevel=2,
        )

    try:
        with urllib.request.urlopen(url) as source, checkpoint_path.open("wb") as output:
            while True:
                chunk = source.read(8192)
                if not chunk:
                    break
                output.write(chunk)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the required OpenAI CLIP checkpoint. "
            "Set model.clip_checkpoint_path to a local ViT-B/16 .pt file, "
            "or set MML_FSAR_CLIP_ROOT to a cache directory containing ViT-B-16.pt."
        ) from exc

    if _sha256(checkpoint_path) != expected_sha256:
        raise RuntimeError(
            f"Downloaded CLIP checkpoint checksum does not match: {checkpoint_path}"
        )
    return checkpoint_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache()
def _default_bpe_path() -> str:
    return str(Path(__file__).resolve().parent / "assets" / CLIP_BPE_FILENAME)


@lru_cache()
def _bytes_to_unicode() -> dict[int, str]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = byte_values[:]
    offset = 0
    for byte in range(2**8):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_values.append(2**8 + offset)
            offset += 1
    return dict(zip(byte_values, [chr(value) for value in unicode_values]))


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs = set()
    previous = word[0]
    for current in word[1:]:
        pairs.add((previous, current))
        previous = current
    return pairs


def _basic_clean(text: str) -> str:
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text: str) -> str:
    text = regex_re.sub(r"\s+", " ", text)
    return text.strip()


class SimpleTokenizer:
    """OpenAI CLIP BPE tokenizer used by the local text encoder."""

    def __init__(self, bpe_path: str = _default_bpe_path()):
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merge_pairs = [tuple(merge.split()) for merge in merges]
        vocab = list(_bytes_to_unicode().values())
        vocab = vocab + [value + "</w>" for value in vocab]
        for merge in merge_pairs:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {value: key for key, value in self.encoder.items()}
        self.bpe_ranks = dict(zip(merge_pairs, range(len(merge_pairs))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pattern = regex_re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"""
            r"""[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex_re.IGNORECASE,
        )

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            index = 0
            while index < len(word):
                try:
                    pair_index = word.index(first, index)
                    new_word.extend(word[index:pair_index])
                    index = pair_index
                except ValueError:
                    new_word.extend(word[index:])
                    break

                if word[index] == first and index < len(word) - 1 and word[index + 1] == second:
                    new_word.append(first + second)
                    index += 2
                else:
                    new_word.append(word[index])
                    index += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        tokenized = " ".join(word)
        self.cache[token] = tokenized
        return tokenized

    def encode(self, text: str) -> list[int]:
        bpe_tokens = []
        text = _whitespace_clean(_basic_clean(text)).lower()
        for token in regex_re.findall(self.pattern, text):
            token = "".join(self.byte_encoder[byte] for byte in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens


@lru_cache()
def _clip_tokenizer() -> SimpleTokenizer:
    return SimpleTokenizer()


def tokenize_clip_text(
    texts: str | list[str],
    context_length: int = CLIP_CONTEXT_LENGTH,
    truncate: bool = False,
) -> Any:
    """Tokenize text prompts for the local OpenAI CLIP text encoder."""

    require_torch()
    if isinstance(texts, str):
        texts = [texts]

    tokenizer = _clip_tokenizer()
    start_token = tokenizer.encoder["<|startoftext|>"]
    end_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[start_token] + tokenizer.encode(text) + [end_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for index, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if not truncate:
                raise RuntimeError(
                    f"Input {texts[index]} is too long for context length {context_length}"
                )
            tokens = tokens[:context_length]
            tokens[-1] = end_token
        result[index, : len(tokens)] = torch.tensor(tokens)
    return result


if is_torch_available():

    def _temporal_attention_mask(num_frames: int, net_level: int) -> Any:
        """Build the stage-specific temporal mask used by T-MSA."""

        if net_level == LOW_STAGE_LEVEL:
            num_segments = LOW_STAGE_TEMPORAL_SEGMENTS
        elif net_level == MID_STAGE_LEVEL:
            num_segments = MID_STAGE_TEMPORAL_SEGMENTS
        else:
            return None

        frame_ids = torch.arange(num_frames)
        segment_ids = torch.div(frame_ids * num_segments, num_frames, rounding_mode="floor")
        same_segment = rearrange(segment_ids, "n -> n 1") == rearrange(segment_ids, "n -> 1 n")
        return ~same_segment


    def _is_trainable_visual_parameter(name: str) -> bool:
        return name.startswith(TRAINABLE_PARAMETER_PREFIXES) or any(
            part in name for part in TRAINABLE_PARAMETER_PARTS
        )


    class Adapter(nn.Module):
        """Bottleneck adapter used for parameter-efficient CLIP adaptation."""

        def __init__(self, features: int, mlp_ratio: float = 0.25, skip_connect: bool = True):
            super().__init__()
            hidden_features = int(features * mlp_ratio)
            self.skip_connect = skip_connect
            self.fc1 = nn.Linear(features, hidden_features)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_features, features)

        def forward(self, x: Any) -> Any:
            residual = self.fc2(self.act(self.fc1(x)))
            return x + residual if self.skip_connect else residual


    class AdapterPlus(nn.Module):
        """Temporal adapter for the Multi-stage Motion Adaptation Network.

        The paper uses progressively wider temporal receptive fields: low-stage
        adapters operate over short snippets, mid-stage adapters over larger
        temporal segments, and high-stage adapters keep global temporal attention.
        """

        def __init__(
            self,
            d_model: int,
            n_head: int,
            num_frames: int = DEFAULT_NUM_FRAMES,
            net_level: int = 0,
            adapter_mlp_ratio: float = DEFAULT_ADAPTER_MLP_RATIO,
        ):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_head)
            self.num_frames = num_frames
            self.adapter = Adapter(
                d_model,
                mlp_ratio=adapter_mlp_ratio,
                skip_connect=False,
            )
            self.attn_mask = _temporal_attention_mask(num_frames, net_level)

        def forward(self, x: Any) -> Any:
            bt, n, d = x.size()
            t = self.num_frames
            b = bt // t
            x = rearrange(x, "(b t) n d -> t (b n) d", b=b, t=t, n=n, d=d)
            attn_mask = (
                self.attn_mask.to(device=x.device)
                if self.attn_mask is not None
                else None
            )
            x = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
            x = self.adapter(x)
            return rearrange(x, "t (b n) d -> (b t) n d", b=b, t=t, n=n, d=d)


    class LayerNorm(nn.LayerNorm):
        """LayerNorm that preserves the input precision used by CLIP."""

        def forward(self, x: Any) -> Any:
            dtype = x.dtype
            return super().forward(x.type(torch.float32)).type(dtype)


    class QuickGELU(nn.Module):
        """Activation function used in the original CLIP ViT implementation."""

        def forward(self, x: Any) -> Any:
            return x * torch.sigmoid(1.702 * x)


    class TextResidualAttentionBlock(nn.Module):
        """Original CLIP text transformer block."""

        def __init__(self, d_model: int, n_head: int, attn_mask: Any = None):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_head)
            self.ln_1 = LayerNorm(d_model)
            self.mlp = nn.Sequential(
                OrderedDict(
                    [
                        ("c_fc", nn.Linear(d_model, d_model * 4)),
                        ("gelu", QuickGELU()),
                        ("c_proj", nn.Linear(d_model * 4, d_model)),
                    ]
                )
            )
            self.ln_2 = LayerNorm(d_model)
            self.attn_mask = attn_mask

        def attention(self, x: Any) -> Any:
            attn_mask = (
                self.attn_mask.to(dtype=x.dtype, device=x.device)
                if self.attn_mask is not None
                else None
            )
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

        def forward(self, x: Any) -> Any:
            x = x + self.attention(self.ln_1(x))
            return x + self.mlp(self.ln_2(x))


    class TextTransformer(nn.Module):
        """Transformer stack used by the local CLIP text encoder."""

        def __init__(self, width: int, layers: int, heads: int, attn_mask: Any):
            super().__init__()
            self.width = width
            self.layers = layers
            self.resblocks = nn.ModuleList(
                [TextResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
            )

        def forward(self, x: Any) -> Any:
            for block in self.resblocks:
                x = block(x)
            return x


    class CLIPTextEncoder(nn.Module):
        """OpenAI CLIP text branch, implemented locally without HuggingFace."""

        def __init__(
            self,
            embed_dim: int = CLIP_PROJECTION_DIM,
            context_length: int = CLIP_CONTEXT_LENGTH,
            vocab_size: int = CLIP_VOCAB_SIZE,
            transformer_width: int = CLIP_TEXT_WIDTH,
            transformer_heads: int = CLIP_TEXT_HEADS,
            transformer_layers: int = CLIP_TEXT_LAYERS,
        ):
            super().__init__()
            self.context_length = context_length
            self.vocab_size = vocab_size
            self.transformer = TextTransformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
            )
            self.token_embedding = nn.Embedding(vocab_size, transformer_width)
            self.positional_embedding = nn.Parameter(
                torch.empty(self.context_length, transformer_width)
            )
            self.ln_final = LayerNorm(transformer_width)
            self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
            self.initialize_parameters()

        @property
        def dtype(self) -> Any:
            return self.token_embedding.weight.dtype

        def initialize_parameters(self) -> None:
            nn.init.normal_(self.token_embedding.weight, std=0.02)
            nn.init.normal_(self.positional_embedding, std=0.01)
            projection_std = self.transformer.width**-0.5
            attention_std = self.transformer.width**-0.5
            fc_std = (2 * self.transformer.width) ** -0.5
            residual_std = projection_std * ((2 * self.transformer.layers) ** -0.5)
            for block in self.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attention_std)
                nn.init.normal_(block.attn.out_proj.weight, std=residual_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=residual_std)
            nn.init.normal_(self.text_projection, std=projection_std)

        def build_attention_mask(self) -> Any:
            mask = torch.empty(self.context_length, self.context_length)
            mask.fill_(float("-inf"))
            mask.triu_(1)
            return mask

        def encode_text(self, text: Any) -> Any:
            x = self.token_embedding(text).type(self.dtype)
            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x).type(self.dtype)
            end_token_positions = text.argmax(dim=-1)
            text_features = x[
                torch.arange(x.shape[0], device=x.device),
                end_token_positions,
            ]
            return text_features @ self.text_projection

        def forward(self, text: Any) -> Any:
            return self.encode_text(text)


    class ResidualAttentionBlock(nn.Module):
        """CLIP ViT block augmented with optional temporal motion modeling."""

        def __init__(
            self,
            d_model: int,
            n_head: int,
            net_level: int,
            is_adapter: bool,
            num_frames: int,
            adapter_mlp_ratio: float,
        ):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_head)
            self.ln_1 = LayerNorm(d_model)
            self.mlp = nn.Sequential(
                OrderedDict(
                    [
                        ("c_fc", nn.Linear(d_model, d_model * 4)),
                        ("gelu", QuickGELU()),
                        ("c_proj", nn.Linear(d_model * 4, d_model)),
                    ]
                )
            )
            self.ln_2 = LayerNorm(d_model)
            self.adapter_plus = (
                AdapterPlus(
                    d_model,
                    n_head,
                    num_frames=num_frames,
                    net_level=net_level,
                    adapter_mlp_ratio=adapter_mlp_ratio,
                )
                if is_adapter
                else None
            )

        def attention(self, x: Any) -> Any:
            batch, length, _ = x.size()
            heads = self.attn.num_heads
            qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
            qkv = qkv.view(batch, length, heads * 3, -1).permute(0, 2, 1, 3)
            q, k, v = qkv.split([heads, heads, heads], dim=1)
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.permute(0, 2, 1, 3).flatten(-2)
            return self.attn.out_proj(out)

        def forward(self, x: Any) -> Any:
            # Paper Eq. (1)-(3): spatial MSA is inherited from CLIP, while the
            # optional T-MSA adapter injects cross-frame motion information.
            x = x + self.attention(self.ln_1(x))
            mlp_input = self.ln_2(x)
            if self.adapter_plus is not None:
                mlp_input = mlp_input + self.adapter_plus(x)
            return x + self.mlp(mlp_input)


    class Transformer(nn.Module):
        """Transformer stack that exposes low-, mid-, and high-stage features."""

        def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            is_adapter: bool,
            num_frames: int,
            adapter_mlp_ratio: float,
        ):
            super().__init__()
            self.resblocks = nn.ModuleList(
                [
                    ResidualAttentionBlock(
                        width,
                        heads,
                        net_level=i // STAGE_BLOCK_SIZE,
                        is_adapter=is_adapter,
                        num_frames=num_frames,
                        adapter_mlp_ratio=adapter_mlp_ratio,
                    )
                    for i in range(layers)
                ]
            )

        def forward(self, x: Any) -> tuple[Any, Any, Any]:
            x_l = None
            x_m = None
            for layer, block in enumerate(self.resblocks, start=1):
                x = block(x)
                if layer == LOW_STAGE_FEATURE_LAYER:
                    x_l = x
                elif layer == MID_STAGE_FEATURE_LAYER:
                    x_m = x
            return x, x_l, x_m


    class VisionTransformer(nn.Module):
        """CLIP visual encoder used as the MMAN backbone."""

        def __init__(
            self,
            input_resolution: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            num_classes: int,
            is_adapter: bool,
            num_frames: int = DEFAULT_NUM_FRAMES,
            adapter_mlp_ratio: float = DEFAULT_ADAPTER_MLP_RATIO,
        ):
            super().__init__()
            self.num_frames = num_frames
            self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
            scale = width**-0.5
            self.class_embedding = nn.Parameter(scale * torch.randn(width))
            self.positional_embedding = nn.Parameter(
                scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
            )
            self.ln_pre = LayerNorm(width)
            self.transformer = Transformer(
                width,
                layers,
                heads,
                is_adapter,
                num_frames=num_frames,
                adapter_mlp_ratio=adapter_mlp_ratio,
            )
            self.ln_post = LayerNorm(width)
            self.fc = nn.Linear(width, num_classes)
            self.proj = nn.Parameter(scale * torch.randn(width, CLIP_PROJECTION_DIM))

            self._freeze_pretrained_visual_parameters()

        def _freeze_pretrained_visual_parameters(self) -> None:
            """Freeze CLIP visual weights and keep adapters/classifier trainable."""

            for name, parameter in self.named_parameters():
                if not _is_trainable_visual_parameter(name):
                    parameter.requires_grad_(False)

        def forward(self, x: Any) -> tuple[Any, Any, Any]:
            batch, frames = x.size(0), x.size(2)
            x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
            x = self.conv1(x).flatten(-2).permute(0, 2, 1)
            x = torch.cat(
                [self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1), x],
                dim=1,
            )
            x = self.ln_pre(x + self.positional_embedding.to(x.dtype))
            x, x_l, x_m = self.transformer(x)

            # The downstream metric module consumes frame-wise [B, T, D]
            # features from high, low, and mid stages, matching the original
            # migrated experiment order.
            outputs = []
            for features in (x, x_l, x_m):
                features = self.ln_post(features[:, 0, :])
                features = features @ self.proj
                outputs.append(rearrange(features, "(b t) d -> b t d", b=batch, t=frames))
            return tuple(outputs)


    def _copy_spatial_attention_to_temporal_attention(model: VisionTransformer) -> None:
        """Initialize T-MSA from the corresponding frozen S-MSA block."""

        for block in model.transformer.resblocks:
            if block.adapter_plus is None:
                continue
            block.adapter_plus.attn.load_state_dict(block.attn.state_dict())


def _freeze_module(module: Any) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def build_clip_vit_base_patch16_adapter(
    num_classes: int,
    clip_checkpoint_path: str | Path | None = None,
    clip_model_id: str = DEFAULT_OPENAI_CLIP_MODEL_ID,
    num_frames: int = DEFAULT_NUM_FRAMES,
    use_adapter: bool = True,
    adapter_mlp_ratio: float = DEFAULT_ADAPTER_MLP_RATIO,
) -> Any:
    """Build the CLIP ViT-B/16 adapter backbone used by MML-FSAR."""

    require_torch()
    checkpoint_path = resolve_clip_checkpoint_path(
        clip_checkpoint_path=clip_checkpoint_path,
        clip_model_id=clip_model_id,
    )
    model = VisionTransformer(
        input_resolution=CLIP_INPUT_RESOLUTION,
        patch_size=CLIP_PATCH_SIZE,
        width=CLIP_VIT_B16_WIDTH,
        layers=CLIP_VIT_B16_LAYERS,
        heads=CLIP_VIT_B16_HEADS,
        is_adapter=use_adapter,
        num_classes=num_classes,
        num_frames=num_frames,
        adapter_mlp_ratio=adapter_mlp_ratio,
    )
    checkpoint = torch.jit.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(checkpoint.visual.state_dict(), strict=False)
    _copy_spatial_attention_to_temporal_attention(model)
    model._freeze_pretrained_visual_parameters()
    return model


def build_clip_text_encoder(
    clip_checkpoint_path: str | Path | None = None,
    clip_model_id: str = DEFAULT_OPENAI_CLIP_MODEL_ID,
) -> Any:
    """Build the frozen OpenAI CLIP ViT-B/16 text encoder."""

    require_torch()
    checkpoint_path = resolve_clip_checkpoint_path(
        clip_checkpoint_path=clip_checkpoint_path,
        clip_model_id=clip_model_id,
    )
    model = CLIPTextEncoder()
    checkpoint = torch.jit.load(str(checkpoint_path), map_location="cpu")
    checkpoint_state = checkpoint.state_dict()
    model_state = model.state_dict()
    text_state = {
        key: value
        for key, value in checkpoint_state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(text_state, strict=False)
    _freeze_module(model)
    return model
