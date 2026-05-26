"""Main MML-FSAR model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mml_fsar.data.metadata import configured_class_names
from mml_fsar.models.clip_vit import (
    DEFAULT_ADAPTER_MLP_RATIO,
    DEFAULT_OPENAI_CLIP_MODEL_ID,
    build_clip_text_encoder,
    build_clip_vit_base_patch16_adapter,
    is_torch_available,
    resolve_clip_checkpoint_path,
    tokenize_clip_text,
)

DEFAULT_CLIP_MODEL_ID = DEFAULT_OPENAI_CLIP_MODEL_ID
TEXT_PROMPT_TEMPLATE = "a photo of {}"

CLIP_FEATURE_DIM = 512
DEFAULT_MFS_INTERMEDIATE_DIM = 128
DEFAULT_TVSH_INTERMEDIATE_DIM = 128
DEFAULT_STAGE_AGGREGATION_ALPHA = 0.15
DEFAULT_LOSS_BETA = 0.25
DEFAULT_INFERENCE_GAMMA = 0.7
COSINE_EPSILON = 0.01
PROBABILITY_EPSILON = 1e-8
DEFAULT_INITIAL_LOGIT_SCALE = 1.0

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


@dataclass(frozen=True)
class MMLFSARConfig:
    """Runtime model settings derived from an experiment YAML file."""

    num_way: int
    num_frames: int
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID
    clip_checkpoint_path: str | None = None
    train_class_names: tuple[str, ...] = ()
    valid_class_names: tuple[str, ...] = ()
    test_class_names: tuple[str, ...] = ()
    mfs_intermediate_dim: int = DEFAULT_MFS_INTERMEDIATE_DIM
    tvsh_intermediate_dim: int = DEFAULT_TVSH_INTERMEDIATE_DIM
    adapter_mlp_ratio: float = DEFAULT_ADAPTER_MLP_RATIO
    stage_aggregation_alpha: float = DEFAULT_STAGE_AGGREGATION_ALPHA
    initial_logit_scale: float = DEFAULT_INITIAL_LOGIT_SCALE
    use_motion_adaptation: bool = True
    use_mfs: bool = True
    use_tvsh: bool = True
    beta: float = DEFAULT_LOSS_BETA
    gamma: float = DEFAULT_INFERENCE_GAMMA

    @classmethod
    def from_experiment_config(cls, config: dict[str, Any]) -> "MMLFSARConfig":
        episode = config["episode"]
        model = config["model"]
        dataset = config["dataset"]
        loss = config.get("loss", {})
        inference = config.get("inference", {})
        train_class_names, valid_class_names, test_class_names = configured_class_names(
            dataset
        )
        return cls(
            num_way=int(episode["way"]),
            num_frames=int(episode["num_frames"]),
            clip_model_id=str(model.get("clip_model_id", DEFAULT_CLIP_MODEL_ID)),
            clip_checkpoint_path=model.get("clip_checkpoint_path"),
            train_class_names=train_class_names,
            valid_class_names=valid_class_names,
            test_class_names=test_class_names,
            mfs_intermediate_dim=int(
                model.get("mfs_intermediate_dim", DEFAULT_MFS_INTERMEDIATE_DIM)
            ),
            tvsh_intermediate_dim=int(
                model.get("tvsh_intermediate_dim", DEFAULT_TVSH_INTERMEDIATE_DIM)
            ),
            adapter_mlp_ratio=float(
                model.get("adapter_mlp_ratio", DEFAULT_ADAPTER_MLP_RATIO)
            ),
            stage_aggregation_alpha=float(
                model.get("stage_aggregation_alpha", DEFAULT_STAGE_AGGREGATION_ALPHA)
            ),
            initial_logit_scale=float(
                model.get("initial_logit_scale", DEFAULT_INITIAL_LOGIT_SCALE)
            ),
            use_motion_adaptation=bool(model.get("use_motion_adaptation", True)),
            use_mfs=bool(model.get("use_mfs", True)),
            use_tvsh=bool(model.get("use_tvsh", True)),
            beta=float(loss.get("beta", DEFAULT_LOSS_BETA)),
            gamma=float(inference.get("gamma", DEFAULT_INFERENCE_GAMMA)),
        )


def require_model_dependencies() -> None:
    dependencies = (torch, nn, F, rearrange)
    if any(dependency is None for dependency in dependencies) or not is_torch_available():
        raise ImportError("MML-FSAR model requires PyTorch and einops.")


if torch is not None and nn is not None:

    def _extract_class_indices(labels: Any, which_class: Any) -> Any:
        return torch.reshape(torch.nonzero(torch.eq(labels, which_class)), (-1,))


    def cos_sim(x: Any, y: Any, epsilon: float = COSINE_EPSILON) -> Any:
        """Pairwise cosine similarity for frame-level metric matching."""

        numerator = torch.matmul(x, y.transpose(-1, -2))
        xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
        ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
        denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + epsilon
        return torch.div(numerator, denominator)


    def distances_to_probabilities(class_dists: Any) -> Any:
        """Convert Bi-MHM class distances into paper-defined probabilities."""

        return F.softmax(-class_dists, dim=1)


    def probability_cross_entropy(probs: Any, labels: Any) -> Any:
        """Cross entropy for already-normalized prediction probabilities."""

        return F.nll_loss(torch.log(probs.clamp_min(PROBABILITY_EPSILON)), labels.long())


    def combine_training_losses(
        q2s_loss: Any,
        vision_text_loss: Any,
        beta: float = DEFAULT_LOSS_BETA,
    ) -> Any:
        """Overall training objective: L = beta * L_V2T + L_Q2S."""

        return beta * vision_text_loss + q2s_loss


    def combine_dagger_predictions(
        vision_text_probs: Any,
        video_matching_probs: Any,
        gamma: float,
    ) -> Any:
        """Dagger inference scores from vision-text and video matching."""

        return vision_text_probs.pow(gamma) * video_matching_probs.pow(1 - gamma)


    class CrossAttention(nn.Module):
        """Generic residual cross-attention used by paper-level modules."""

        def __init__(self, feature_dim: int, intermediate_dim: int):
            super().__init__()
            self.query_transform = nn.Linear(feature_dim, intermediate_dim)
            self.key_transform = nn.Linear(feature_dim, intermediate_dim)
            self.value_transform = nn.Linear(feature_dim, feature_dim)
            self.attention_scale = intermediate_dim**-0.5

        def forward(self, local_features: Any, global_features: Any) -> Any:
            queries = self.query_transform(local_features)
            keys = self.key_transform(global_features)
            values = self.value_transform(global_features)
            scores = torch.matmul(queries, keys.transpose(-2, -1))
            scores = scores * self.attention_scale
            weights = F.softmax(scores, dim=-1)
            return torch.matmul(weights, values) + local_features


    class MultiStageFeatureSynergizer(nn.Module):
        """Paper MFS module: high-stage features guide low and mid stages."""

        def __init__(self, feature_dim: int, intermediate_dim: int):
            super().__init__()
            self.low_stage_synergizer = CrossAttention(feature_dim, intermediate_dim)
            self.mid_stage_synergizer = CrossAttention(feature_dim, intermediate_dim)

        def forward(
            self,
            high_features: Any,
            low_features: Any,
            mid_features: Any,
        ) -> tuple[Any, Any, Any]:
            refined_low_features = self.low_stage_synergizer(low_features, high_features)
            refined_mid_features = self.mid_stage_synergizer(mid_features, high_features)
            return high_features, refined_low_features, refined_mid_features


    class TextToVideoSemanticHarmonizer(nn.Module):
        """Text-guided cross-attention for support video features."""

        def __init__(self, feature_dim: int, intermediate_dim: int):
            super().__init__()
            self.cross_attention = CrossAttention(feature_dim, intermediate_dim)

        def forward(self, video_features: Any, text_features: Any) -> Any:
            text_features = self._replicate_text_features(text_features, video_features.size(1))
            return self.cross_attention(video_features, text_features)

        @staticmethod
        def _replicate_text_features(text_features: Any, num_frames: int) -> Any:
            if text_features.dim() == 2:
                return text_features.unsqueeze(1).expand(-1, num_frames, -1)
            if text_features.size(1) == 1:
                return text_features.expand(-1, num_frames, -1)
            return text_features


    class MMLFSAR(nn.Module):
        """MML-FSAR model migrated from the Kinetics experiment."""

        def __init__(self, config: MMLFSARConfig):
            require_model_dependencies()
            super().__init__()
            self.config = config
            resolved_clip_checkpoint_path = resolve_clip_checkpoint_path(
                clip_checkpoint_path=config.clip_checkpoint_path,
                clip_model_id=config.clip_model_id,
            )
            self.vit = build_clip_vit_base_patch16_adapter(
                num_classes=config.num_way,
                clip_checkpoint_path=resolved_clip_checkpoint_path,
                clip_model_id=config.clip_model_id,
                num_frames=config.num_frames,
                use_adapter=config.use_motion_adaptation,
                adapter_mlp_ratio=config.adapter_mlp_ratio,
            )
            self.multi_stage_feature_synergizer = MultiStageFeatureSynergizer(
                feature_dim=CLIP_FEATURE_DIM,
                intermediate_dim=config.mfs_intermediate_dim,
            )
            self.text_to_video_harmonizer = TextToVideoSemanticHarmonizer(
                feature_dim=CLIP_FEATURE_DIM,
                intermediate_dim=config.tvsh_intermediate_dim,
            )
            self.text_encoder = build_clip_text_encoder(
                clip_checkpoint_path=resolved_clip_checkpoint_path,
                clip_model_id=config.clip_model_id,
            )
            self.text_encoder.requires_grad_(False)
            self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.scale.data.fill_(config.initial_logit_scale)

            # Frozen CLIP text embeddings provide the class-semantic branch used
            # by the Multimodal Interaction Module.
            self.text_features_train = self._encode_text_features(config.train_class_names)
            self.text_features_valid = self._encode_text_features(config.valid_class_names)
            self.text_features_test = self._encode_text_features(config.test_class_names)
            self.text_features_by_split = {
                "train": self.text_features_train,
                "valid": self.text_features_valid,
                "test": self.text_features_test,
            }
            self.evaluation_split = "test"

        def _encode_text_features(self, class_names: tuple[str, ...]) -> Any:
            if not class_names:
                return None
            with torch.no_grad():
                prompts = [TEXT_PROMPT_TEMPLATE.format(name) for name in class_names]
                tokens = tokenize_clip_text(prompts)
                tokens = tokens.to(next(self.text_encoder.parameters()).device)
                return self.text_encoder.encode_text(tokens)

        def set_evaluation_split(self, split: str) -> None:
            """Select the class-name split used by evaluation episodes."""

            if split not in self.text_features_by_split:
                available = ", ".join(sorted(self.text_features_by_split))
                raise ValueError(f"Unknown evaluation split {split!r}. Available: {available}.")
            self.evaluation_split = split

        def _active_text_features(self) -> Any:
            split = "train" if self.training else self.evaluation_split
            text_features = self.text_features_by_split[split]
            if text_features is None:
                raise ValueError(
                    f"MML-FSAR forward requires class names for the {split!r} split."
                )
            return text_features

        def episode_text_features(
            self,
            support_labels: Any,
            real_support_labels: Any | None = None,
        ) -> Any:
            """Select one class text feature per episode class."""

            text_features = self._active_text_features().to(support_labels.device)
            unique_labels = torch.unique(support_labels)
            if real_support_labels is None:
                class_indices = unique_labels.long()
            else:
                real_support_labels = real_support_labels.to(support_labels.device)
                class_indices = torch.stack(
                    [
                        real_support_labels[_extract_class_indices(support_labels, label)[0]]
                        for label in unique_labels
                    ]
                ).long()
            return torch.index_select(text_features, 0, class_indices)

        def support_text_features(
            self,
            support_labels: Any,
            real_support_labels: Any | None = None,
        ) -> Any:
            """Select text features aligned with each support video."""

            episode_text_features = self.episode_text_features(
                support_labels,
                real_support_labels,
            )
            unique_labels = torch.unique(support_labels)
            support_indices = torch.stack(
                [
                    _extract_class_indices(unique_labels, label)[0]
                    for label in support_labels
                ]
            ).long()
            return torch.index_select(episode_text_features, 0, support_indices)

        def bimhm_metric(self, total_features: Any, support_labels: Any) -> Any:
            """Metric Relation Module using the Bi-MHM frame matching distance."""

            num_supports = int(support_labels.numel())
            num_queries = int(total_features.size(0)) - num_supports
            if num_supports <= 0 or num_queries <= 0:
                raise ValueError("Bi-MHM requires at least one support and one query video.")

            support_features = total_features[:num_supports]
            query_features = total_features[num_supports:]
            unique_labels = torch.unique(support_labels)
            support_features = rearrange(support_features, "b s d -> (b s) d")
            query_features = rearrange(query_features, "b s d -> (b s) d")
            frame_dists = 1 - cos_sim(query_features, support_features)
            dists = rearrange(
                frame_dists,
                "(num_queries query_frames) (num_supports support_frames) "
                "-> num_queries num_supports query_frames support_frames",
                num_queries=num_queries,
                num_supports=num_supports,
            )
            cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2)
            class_dists = [
                torch.mean(
                    torch.index_select(cum_dists, 1, _extract_class_indices(support_labels, c)),
                    dim=1,
                )
                for c in unique_labels
            ]
            return rearrange(torch.stack(class_dists), "c q -> q c")

        def aggregate_stage_predictions(
            self,
            low_stage_dists: Any,
            mid_stage_dists: Any,
            high_stage_dists: Any,
            alpha: float = DEFAULT_STAGE_AGGREGATION_ALPHA,
        ) -> Any:
            """Weighted stage-probability aggregation from the paper."""

            low_stage_probs = distances_to_probabilities(low_stage_dists)
            mid_stage_probs = distances_to_probabilities(mid_stage_dists)
            high_stage_probs = distances_to_probabilities(high_stage_dists)
            return (
                alpha * (low_stage_probs + mid_stage_probs)
                + (1 - 2 * alpha) * high_stage_probs
            )

        def metric_relation_module(
            self,
            low_stage_features: Any,
            mid_stage_features: Any,
            high_stage_features: Any,
            support_labels: Any,
        ) -> Any:
            """Stage-wise Bi-MHM probabilities with paper-defined fusion."""

            low_stage_dists = self.bimhm_metric(low_stage_features, support_labels)
            mid_stage_dists = self.bimhm_metric(mid_stage_features, support_labels)
            high_stage_dists = self.bimhm_metric(high_stage_features, support_labels)
            return self.aggregate_stage_predictions(
                low_stage_dists,
                mid_stage_dists,
                high_stage_dists,
                alpha=self.config.stage_aggregation_alpha,
            )

        def compute_training_loss(self, q2s_loss: Any, vision_text_loss: Any) -> Any:
            """Combine Q2S and V2T losses with the configured beta."""

            return combine_training_losses(
                q2s_loss,
                vision_text_loss,
                beta=self.config.beta,
            )

        def dagger_inference_scores(
            self,
            vision_text_probs: Any,
            video_matching_probs: Any,
        ) -> Any:
            """Combine optional dagger inference probabilities with gamma."""

            return combine_dagger_predictions(
                vision_text_probs,
                video_matching_probs,
                gamma=self.config.gamma,
            )

        def vision_text_logits(self, query_features: Any, text_features: Any) -> Any:
            """Vision-text logits for query videos and episode text prompts."""

            return cos_sim(query_features.mean(dim=1), text_features) * self.scale

        def synergize_multistage_features(
            self,
            high_features: Any,
            low_features: Any,
            mid_features: Any,
        ) -> tuple[Any, Any, Any]:
            """Multi-stage Feature Synergizer from the paper formulation."""

            if not self.config.use_mfs:
                return high_features, low_features, mid_features
            return self.multi_stage_feature_synergizer(
                high_features,
                low_features,
                mid_features,
            )

        def harmonize_text_to_video(
            self,
            support_features: Any,
            target_features: Any,
            support_text_features: Any,
        ) -> Any:
            """Text-to-Video Semantic Harmonizer for support/query features."""

            if not self.config.use_tvsh:
                return torch.cat([support_features, target_features], dim=0)
            enriched_support_features = self.text_to_video_harmonizer(
                support_features,
                support_text_features,
            )
            return torch.cat([enriched_support_features, target_features], dim=0)

        def forward(
            self,
            support_images: Any,
            support_labels: Any,
            query_images: Any,
            query_labels: Any | None = None,
            real_support_labels: Any | None = None,
            real_query_labels: Any | None = None,
            return_loss: bool | None = None,
        ) -> dict[str, Any]:
            """End-to-end MML-FSAR episode forward path."""

            del real_query_labels

            num_supports = support_images.size(0)
            total_images = torch.cat([support_images, query_images], dim=0)
            high_stage_features, low_stage_features, mid_stage_features = self.vit(
                total_images
            )
            (
                high_stage_features,
                low_stage_features,
                mid_stage_features,
            ) = self.synergize_multistage_features(
                high_stage_features,
                low_stage_features,
                mid_stage_features,
            )

            support_text_features = self.support_text_features(
                support_labels,
                real_support_labels,
            )
            high_stage_features = self.harmonize_text_to_video(
                high_stage_features[:num_supports],
                high_stage_features[num_supports:],
                support_text_features,
            )
            low_stage_features = self.harmonize_text_to_video(
                low_stage_features[:num_supports],
                low_stage_features[num_supports:],
                support_text_features,
            )
            mid_stage_features = self.harmonize_text_to_video(
                mid_stage_features[:num_supports],
                mid_stage_features[num_supports:],
                support_text_features,
            )

            video_matching_probs = self.metric_relation_module(
                low_stage_features,
                mid_stage_features,
                high_stage_features,
                support_labels,
            )
            text_features = self.episode_text_features(
                support_labels,
                real_support_labels,
            )
            vision_text_logits = self.vision_text_logits(
                high_stage_features[num_supports:],
                text_features,
            )
            vision_text_probs = F.softmax(vision_text_logits, dim=1)
            dagger_probs = self.dagger_inference_scores(
                vision_text_probs,
                video_matching_probs,
            )
            outputs = {
                "video_matching_probs": video_matching_probs,
                "vision_text_probs": vision_text_probs,
                "vision_text_logits": vision_text_logits,
                "dagger_probs": dagger_probs,
            }

            should_return_loss = self.training if return_loss is None else return_loss
            if should_return_loss and query_labels is not None:
                q2s_loss = probability_cross_entropy(video_matching_probs, query_labels)
                all_vision_text_logits = self.vision_text_logits(
                    high_stage_features,
                    text_features,
                )
                vision_text_loss_labels = torch.cat([support_labels, query_labels], dim=0)
                vision_text_loss = F.cross_entropy(
                    all_vision_text_logits,
                    vision_text_loss_labels.long(),
                )
                outputs["q2s_loss"] = q2s_loss
                outputs["vision_text_loss"] = vision_text_loss
                outputs["loss"] = self.compute_training_loss(q2s_loss, vision_text_loss)

            return outputs


else:
    MMLFSAR = None
