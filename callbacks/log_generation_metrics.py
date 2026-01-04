"""Callback for computing generation quality metrics (FID, CMMD, DINO MMD) during evaluation,
with configurable frequency like '100ba', '1ep', '5000sp', or an int (batches).
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from composer import Callback, Logger, State
from composer.core import TimeUnit
from composer.utils import dist
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Metric
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from dataset.constants import BatchKeys
from pipeline.pipeline import LatentDiffusion


class CLIPFeatureExtractor(nn.Module):
    """CLIP-based feature extractor for CMMD computation.

    Extracts visual features from images using CLIP's vision encoder from transformers.
    Compatible with torchmetrics KernelInceptionDistance as a custom feature extractor.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14-336"
    ):
        super().__init__()
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        # Load CLIP model and processor
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self.model.eval()

        # Get the input image size from the processor
        self.input_size = self.processor.crop_size["height"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLIP features from images.

        Args:
            images: Tensor of shape (N, 3, H, W) with values in [0, 255], uint8

        Returns:
            Tensor of shape (N, feature_dim) with normalized CLIP embeddings
        """
        # Convert from uint8 [0, 255] to float [0, 1]
        x = images.float() / 255.0

        # Resize using bicubic interpolation
        x = torch.nn.functional.interpolate(
            x, size=(self.input_size, self.input_size), mode='bicubic', align_corners=False
        )

        # Permute to (N, H, W, C) for processor
        x = x.permute(0, 2, 3, 1)

        # Process with CLIP processor (handles normalization)
        inputs = self.processor(
            images=x,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )

        # Move to same device as this module
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            embeddings = self.model(**inputs).image_embeds

        # Normalize embeddings
        embeddings = embeddings / torch.linalg.norm(embeddings, dim=-1, keepdim=True)

        return embeddings


class DINOFeatureExtractor(nn.Module):
    """DINO-based feature extractor for DINO MMD computation.

    Extracts visual features using DINO v2 self-supervised model.
    Compatible with torchmetrics KernelInceptionDistance as a custom feature extractor.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitl14_reg",
        resize_size: int = 518
    ):
        """
        Args:
            model_name: DINO model variant from torch hub
            resize_size: Square size to resize images to (default 518x518)
        """
        super().__init__()
        # Load DINO v2 from torch hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.resize_size = resize_size

        # DINO v2 expects ImageNet normalized inputs
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract DINO features from images.

        Args:
            images: Tensor of shape (N, 3, H, W) with values in [0, 255], uint8

        Returns:
            Tensor of shape (N, feature_dim) with DINO features
        """
        # Convert from uint8 [0, 255] to float [0, 1]
        x = images.float() / 255.0

        # Normalize using ImageNet mean/std
        x = (x - self.mean) / self.std

        # Resize to configured size
        x = torch.nn.functional.interpolate(
            x, size=(self.resize_size, self.resize_size), mode='bicubic', align_corners=False
        )

        # Extract features
        with torch.no_grad():
            features = self.model(x)

        return features


@dataclass
class _Frequency:
    value: int
    unit: TimeUnit  # BA, EP, or SP

    @classmethod
    def from_input(cls, x: Union[int, str]) -> "_Frequency":
        if isinstance(x, int):
            return cls(x, TimeUnit.BATCH)

        if isinstance(x, str):
            m = re.fullmatch(r"\s*([\d_]+)\s*(ba|ep|sp)\s*", x.lower())
            if not m:
                raise ValueError(f"Invalid frequency string: {x!r}. Use like '100ba', '1ep', or '5000sp'.")
            v_str, u = m.group(1), m.group(2)
            v = int(v_str.replace("_", ""))
            unit = {"ba": TimeUnit.BATCH, "ep": TimeUnit.EPOCH, "sp": TimeUnit.SAMPLE}[u]
            return cls(v, unit)

        raise TypeError(f"Unsupported frequency type for 'frequency': {type(x)}")


class LogGenerationMetrics(Callback):
    """Computes generation quality metrics (FID, CMMD, DINO MMD) during evaluation.

    Args:
        frequency (Union[int, str]): When to run metrics. Other eval metrics can
            still run every eval pass.
            Examples: 100 or "100ba" (every 100 batches), "1ep" (each epoch),
            "5000sp" (every 5k samples). Default: "1ep".
        guidance_scales (List[float]): CFG scales to evaluate. Default: [3.5].
        seed (int): RNG seed used in generation. Default: 1138.
        num_inference_steps (int): Denoising steps. Default: 28.
        max_samples (int): Maximum number of samples to use for metric computation (global across all workers).
            This value is divided by the number of workers to get the per-worker limit.
            Once the limit is reached, metrics are computed and remaining eval batches
            are skipped for metric updates. Default: 10000.
        compute_fid (bool): Enable FID computation. Default: True.
        compute_cmmd (bool): Enable CMMD (CLIP-based MMD) computation. Default: True.
        compute_dino_mmd (bool): Enable DINO MMD computation. Default: True.
        clip_model_name (str): CLIP model for CMMD. Default: "openai/clip-vit-large-patch14-336".
        dino_model_name (str): DINO model for DINO MMD. Default: "dinov2_vitl14_reg".
        dino_resize_size (int): Resize size for DINO feature extraction. Default: 518.
    """

    def __init__(
        self,
        frequency: Union[int, str] = "1ep",
        guidance_scales: List[float] = [3.5],
        seed: int = 1138,
        num_inference_steps: int = 28,
        max_samples: int = 10000,
        compute_fid: bool = True,
        compute_cmmd: bool = True,
        compute_dino_mmd: bool = True,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        dino_model_name: str = "dinov2_vitl14_reg",
        dino_resize_size: int = 518,
    ) -> None:
        self.frequency = _Frequency.from_input(frequency)
        self.guidance_scales = list(guidance_scales)
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.max_samples = max_samples

        # Metric enable flags
        self.compute_fid = compute_fid
        self.compute_cmmd = compute_cmmd
        self.compute_dino_mmd = compute_dino_mmd

        # Model names for custom extractors
        self.clip_model_name = clip_model_name
        self.dino_model_name = dino_model_name
        self.dino_resize_size = dino_resize_size

        # Metrics dictionary: {metric_type: {scale: metric_instance}}
        self.metrics: Optional[Dict[str, Dict[float, Metric]]] = None
        self._metrics_initialized = False

        # Feature extractors (stored separately for device management)
        self.clip_extractor: Optional[CLIPFeatureExtractor] = None
        self.dino_extractor: Optional[DINOFeatureExtractor] = None

        # Gate for whether we should run on the current eval pass
        self._run_this_eval = False
        # Track samples processed in current eval (per-worker count)
        self._samples_processed = 0
        # Per-worker sample limit (max_samples divided by world size)
        self._per_worker_limit = 0
        # Flag to indicate we've hit the sample limit
        self._metric_computation_done = False

    def get_model(self, state: State) -> LatentDiffusion:
        """Get the model object, unwrapping from DDP if necessary."""
        if isinstance(state.model, DistributedDataParallel):
            return state.model.module
        return state.model

    def _should_run_now(self, state: State) -> bool:
        """Decide whether to act on this eval pass based on training progress."""
        cur_ba = state.timestamp.batch.value
        cur_ep = int(state.timestamp.epoch.value)
        cur_sp = state.timestamp.sample.value

        freq = self.frequency
        if freq.unit == TimeUnit.BATCH:
            return bool(cur_ba > 0 and (cur_ba % freq.value == 0))
        if freq.unit == TimeUnit.EPOCH:
            return bool(cur_ep > 0 and (cur_ep % freq.value == 0))
        if freq.unit == TimeUnit.SAMPLE:
            return bool(cur_sp > 0 and (cur_sp % freq.value == 0))
        return False

    def eval_start(self, state: State, logger: Logger) -> None:
        """Initialize metrics and reset sample counts at the start of selected evals."""
        self._run_this_eval = self._should_run_now(state)
        if not self._run_this_eval:
            return

        # Reset sample counter and computation flag for this eval pass
        self._samples_processed = 0
        self._metric_computation_done = False

        # Calculate per-worker sample limit (global max_samples divided by world size)
        world_size = dist.get_world_size()
        self._per_worker_limit = max(1, self.max_samples // world_size)

        if dist.get_global_rank() == 0:
            print(f"Global max_samples: {self.max_samples}, World size: {world_size}, "
                  f"Per-worker limit: {self._per_worker_limit}")

        model = self.get_model(state)
        device = model.denoiser_device

        # Initialize metrics lazily on the first run
        if not self._metrics_initialized:
            # Initialize nested metrics dictionary
            self.metrics = {}

            # Initialize FID metrics
            if self.compute_fid:
                self.metrics["FID"] = {
                    scale: FrechetInceptionDistance(normalize=True).to(device)
                    for scale in self.guidance_scales
                }

            # Initialize CMMD metrics (CLIP-based KID)
            if self.compute_cmmd:
                # Create CLIP feature extractor on GPU
                self.clip_extractor = CLIPFeatureExtractor(
                    model_name=self.clip_model_name
                ).to(device)

                self.metrics["CMMD"] = {
                    scale: KernelInceptionDistance(
                        feature=self.clip_extractor,
                        subset_size=50,
                        normalize=True
                    ).to(device)
                    for scale in self.guidance_scales
                }

            # Initialize DINO MMD metrics
            if self.compute_dino_mmd:
                # Create DINO feature extractor on GPU
                self.dino_extractor = DINOFeatureExtractor(
                    model_name=self.dino_model_name,
                    resize_size=self.dino_resize_size
                ).to(device)

                self.metrics["DINO_MMD"] = {
                    scale: KernelInceptionDistance(
                        feature=self.dino_extractor,
                        subset_size=50,
                        normalize=True
                    ).to(device)
                    for scale in self.guidance_scales
                }

            self._metrics_initialized = True
        else:
            # Move feature extractors to GPU for subsequent eval passes
            if self.clip_extractor is not None:
                self.clip_extractor.to(device)
            if self.dino_extractor is not None:
                self.dino_extractor.to(device)

    @torch.inference_mode()  # type: ignore
    def eval_batch_end(self, state: State, logger: Logger) -> None:
        if not self._run_this_eval:
            return

        # If we've already computed metrics (hit sample limit), skip remaining batches
        if self._metric_computation_done:
            return

        assert self.metrics is not None, "Metrics should be initialized in eval_start."

        model = self.get_model(state)
        batch = state.batch

        # Check if we've hit the per-worker sample limit
        batch_size = batch[BatchKeys.image].shape[0] if BatchKeys.image in batch else batch[BatchKeys.image_latent].shape[0]
        if self._samples_processed >= self._per_worker_limit:
            if dist.get_global_rank() == 0:
                print(f"Rank 0: Reached per-worker limit ({self._per_worker_limit} samples). "
                      f"Computing metrics (global total: ~{self._per_worker_limit * dist.get_world_size()})...")
            self._compute_and_log_metrics(state, logger)
            self._metric_computation_done = True
            return

        image_h, image_w = model.get_image_size_from_batch(batch)
        image_size = (image_h, image_w)

        # Prepare real images once (shared across all metrics)
        if BatchKeys.image in batch:
            real_images = batch[BatchKeys.image]
        else:
            real_images = model.latent_to_image(batch[BatchKeys.image_latent])

        real_images = torch.clamp(real_images * 255.0, 0, 255).to(torch.uint8)
        # Ensure channel-first (N, C, H, W)
        if real_images.ndim == 4 and real_images.shape[-1] in (1, 3):
            real_images = real_images.permute(0, 3, 1, 2).contiguous()

        # For each guidance scale, generate ONCE and update ALL metrics
        for guidance_scale in self.guidance_scales:
            # Generate images ONCE for this guidance scale
            gen_images = model.generate(
                batch=batch,
                image_size=image_size,
                guidance_scale=guidance_scale,
                seed=self.seed,
                num_inference_steps=self.num_inference_steps,
                progress_bar=False,
                denoiser=model.ema_denoiser
                if getattr(model, "ema_denoiser", None) and model.ema_denoiser.is_active
                else model.denoiser,
            )

            gen_images = torch.clamp(gen_images * 255.0, 0, 255).to(torch.uint8)
            if gen_images.ndim == 4 and gen_images.shape[-1] in (1, 3):
                gen_images = gen_images.permute(0, 3, 1, 2).contiguous()

            # Update ALL enabled metrics with the SAME generated images
            for metric_type, metrics_dict in self.metrics.items():
                metric = metrics_dict[guidance_scale]

                # Move images to metric's device
                metric_device = next(metric.parameters()).device if hasattr(metric, "parameters") else real_images.device
                real_slice = real_images.to(metric_device)
                gen_slice = gen_images.to(metric_device)

                # Update metric
                metric.update(real_slice, real=True)
                metric.update(gen_slice, real=False)

        # Increment sample counter
        self._samples_processed += batch_size

    def _compute_and_log_metrics(self, state: State, logger: Logger) -> None:
        if self.metrics is None:
            return

        for metric_type, metrics_dict in self.metrics.items():
            for scale, metric in metrics_dict.items():
                try:
                    score = metric.compute()
                    scale_str = str(scale).replace(".", "p")

                    # KID / MMD returns (mean, std) → log mean only
                    if isinstance(score, (tuple, list)):
                        score = score[0]

                    score = score.detach().float().cpu().item()

                    metric_name = f"metrics/eval/{metric_type}_scale_{scale_str}"

                    if dist.get_global_rank() == 0:
                        print(f"Logging {metric_name}: {score}")

                    logger.log_metrics(
                        {metric_name: score},
                        step=state.timestamp.batch.value,
                    )

                except RuntimeError as e:
                    print(f"Rank {dist.get_global_rank()}: Error computing {metric_type} for scale {scale}: {e}")
                except Exception as e:
                    print(f"Rank {dist.get_global_rank()}: Unexpected error computing {metric_type} for scale {scale}: {type(e).__name__}: {e}")
                finally:
                    metric.reset()


    def eval_end(self, state: State, logger: Logger) -> None:
        if not self._run_this_eval:
            return
        if self.metrics is None:
            return

        # Compute metrics if not already done (normal case without hitting sample limit)
        if not self._metric_computation_done:
            if dist.get_global_rank() == 0:
                global_total = self._samples_processed * dist.get_world_size()
                print(f"Computing metrics after processing {self._samples_processed} samples per worker "
                      f"(global total: ~{global_total})...")
            self._compute_and_log_metrics(state, logger)

        # Move feature extractors back to CPU to free GPU memory during training
        if self.clip_extractor is not None:
            self.clip_extractor.to("cpu")
        if self.dino_extractor is not None:
            self.dino_extractor.to("cpu")
