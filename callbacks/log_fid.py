"""Callback for computing FID (Frechet Inception Distance) during evaluation,
with configurable frequency like '100ba', '1ep', '5000sp', or an int (batches).
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from composer import Callback, Logger, State
from composer.core import TimeUnit
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance

from dataset.constants import BatchKeys
from pipeline.pipeline import LatentDiffusion


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


class LogFID(Callback):
    """Computes FID scores during evaluation at specified guidance scales.

    Args:
        frequency (Union[int, str]): When to run FID. Other eval metrics can
    still run every eval pass.
            Examples: 100 or "100ba" (every 100 batches), "1ep" (each epoch),
            "5000sp" (every 5k samples). Default: "1ep".
        guidance_scales (List[float]): CFG scales to evaluate. Default: [3.5].
        seed (int): RNG seed used in generation. Default: 1138.
        num_inference_steps (int): Denoising steps. Default: 28.
    """

    def __init__(
        self,
        frequency: Union[int, str] = "1ep",
        guidance_scales: List[float] = [3.5],
        seed: int = 1138,
        num_inference_steps: int = 28,
    ) -> None:
        self.frequency = _Frequency.from_input(frequency)
        self.guidance_scales = list(guidance_scales)
        self.seed = seed
        self.num_inference_steps = num_inference_steps

        self.fid_metrics: Optional[Dict[float, FrechetInceptionDistance]] = None
        self._metrics_initialized = False

        # Gate for whether we should run on the current eval pass
        self._run_this_eval = False

        self._run_this_eval = False

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

        # Initialize metrics lazily on the correct device
        if not self._metrics_initialized:
            model = self.get_model(state)
            device = model.denoiser_device
            self.fid_metrics = {
                scale: FrechetInceptionDistance(normalize=True).to(device) for scale in self.guidance_scales
            }
            self._metrics_initialized = True

    @torch.inference_mode()  # type: ignore
    def eval_batch_end(self, state: State, logger: Logger) -> None:
        if not self._run_this_eval:
            return
        assert self.fid_metrics is not None, "FID metrics should be initialized in eval_start."

        model = self.get_model(state)
        batch = state.batch

        image_h, image_w = model.get_image_size_from_batch(batch)
        image_size = (image_h, image_w)

        if BatchKeys.image in batch:
            real_images = batch[BatchKeys.image]
        else:
            real_images = model.latent_to_image(batch[BatchKeys.image_latent])

        real_images = torch.clamp(real_images * 255.0, 0, 255).to(torch.uint8)
        # Ensure channel-first (N, C, H, W)
        if real_images.ndim == 4 and real_images.shape[-1] in (1, 3):
            real_images = real_images.permute(0, 3, 1, 2).contiguous()

        for guidance_scale in self.guidance_scales:
            metric = self.fid_metrics[guidance_scale]

            real_slice = real_images.to(
                next(metric.parameters()).device if hasattr(metric, "parameters") else real_images.device
            )

            metric.update(real_slice, real=True)

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
            gen_slice = gen_images.to(real_slice.device)

            metric.update(gen_slice, real=False)

    def eval_end(self, state: State, logger: Logger) -> None:
        if not self._run_this_eval:
            return
        if self.fid_metrics is None:
            return

        for scale, metric in self.fid_metrics.items():
            try:
                fid_score = metric.compute()
                scale_str = str(scale).replace(".", "p")
                logger.log_metrics(
                    {f"metrics/eval/FID_scale_{scale_str}": fid_score},
                    step=state.timestamp.batch.value,
                )
            except RuntimeError as e:
                print(f"Error computing FID for scale {scale}: {e}")
            finally:
                metric.reset()
