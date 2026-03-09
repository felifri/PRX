"""Base wrapper for image reward models backed by imscore."""

import torch
from torch import Tensor


class RewardModel(torch.nn.Module):
    """Thin wrapper around an imscore model.

    Provides a stable PRX interface and configurable gradient flow.

    Args:
        imscore_model: An imscore model instance (e.g. from HPSv2.from_pretrained()).
        differentiable: If True, allow gradients through the scoring call.
            Defaults to False (evaluation-only, wrapped in torch.no_grad).
    """

    def __init__(self, imscore_model: torch.nn.Module, differentiable: bool = False):
        super().__init__()
        self.model = imscore_model
        self.differentiable = differentiable
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images: Tensor, prompts: list[str]) -> Tensor:
        """Compute reward scores for image-prompt pairs.

        Args:
            images: [B, 3, H, W] float tensor in [0, 1] range.
            prompts: List of B prompt strings.

        Returns:
            [B] float tensor of reward scores.
        """
        if self.differentiable:
            scores = self.model.score(images, prompts)
        else:
            with torch.no_grad():
                scores = self.model.score(images, prompts)
        # Normalize to [B]: some imscore models return [B,1] or [B,N]
        if scores.ndim > 1:
            scores = scores[:, 0]
        return scores
