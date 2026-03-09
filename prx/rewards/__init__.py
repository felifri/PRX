"""Image reward models for PRX, backed by imscore.

Registry of reward models that compute scores for image-prompt pairs.
"""

from prx.rewards.base import RewardModel

# Lazy import map: name -> (module_path, class_name, pretrained_id)
REWARD_MODELS: dict[str, tuple[str, str, str]] = {
    "hpsv2": ("imscore.hps.model", "HPSv2", "RE-N-Y/hpsv21"),
    "pickscore": ("imscore.pickscore.model", "PickScorer", "RE-N-Y/pickscore"),
    "image_reward": ("imscore.imreward.model", "ImageReward", "RE-N-Y/ImageReward"),
    "mps": ("imscore.mps.model", "MPS", "RE-N-Y/mpsv1"),
    "clip_score": ("imscore.preference.model", "CLIPScore", "RE-N-Y/clip-score"),
    "aesthetic": ("imscore.aesthetic.model", "ShadowAesthetic", "RE-N-Y/aesthetic-shadow-v2"),
}

__all__ = ["RewardModel", "REWARD_MODELS", "build_reward_model"]


def build_reward_model(name: str, differentiable: bool = False, **kwargs) -> RewardModel:
    """Build a reward model by name.

    Args:
        name: Key in REWARD_MODELS (e.g. "hpsv2", "pickscore").
        differentiable: If True, allow gradients through scoring.
        **kwargs: Passed to from_pretrained().

    Returns:
        An initialized RewardModel instance.

    Raises:
        KeyError: If name is not in REWARD_MODELS.
    """
    import importlib

    module_path, class_name, pretrained_id = REWARD_MODELS[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    imscore_model = cls.from_pretrained(pretrained_id, **kwargs)
    return RewardModel(imscore_model, differentiable=differentiable)
