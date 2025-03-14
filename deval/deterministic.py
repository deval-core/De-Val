# Copied from https://github.com/backend-developers-ltd/deterministic-ml

import logging
import os
import random
import warnings

LOG = logging.getLogger(__name__)


def set_deterministic(seed: int, *, any_card: bool = False):
    LOG.info(f"Setting deterministic mode with seed {seed!r}, any_card={any_card!r}")
    random.seed(seed)
    try:
        import numpy  # noqa: F401
    except ImportError:
        pass
    else:
        numpy.random.seed(seed)

    try:
        set_deterministic_pytorch(seed, any_card=any_card)
    except ModuleNotFoundError:
        pass


def set_deterministic_pytorch(seed: int, *, any_card: bool = False):
    """
    Set random seed for PyTorch and ensure deterministic operations.

    :param seed:
    :param any_card: attempt to make it deterministic on any GPU. Most likely won't work, but will decrease performance.
    :return: None
    """
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_workspace_config is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch  # noqa: F401

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if any_card:
        warnings.warn(
            'Attempt to set "any_card" deterministic mode. '
            "This will further decrease performance and does not guarantee deterministic result across all GPUs.",
            UserWarning,
        )
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
