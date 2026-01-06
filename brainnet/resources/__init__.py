import json
from pathlib import Path

import torch

RESOURCES_DIR = Path(__file__).parent
PRETRAINED_MODELS_DIR = RESOURCES_DIR / "models"


def load_pretrained_state(
    model,
    contrast,
    resolution,
    suffix: str | None = None,
    device: str | torch.device = "cpu",
):
    if suffix is None:
        name = f"state_{contrast}_{resolution}.pt"
    else:
        name = f"state_{contrast}_{resolution}_{suffix}.pt"
    return torch.load(
        PRETRAINED_MODELS_DIR / model / name,
        map_location=device,
        weights_only=True,
    )


def load_pretrained_config(model, contrast, resolution, suffix: str | None = None):
    if suffix is None:
        name = f"config_{contrast}_{resolution}.json"
    else:
        name = f"config_{contrast}_{resolution}_{suffix}.json"

    with open(PRETRAINED_MODELS_DIR / model / name, "r") as f:
        config = json.load(f)
    return config
