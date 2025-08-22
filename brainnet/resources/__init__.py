import json
from pathlib import Path

import torch

RESOURCES_DIR = Path(__file__).parent
PRETRAINED_MODELS_DIR = RESOURCES_DIR / "models"


def load_pretrained_state(model, contrast, resolution, device):
    return torch.load(
        PRETRAINED_MODELS_DIR / model / f"{contrast}_{resolution}_state.pt",
        map_location=device,
        weights_only=True,
    )


def load_pretrained_config(model, contrast, resolution):
    with open(
        PRETRAINED_MODELS_DIR / model / f"{contrast}_{resolution}_config.json", "r"
    ) as f:
        config = json.load(f)
    return config
