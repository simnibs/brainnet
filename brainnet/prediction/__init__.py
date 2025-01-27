from collections import namedtuple
import json

import torch

from brainsynth import Synthesizer
from brainsynth.config import PredictionConfig

import brainnet
import brainnet.config
from brainnet.initializers import init_model
from brainnet.modules import body, head

PretrainedModel = namedtuple("PretrainedModel", field_names=["name", "specs"])

class PretrainedModels:
    def __init__(self):
        self.dir = brainnet.resources_dir / "models"

        self.known_models = [
            PretrainedModel("topofit", ("t1w", "1mm")),
            PretrainedModel("topofit", ("synth", "1mm")),
            PretrainedModel("topofit", ("synth", "random")),
        ]

    def _model_loader(self, name, model_config, model_state, device):
        device = torch.device(device)

        match name:
            case "topofit":
                cfg_model = brainnet.config.BrainNetParameters(
                    device = device,
                    body = body.UNet(**model_config["unet_kwargs"]),
                    heads = dict(surface = head.TopoFit(**model_config["topofit_kwargs"], device=device)),
                )

        model = init_model(cfg_model)
        model.load_state_dict(model_state)

        return model

    def load_model(self, name, specs, device):
        assert any([(name == m.name and specs == m.specs) for m in self.known_models]), "Invalid model name or specs."

        model_config = self._get_model_config(name, specs)
        model_state = self._get_model_state(name, specs, device)

        return self._model_loader(name, model_config, model_state, device)

    def _get_model_state(self, name, specs, device):
        device = torch.device(device)
        specs_str = "_".join(specs)
        return  torch.load(self.dir / name / f"{specs_str}_state.pt", map_location=device, weights_only=True)

    def _get_model_config(self, name, specs):
        specs_str = "_".join(specs)
        with open(self.dir / name / f"{specs_str}_config.json", "r") as f:
            model_config = json.load(f)
        return model_config

    def load_preprocessor(self, name, specs, device):
        assert any([(name == m.name and specs == m.specs) for m in self.known_models]), "Invalid model name or specs."

        model_config = self._get_model_config(name, specs)

        config = PredictionConfig(
            "PredictionBuilder",
            model_config["out_size"],
            model_config["out_center_str"],
            device=device
        )
        return Synthesizer(config)
