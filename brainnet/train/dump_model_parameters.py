import argparse
import importlib
from pathlib import Path
import pickle
import sys

import brainnet.initializers


import zipfile

import torch

HEMISPHERES = ("lh", "rh")




import json
filename = "test.json"

train_setup_file = "brainnet.config.topofit.mri.main"
config = importlib.import_module(train_setup_file)

with open(filename, "w") as f:
    json.dump(dict(
        out_order = config.out_order,
        out_center_str = config.out_center_str,
        out_size = config.out_size,
        unet_kwargs = config.unet_kwargs,
        topofit_kwargs = config.topofit_kwargs,
    ), f
    )


def compile_model(device="cpu"):
    """

    args = parse_args([
        "brainnet/train/dump_model_parameters.py",
        "brainnet.config.topofit.mri.main",
        "test"
    ])



    """
    train_setup_file = args.config  # "brainnet.config.cortex.main"
    train_setup = getattr(importlib.import_module(train_setup_file), "train_setup")
    model = brainnet.initializers.init_model(train_setup.model)
    model.eval()

    """Script compilation of TopoFit model.

    This will compile the model as a torch *script* model, i.e., preserving
    conditional switches.

    """

    assert device == "cpu"

    with torch.inference_mode():
        compiled_model = torch.compile(model)

    filename = Path("best_t1w_1mm.pt")
    torch.save(compiled_model, filename)
    with zipfile.ZipFile(
        filename.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED
    ) as f:
        f.writestr(filename.name, compiled_model.save_to_buffer())


def parse_args(argv):
    description = "Main interface to training a BrainNet model. For convenience, a few parameters are exposed on the command line. Values provided here will overwrite those set in the configuration file."
    parser = argparse.ArgumentParser(
        prog="BrainNetTrainer",
        description=description,
    )
    parser.add_argument(
        "config", help="Configuration file defining the parameters for training."
    )
    parser.add_argument(
        "filename", help="Name of the file to save."
    )

    return parser.parse_args(argv[1:])

if __name__ == "__main__":
    args = parse_args(sys.argv)
    save_model_config(args)
