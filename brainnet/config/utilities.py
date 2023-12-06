from types import SimpleNamespace

import yaml

from brainnet import root_dir

def load_config(filename=None):
    filename = filename or root_dir / "config" / "synthesizer.yaml"
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return recursive_dict_to_namespace(config)

def recursive_dict_to_namespace(d):
    namespace = d
    if isinstance(d, dict):
        namespace = SimpleNamespace(**d)
        for k,v in namespace.__dict__.items():
            setattr(namespace, k, recursive_dict_to_namespace(v))
    return namespace
