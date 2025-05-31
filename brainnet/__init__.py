from pathlib import Path

root_dir = Path(__file__).parent
config_dir = root_dir / "config"
resources_dir = root_dir / "resources"

__all__ = ["Criterion", "DeepSurferTopology", "Surface"]

from brainnet import config
from brainnet.mesh.surface import Surface
from brainnet.mesh.topology import DeepSurferTopology

from brainnet.modules.brainnet import BrainNet, BrainReg
from brainnet.modules.criterion import Criterion
