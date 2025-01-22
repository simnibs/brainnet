from nibabel.affines import apply_affine
import numpy as np

import brainnet

class Resources:
    def __init__(self):
        self.dir = brainnet.resources_dir

    def load_affine_lh_to_rh(self):
        return np.load(self.dir / "affine-lh-to-rh.npy")

    def load_template_vertices(self):
        return np.load(self.dir / "vertices-white-lh.npy")

    def get_template_vertices(self):
        """Template vertices are in RAS coordinate system in MNI305 (FreeSurfer
        Talairach). We store the vertices for the left hemisphere along with
        the affine to transform between left and right hemispheres.
        """
        vertices = dict(lh = self.load_template_vertices())
        vertices["rh"] = apply_affine(self.load_affine_lh_to_rh(), vertices["lh"])
        return vertices
