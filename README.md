# BrainNet
Foundation model for different tasks related to analysis of brain anatomy.

## Configuration

Currently, the CUDA extension is not build as part of the package installation and so needs to be build manually. To do this, `cd` into `/brainnet/mesh/cuda` and execute `python build.py build_ext --inplace` (having installed the proper CUDA libraries, torch, etc.)

    # get cuda version used to compile torch
    python -c "import torch; print(torch.version.cuda)"

    # and use this here
    conda install -c conda-forge cudatoolkit-dev=[torch.version.cuda]
