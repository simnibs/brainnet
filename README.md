# BrainNet
Foundation model for different tasks related to analysis of brain anatomy.

    # torchvision is for kaolin
    # torchaudio==2.6.0
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    # kaolin should match torch and cuda versions
    pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html

## Configuration

Currently, the CUDA extension is not build as part of the package installation and so needs to be build manually. To do this, `cd` into `/brainnet/mesh/cuda` and execute `python build.py build_ext --inplace` (having installed the proper CUDA libraries, torch, etc.)

    # get cuda version used to compile torch
    python -c "import torch; print(torch.version.cuda)"

    # and use this here
    conda install -c conda-forge cudatoolkit-dev=[torch.version.cuda]


## Kaolin

