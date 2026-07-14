# BrainNet

Deep learning models for different tasks related to analysis of brain anatomy. Currently, two model are included:

- **TREGA** which estimates an affine transformation to MNI152 or MNI305 space.
- **TopoFit** which estimates cortical surfaces from (almost) any MRI scan.

##### A Note on Expected Input Orientation and Resolution

> The networks expect the input scan(s) to be "conformed" meaning that it has to be (1) 1 mm isotropic resolution and (2) RAS oriented. Effectively, this means that the linear part of the affine transformation matrix should a 3x3 identity matrix (or very close to). If this is not the case, the scan can be automatically resampled to this orientation by specifying the `--conform` flag by doing `brainnet --conform [your command]`. In the examples below, we assume that the input scan is already "conformed".

## Usage

### TREGA
Estimate an affine registration from image space to MNI152 or MNI305.

    brainnet trega image.nii.gz ./output_dir

### TopoFit
Estimate cortical surfaces, uncertainty, and spherical registration.

    brainnet topofit image.nii.gz ./output_dir

Default is to use the model trained on 1 mm isotropic T1w images. To specify models based on other contrasts and resolutions, use `-c` and `-r` flags, e.g.,

    brainnet topofit -c synth -r random image.nii.gz ./output_dir

to use a model trained on synthetic data with random resolution.

## Installation

### Basic Installation
`brainnet` is not available on PyPI (yet). Basic installation (enough to do inference) is simple, however, for now you need to handle some dependencies manually. For example,

    pip install https://github.com/simnibs/cortech/releases/download/v0.1/cortech-0.1-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
    pip install brainsynth@git+https://github.com/simnibs/brainsynth@v0.1
    pip install brainnet@git+https://github.com/simnibs/brainnet@v0.2


For an editable installation, do (with similar dependency handling as above)

    git clone
    cd brainnet
    pip install -e .


### Building the included CUDA Extension

For training we rely on a few CUDA extensions which at the moment are not build as part of the package installation and so needs to be build manually. To do this, `cd` into `/brainnet/mesh/cuda` and execute `python build.py build_ext --inplace` (having installed the proper CUDA libraries, torch, etc.)

    # get cuda version used to compile torch
    python -c "import torch; print(torch.version.cuda)"

    # and use this here
    conda install -c conda-forge cudatoolkit=[torch.version.cuda]


### Kaolin (currently not used)
I have considered using Kaolin instead of the included CUDA extensions, however, we currently do not use Kaolin.

    # torchvision is for kaolin
    # torchaudio==2.6.0
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    # kaolin should match torch and cuda versions
    pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html
