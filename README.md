# BrainNet
Foundation model for different tasks related to analysis of brain anatomy.

## Configuration

Currently, the CUDA extension is not build as part of the package installation and so needs to be build manually. To do this, `cd` into `/brainnet/mesh/cuda` and execute `python build.py build_ext --inplace` (having installed the proper CUDA libraries, torch, etc.)

## YAML Custom Tags
- `!Path` constructs a `pathlib.Path` object from the value.
- `!LabelingScheme` reads the labeling scheme from `brainsynth.config.utilities.labeling_scheme` associated with the given value.
- `!include` includes the contents of the yaml file being pointed to in the current file.