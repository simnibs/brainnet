# BrainNet
Foundation model for different tasks related to analysis of brain anatomy.


## Configuration

### Custom Tags
- `!Path` constructs a `pathlib.Path` object from the value.
- `!LabelingScheme` reads the labeling scheme from `brainsynth.config.utilities.labeling_scheme` associated with the given value.
- `!include` includes the contents of the yaml file being pointed to in the current file.