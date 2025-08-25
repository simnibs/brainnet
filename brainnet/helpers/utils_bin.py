from pathlib import Path
import numpy as np


def _get_images(image: Path | str):
    image = Path(image)
    valid_text_file_suffixes = {".csv", ".txt"}
    if image.suffix in valid_text_file_suffixes:
        images = np.loadtxt(image, dtype=str).tolist()
        images = [images] if isinstance(images, str) else images  # single line
    else:
        images = [image]
    return [Path(i) for i in images]


def _get_out_dirs(out_dir: Path | str):
    out_dir = Path(out_dir)
    valid_text_file_suffixes = {".csv", ".txt"}
    if out_dir.suffix in valid_text_file_suffixes:
        out_dirs = np.loadtxt(out_dir, dtype=str).tolist()
        out_dirs = [out_dirs] if isinstance(out_dirs, str) else out_dirs
    else:
        out_dirs = [out_dir]
    return [Path(d) for d in out_dirs]


def _get_transforms(transform: Path | str | None):
    if transform is None:
        transforms = None
    else:
        try:
            # Try to read as if it is a matrix
            _ = np.loadtxt(transform, dtype=float)
            transforms = [transform]
        except ValueError:  # a path cannot be converted to a float
            # If this fails, assume a list of filenames
            transforms = np.loadtxt(transform, dtype=str).tolist()
            transforms = [transforms] if isinstance(transforms, str) else transforms
        transforms = [Path(t) for t in transforms]
    return transforms
