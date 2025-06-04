from collections import defaultdict
import copy
from pathlib import Path
from typing import Callable

import nibabel as nib
import torch

from ignite.engine import Engine

from brainsynth.transforms.utils import channel_last

import brainnet
from brainnet.dict_utils import recursive_keys_to_str


FREESURFER_VOLUME_INFO = dict(
    head=[2, 0, 20],
    valid="1  # volume info valid",
    filename="vol.nii",
    voxelsize=[1, 1, 1],
    volume=(256, 256, 256),
    xras=[-1, 0, 0],
    yras=[0, 0, -1],
    zras=[0, 1, 0],
    cras=[0, 0, 0],
)


def synchronize_state(engine, other, attrs):
    """Synchronize `iteration` and `epoch` from other to engine (self)."""
    for attr in attrs:
        setattr(engine.state, attr, getattr(other.state, attr))


def set_head_weight(engine, weights):
    print("Setting head weights")
    print(weights)
    engine._process_function.criterion.update_head_weights(weights)


def set_loss_weight(engine, weights):
    print("Setting loss weights")
    print(weights)
    engine._process_function.criterion.update_loss_weights(weights)


def wandb_finish(engine, logger):
    logger.finish()


def multiply_loss_weight(engine, loss_weights):
    for k, v in action_dict.items():
        # multiplyloss key to value
        ...


# class TerminalLogger:
#     def __init__(self, state_attribute="output"):
#         self.state_attr = state_attribute

#         headers = ("Epoch", "Time (s)", "Raw (total)", "Weighted (total)")
#         col_width = (6, 8, 15, 15)
#         self._header = "   ".join(f"{j:>{i}s}" for i, j in zip(col_width, headers))

#         self._print_format = (
#             f"\033[1m {{{'epoch'}:{col_width[0]}d}}\033[00m   "
#             f"{{{'time'}:{col_width[1]}.2f}}   "
#             f"\033[95m {{{'raw_total'}:{col_width[2]}.4f}}\033[00m   "
#             f"\033[96m {{{'weighted_total'}:{col_width[3]}.4f}}\033[00m   "
#         )

#         self._header_losses = "   ".join(
#             f"{j:>{i}s}" for i, j in zip(col_width, headers)
#         )
#         self._loss_fmt = "\033[32m {:12s}\033[00m {:10.5f}"

#     @staticmethod
#     def format_loss_dict(loss, total):
#         """https://stackabuse.com/how-to-print-colored-text-in-python/"""
#         color_start = "\033["
#         color_end = "\033[00m"
#         bold = "1"
#         light = "2"
#         green = "32"
#         blue = "34"
#         bg_blue = "44"
#         _title_fmt = f"{color_start};{bold}m {{:10s}}{color_end}"
#         _name_fmt = f"{color_start}{light}m {{:10s}}{color_end}"
#         _loss_fmt = f"{color_start}{green}m {{:10s}}{color_end}{{:10.3f}}"

#         for k, v in loss.items():
#             print(_title_fmt.format(k.upper()))  # : {total[k]:.2f}
#             for kk, vv in v.items():
#                 s = " | ".join([_loss_fmt.format(x, y) for x, y in vv.items()])
#                 print(f"  {_name_fmt.format(kk)} : {s}")

#     def __call__(self, engine):
#         loss = getattr(engine.state, self.state_attr)[0]

#         total = {k: recursive_dict_sum(v) for k, v in loss.items()}

#         print(
#             f"Epoch {engine.state.epoch:4d} :: Iterations {engine.state.iteration} :: Duration {engine.state.times['EPOCH_COMPLETED']:6.2f} s"
#         )

#         self.format_loss_dict(loss, total)


def log_epoch(engine):
    s = " :: ".join(
        [
            f"\033[35mEpoch {engine.state.epoch:4d}\033[0;0m",
            # f"\033[35mIteration {engine.state.iteration:6d}\033[0;0m",
            f"\033[35mDuration {engine.state.times['EPOCH_COMPLETED']:8.2f} s\033[0;0m",
        ]
    )
    print(s)


class TerminalLogger:
    def __init__(self):
        """https://stackabuse.com/how-to-print-colored-text-in-python/"""

        self._colors = dict(
            zip(
                ["black", "red", "green", "yellow", "blue", "purple", "cyan", "white"],
                range(8),
            )
        )
        self._fg_colors = {k: v + 30 for k, v in self._colors.items()}
        self._bg_colors = {k: v + 40 for k, v in self._colors.items()}

        self._styles = dict(
            zip(["normal", "bold", "light", "italic", "underline", "blink"], range(7))
        )

        self._start = "\033["
        self._end = "\033[0;0m"
        self._fmt_title = f"{self._start}{self._styles['normal']}m {{:10s}}{self._end}"
        self._fmt_name = f"{self._start}{self._styles['light']}m {{:10s}}{self._end}"
        self._fmt_loss = (
            f"{self._start}{self._fg_colors['green']}m {{:10s}}{self._end}{{:10.4f}}"
        )

    def _repr_format(self, loss):
        s = ""

        # when loss includes "raw"/"weighted" as top level key
        # for k, v in loss.items():
        #     s += self._fmt_title.format(k.upper())  # : {total[k]:.2f}
        #     s += "\n"
        # for kk, vv in v.items():
        #     s += f"  {self._fmt_name.format(kk)}"
        #     s += " : "
        #     s += " | ".join([self._fmt_loss.format(x, y) for x, y in vv.items()])
        #     s += "\n"
        # return s
        for kk, vv in loss.items():
            kk = kk if isinstance(kk, str) else ":".join(kk)
            s += f"{self._fmt_name.format(kk)}"
            s += " : "
            s += " | ".join([self._fmt_loss.format(x, y) for x, y in vv.items()])
            s += "\n"
        return s

    def __call__(self, engine):
        raise NotImplementedError


# class LossLogger(TerminalLogger):
#     def __init__(self, key):
#         super().__init__(key)

#     def __call__(self, engine):
#         loss = engine.state.output[self.key]

#         print(
#             f"Epoch {engine.state.epoch:4d} :: Iteration {engine.state.iteration:4d} :: Duration {engine.state.times['EPOCH_COMPLETED']:6.2f} s"
#         )
#         print(self._repr_format(loss))


class MetricLogger(TerminalLogger):
    def __init__(self, key, name: None | str = None):
        super().__init__()
        self.key = key
        self.name = name
        if self.name is not None:
            i = f"{self._start}{self._styles['bold']}m {self.name.upper():15s}{self._end}"
            self.header = f"Evaluation {i}"
            self.header += " "
        else:
            self.header = ""

    def __call__(self, engine):
        epoch = engine.state.epoch  # from trainer (!)
        # iteration = engine.state.iteration   # from engine - basically just n_iter

        time_elapsed = engine.state.times["EPOCH_COMPLETED"]
        loss = engine.state.metrics[self.key]

        header = self.header + f"Duration {time_elapsed:8.2f} s"
        header = self.header + " - ".join(
            [
                f"Epoch {epoch:5d}",
                # f"Iteration {iteration:5d}",
                f"Duration {time_elapsed:6.2f} s",
            ]
        )
        print(header)
        print(self._repr_format(loss))


def evaluate_model(
    engine, evaluator, dataloader, epoch_length: int, logger: Callable
) -> None:
    # set state
    evaluator.state.epoch = engine.state.epoch - 1
    evaluator.state.max_epochs = engine.state.epoch
    evaluator.state.epoch_length = epoch_length
    evaluator.state.iteration = evaluator.state.epoch * epoch_length

    evaluator.run(dataloader)
    logger(evaluator)


def optimizer_set_lr(engine, lr):
    if isinstance(lr, (float, int)):
        for param_group in engine._process_function.optimizer.param_groups:
            param_group["lr"] = lr
    else:
        for param_group, gr_lr in zip(
            engine._process_function.optimizer.param_groups, lr
        ):
            param_group["lr"] = gr_lr


def optimizer_multiply_lr(engine, factor):
    if isinstance(factor, (float, int)):
        for param_group in engine._process_function.optimizer.param_groups:
            param_group["lr"] *= factor
    else:
        for param_group, gr_factor in zip(
            engine._process_function.optimizer.param_groups, factor
        ):
            param_group["lr"] *= gr_factor


def optimizer_reset(engine):
    """Reset the adaptive parameters."""
    print("Resetting optimizer state")
    engine._process_function.optimizer.state = defaultdict(dict, {})


# WANDB LOGGING


# def wandb_log_optimizer(
#     engine,
#     wandb_logger,
# ):

#     wandb_logger.log(data, step=engine.state.epoch)


def wandb_log_evaluator(engine, wandb_logger, name, evaluator):
    loss = recursive_keys_to_str(evaluator.state.metrics["loss"])
    data = {name: {"loss": loss}}
    wandb_logger.log(data, step=engine.state.epoch)


def wandb_log_engine(engine, logger, name):
    param_groups = engine._process_function.optimizer.param_groups
    data = {
        name: {
            "loss": recursive_keys_to_str(engine.state.metrics["loss"]),
            "time[EPOCH_COMPLETED]": engine.state.times["EPOCH_COMPLETED"],
            "optimizer": {f"lr[{i}]": pg["lr"] for i, pg in enumerate(param_groups)},
        }
    }
    logger.log(data, step=engine.state.epoch)


def write_metric(engine, name, out_dir: Path):
    metric = engine.state.metrics[name]
    metric.to_pickle(out_dir / (name + ".pickle"))


def write_curv(surface: brainnet.Surface, filename: Path | str):
    """Write vertex data from a Surface object as a FreeSurfer curvature file."""
    filename = str(filename) + ".{data}"
    nbatch = len(surface.vertices)
    for k, v in surface.vertex_data.items():
        for i, vv in enumerate(v):
            nib.freesurfer.write_morph_data(
                f"{filename.format(data=k)}_{i:02d}"
                if nbatch > 1
                else filename.format(data=k),
                vv.norm(dim=-1).detach().to(torch.float).cpu().numpy(),
            )


def write_surface(
    surface: brainnet.Surface,
    filename: Path | str,
    vol_info: dict | None = None,
):
    """Write surface geometry as a FreeSurfer surface file."""
    vol_info = vol_info or FREESURFER_VOLUME_INFO
    nbatch = len(surface.vertices)
    f = surface.faces.detach().to(torch.int).cpu().numpy()
    for i, v in enumerate(surface.vertices):
        nib.freesurfer.write_geometry(
            f"{filename}_{i:02d}" if nbatch > 1 else filename,
            v.detach().to(torch.float).cpu().numpy(),
            f,
            volume_info=vol_info,
        )
        if len(surface.vertex_data) > 0:
            write_curv(surface, filename)


def write_surfaces(
    surfaces: dict[str, brainnet.Surface] | dict[str, dict[str, brainnet.Surface]],
    out_dir: Path,
    prefix: str | None = None,
    tag: str | None = None,
    label: str | None = None,
    surf: str | None = None,
    vol_info: dict | None = None,
):
    """Assemble filename as

    [prefix.][tag.]hemi.surface[.label]
    """
    filename = "{hemi}.{surf}"
    items = {}
    if tag is not None:
        filename = "{tag}." + filename
        items["tag"] = tag
    if prefix is not None:
        filename = "{prefix}." + filename
        items["prefix"] = prefix
    if label is not None:
        filename = filename + ".{label}"
        items["label"] = label

    for k, v in surfaces.items():
        if isinstance(v, brainnet.Surface):
            items["hemi"] = k
            f = out_dir / filename.format(surf=surf, **items)
            write_surface(v, f, vol_info)
        else:  # assume dict
            items["surf"] = k
            f = out_dir / filename.format(hemi="{hemi}", **items)
            write_surfaces_dict(v, f, vol_info=vol_info)


def write_surfaces_dict(
    surfaces: dict[str, brainnet.Surface],
    filename: Path | str,
    hemi: str | None = None,
    vol_info: dict | None = None,
):
    for hemi, surface in surfaces.items():
        write_surface(surface, str(filename).format(hemi=hemi), vol_info)


def write_volume(
    volume: torch.Tensor,
    affine: torch.Tensor,
    out_dir: Path | str,
    prefix: str,
    tag: str | None = None,
    label: str | None = None,
    extension: str = "nii.gz",
):
    """Assemble filename as

        prefix[.tag][.label][.batch].ext

    where [.batch] is added if batch size > 1.

    Parameters
    ----------
    vol : torch.Tensor
        (N, C, *spatial_dims)
    affine : _type_
        (N, 4, 4)
    out_dir : Path | str
        _description_
    prefix : str
        _description_
    tag : str | None, optional
        _description_, by default None
    label : str | None, optional
        _description_, by default None
    ext : str, optional
        _description_, by default "nii.gz"
    """
    out_dir = Path(out_dir)

    filename = prefix
    if tag is not None:
        filename = filename + f".{tag}"
    if label is not None:
        filename = filename + f".{label}"
    if len(volume) > 1:
        filename += "_{image_no:02d}"
    filename = filename + f".{extension}"

    batch = zip(volume.detach(), affine.detach())
    for i, (vol, aff) in enumerate(batch):
        if vol.is_floating_point():
            # v = v.float()
            # ql = v.amin()
            # qu = v.amax()
            # v = torch.clip((v - ql) / (qu - ql), 0.0, 1.0)

            # v = (255 * channel_last(v)).to(torch.uint8)
            vol = vol.float()
        else:
            # assume a one-hot encoded image
            vol = vol.to(torch.uint8).argmax(0)[None] if vol.shape[0] > 1 else vol
            vol = vol.to(torch.uint8)

        vol = channel_last(vol).cpu().numpy()
        aff = aff.cpu().numpy()
        nib.Nifti1Image(vol, aff).to_filename(out_dir / filename.format(image_no=i))


def write_example(
    engine: Engine,
    evaluators: dict[str, Engine],
    config: brainnet.config.ResultsParameters,
):
    vol_info = copy.deepcopy(FREESURFER_VOLUME_INFO)

    for prefix, e in (dict(trainer=engine) | evaluators).items():
        _, x, y_pred, y_true = e.state.output

        vol_info["volume"] = tuple(x.shape[-3:])

        prefix = f"epoch-{engine.state.epoch:05d}.{prefix}"
        affine = torch.eye(4)[None]

        for tag, y in zip((None, "pred", "true"), (dict(x=x), y_pred, y_true)):
            for label, v in y.items():
                if config.examples_keys is None or label in config.examples_keys:
                    if label == "surface":
                        write_surfaces(
                            v, config.examples_dir, vol_info, prefix, tag, label
                        )
                    else:
                        v = v[:, :5] if "dec:" in label else v
                        label = label.replace(":", "-")
                        write_volume(v, affine, config.examples_dir, prefix, tag, label)
