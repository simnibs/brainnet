from pathlib import Path
from typing import Callable

import nibabel as nib
import torch

from ignite.engine import Engine

import brainnet.config


def synchronize_state(engine, other, attrs):
    """Synchronize `iteration` and `epoch` from other to engine (self)."""
    for attr in attrs:
        setattr(engine.state, attr, getattr(other.state, attr))


def set_head_weight(engine, weights):
    print("Setting head weights")
    engine._process_function.criterion.update_head_weights(weights)


def set_loss_weight(engine, weights):
    print("Setting loss weights")
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
    def __init__(self, key):
        """https://stackabuse.com/how-to-print-colored-text-in-python/"""
        self.key = key

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
        for k, v in loss.items():
            s += self._fmt_title.format(k.upper())  # : {total[k]:.2f}
            s += "\n"
            for kk, vv in v.items():
                s += f"  {self._fmt_name.format(kk)}"
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
        super().__init__(key)
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


# WANDB LOGGING


# def wandb_log_optimizer(
#     engine,
#     wandb_logger,
# ):

#     wandb_logger.log(data, step=engine.state.epoch)


def wandb_log_evaluator(engine, wandb_logger, name, evaluator):
    data = {name: {"loss": evaluator.state.metrics["loss"]}}
    wandb_logger.log(data, step=engine.state.epoch)


def wandb_log_engine(engine, logger, name):
    param_groups = engine._process_function.optimizer.param_groups
    data = {
        name: {
            "loss": engine.state.metrics["loss"],
            "time[EPOCH_COMPLETED]": engine.state.times["EPOCH_COMPLETED"],
            "optimizer": {f"lr[{i}]": pg["lr"] for i, pg in enumerate(param_groups)},
        }
    }
    logger.log(data, step=engine.state.epoch)


# # Training loss
# @trainer.on(Events.EPOCH_COMPLETED)
# def log_training_results(trainer):
#     train_evaluator.run(train_loader)
#     metrics = train_evaluator.state.metrics
#     print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

# # Validation loss
# @trainer.on(Events.EPOCH_COMPLETED)
# def log_validation_results(trainer):
#     val_evaluator.run(val_loader)
#     metrics = val_evaluator.state.metrics
#     print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

# @trainer.on(Events.EPOCH_COMPLETED)
# def write_results():
#     write_results("train", n, image, y_pred, y_true, init_verts)

# Write example to disk


def write_surface(
    v: dict, vol_info: dict, out_dir: Path, prefix: str, tag: str, label: str
):
    for hemi, s in v.items():
        for surface, ss in s.items():
            name = ".".join([prefix, tag, hemi, surface, label])

            v = ss.vertices[0].detach().to(torch.float).cpu().numpy()
            f = ss.faces.detach().to(torch.int).cpu().numpy()

            nib.freesurfer.write_geometry(
                out_dir / name,
                v,
                f,
                volume_info=vol_info,
            )


def write_volume(
    v: torch.Tensor, affine, out_dir: Path, prefix: str, tag: str, label: None | str
):
    ext = "nii.gz"

    if label is None:
        name = ".".join((prefix, tag, ext))
    else:
        name = ".".join((prefix, tag, label, ext))

    v = v[0].detach()  # first example from batch
    if v.is_floating_point():
        v = (255 * v[0]).to(torch.uint8)
    else:
        # assume a one-hot encoded image
        v = v.argmax(0).to(torch.uint8)

    nib.Nifti1Image(v.cpu().numpy(), affine).to_filename(out_dir / name)


def write_example(
    engine: Engine,
    evaluators: dict[str, Engine],
    config: brainnet.config.ResultsParameters,
):
    vol_info = dict(
        head=[2, 0, 20],
        valid="1  # volume info valid",
        filename="vol.nii",
        voxelsize=[1, 1, 1],
        volume=(0, 0, 0),
        xras=[-1, 0, 0],
        yras=[0, 0, -1],
        zras=[0, 1, 0],
        cras=[0, 0, 0],
    )

    for prefix, e in (dict(trainer=engine) | evaluators).items():
        _, x, y_pred, y_true = e.state.output

        vol_info["volume"] = tuple(x.shape[-3:])

        st = f"epoch-{engine.state.epoch:05d}.{prefix}"
        affine = torch.eye(4).numpy()

        for label, y in zip((None, "pred", "true"), (dict(x=x), y_pred, y_true)):
            for k, v in y.items():
                if k == "surface":
                    write_surface(v, vol_info, config.examples_dir, st, k, label)
                else:
                    write_volume(v, affine, config.examples_dir, st, k, label)
