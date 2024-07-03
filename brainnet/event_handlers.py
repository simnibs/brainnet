from typing import Callable


def set_head_weight(engine, weights):
    print("Setting head weights")
    engine._process_function.criterion.update_head_weights(weights)

def set_loss_weight(engine, weights):
    print("Setting loss weights")
    engine._process_function.criterion.update_loss_weights(weights)



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
    print(s) #


class TerminalLogger:
    def __init__(self, key):
        """https://stackabuse.com/how-to-print-colored-text-in-python/"""
        self.key = key

        self._colors = dict(zip(["black", "red", "green", "yellow", "blue", "purple", "cyan", "white"], range(8)))
        self._fg_colors = {k:v + 30 for k,v in self._colors.items()}
        self._bg_colors = {k:v + 40 for k,v in self._colors.items()}

        self._styles = dict(zip(["normal", "bold", "light", "italic", "underline", "blink"], range(7)))

        self._start = "\033["
        self._end = "\033[0;0m"
        self._fmt_title = f"{self._start}{self._styles['normal']}m {{:10s}}{self._end}"
        self._fmt_name = f"{self._start}{self._styles['light']}m {{:10s}}{self._end}"
        self._fmt_loss = f"{self._start}{self._fg_colors['green']}m {{:10s}}{self._end}{{:10.4f}}"

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

class LossLogger(TerminalLogger):
    def __init__(self, key):
        super().__init__(key)

    def __call__(self, engine):
        loss = engine.state.output[self.key]

        print(
            f"Epoch {engine.state.epoch:4d} :: Iteration {engine.state.iteration:4d} :: Duration {engine.state.times['EPOCH_COMPLETED']:6.2f} s"
        )
        print(self._repr_format(loss))


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

    def __call__(self, engine, evaluator):
        # epoch = engine.state.epoch              # from trainer (!)
        # iteration = evaluator.state.iteration   # from evaluator - basically just n_iter
        time_elapsed = evaluator.state.times['EPOCH_COMPLETED']
        loss = evaluator.state.metrics[self.key]

        header = self.header + f"Duration {time_elapsed:8.2f} s"
        # header = self.header + " - ".join(
        #     [
        #         f"Epoch {epoch:4d}",
        #         f"Duration {time_elapsed:6.2f} s",
        #     ]
        # )
        print(header)
        print(self._repr_format(loss))


def evaluate_model(engine, evaluator, dataloader, epoch_length: int, logger: Callable) -> None:
    evaluator.run(dataloader, max_epochs=1, epoch_length=epoch_length)
    logger(engine, evaluator)


def optimizer_set_lr(engine, lr):
    if isinstance(lr, (float, int)):
        for param_group in engine.process_function.optimizer.param_groups:
            param_group["lr"] = lr
    else:
        for param_group, gr_lr in zip(
            engine.process_function.optimizer.param_groups, lr
        ):
            param_group["lr"] = gr_lr


def optimizer_multiply_lr(engine, factor):
    if isinstance(factor, (float, int)):
        for param_group in engine.process_function.optimizer.param_groups:
            param_group["lr"] = factor
    else:
        for param_group, gr_factor in zip(
            engine.process_function.optimizer.param_groups, factor
        ):
            param_group["lr"] = gr_factor


# WANDB LOGGING




def wandb_log(engine, logger, name, evaluator):

    data = {name: {"loss": evaluator.state.metrics["loss"]}}
    # timing, lr, ...

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
