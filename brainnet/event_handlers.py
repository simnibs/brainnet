

def set_loss_weight(engine, action_dict):

    for k,v in action_dict.items():
        # set loss weight k to v
        engine.process_function.criterion.loss_weights(k, v)

        ...

def multiply_loss_weight(engine, action_dict):

    for k,v in action_dict.items():
        # multiplyloss key to value
        ...


class TerminalLogger:
    def __init__(self):

        headers = ("Epoch", "Time (s)", "Raw (total)", "Weighted (total)")
        col_width = (6, 8, 15, 15)
        self._header = "   ".join(f"{j:>{i}s}" for i, j in zip(col_width, headers))

        self._print_format = (
            f"\033[1m {{{'epoch'}:{col_width[0]}d}}\033[00m   "
            f"{{{'time'}:{col_width[1]}.2f}}   "
            f"\033[95m {{{'raw_total'}:{col_width[2]}.4f}}\033[00m   "
            f"\033[96m {{{'weighted_total'}:{col_width[3]}.4f}}\033[00m   "
        )

        self._header_losses = "   ".join(
            f"{j:>{i}s}" for i, j in zip(col_width, headers)
        )
        self._loss_fmt = "\033[32m {:12s}\033[00m {:10.5f}"


    def __call__(self, engine):

        loss_total = {k: recursive_dict_sum(v) for k,v in engine.state.loss.items()}

        print(f"Epoch[{engine.state.epoch}] | Iter[{engine.state.iteration}] | Duration: {engine.state.times["EPOCH_COMPLETED"]:6.2f} s")

        print("  Loss[RAW       ]: ", " | ".join(
                [
                    self._loss_fmt.format(f"{k}[{kk}]", vv)
                    for k, v in engine.state.loss["raw"].items()
                    for kk, vv in v.items()
                ]
            ))
        print("  Loss[WEIGHTED  ]: ", " | ".join(
                [
                    self._loss_fmt.format(f"{k}[{kk}]", vv)
                    for k, v in engine.state.loss["weighted"].items()
                    for kk, vv in v.items()
                ]
            ))



def evaluate_model(engine, name, evaluator, dataloader):
    evaluator.run(dataloader)
    metrics = evaluator.state.metrics
    engine.state.evaluation[name] =
    print(f"Validation Results - Epoch[{engine.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")



def optimizer_set_lr(engine, lr):
    if isinstance(lr, (float, int)):
        for param_group in engine.process_function.optimizer.param_groups:
            param_group["lr"] = lr
    else:
        for param_group,gr_lr in zip(engine.process_function.optimizer.param_groups, lr):
            param_group["lr"] = gr_lr

def optimizer_multiply_lr(engine, factor):
    if isinstance(factor, (float, int)):
        for param_group in engine.process_function.optimizer.param_groups:
            param_group["lr"] = factor
    else:
        for param_group,gr_factor in zip(engine.process_function.optimizer.param_groups, factor):
            param_group["lr"] = gr_factor




# WANDB LOGGING

def wandb_init(engine, config):
    wandb_dir = config.wandb_dir
    if not wandb_dir.exists():
        wandb_dir.mkdir(parents=True)

    engine.state.wandb = wandb.init(
        project=config.project,
        name=config.name,
        dir=wandb_dir,
        resume=config.resume,
        **config.kwargs,
        # log the configuration of the run
        # config=recursive_namespace_to_dict(config),
    )


def wandb_log(engine):
    if engine.state.wandb is None:
        return

    engine.state.evaluation

    kwargs = {} if kwargs is None else kwargs

    data = {"train": train_loss}
    if val_loss:
        data["val"] = val_loss
    if hyper_params is not None:
        data["hyper"] = hyper_params
    if timing is not None:
        data["time"] = timing

    engine.state.wandb.log(data, step=engine.state.epoch, **kwargs)

def wandb_stop(engine):
    if engine.state.wandb is not None:
        engine.state.wandb.finish()

# Training loss
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

# Validation loss
@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def write_results():
    write_results("train", n, image, y_pred, y_true, init_verts)

