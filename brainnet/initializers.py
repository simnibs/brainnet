import torch

import brainsynth
import brainnet.config
import brainnet.modules

def init_model(config: brainnet.config.ModelParameters):
    # Device is needed as arg for topofit for now...
    device = torch.device(config.device)
    model = brainnet.modules.BrainNet(config.body, config.heads, device)
    model.to(device)
    return model


def init_optimizer(config, model):

    n_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Number of trainable parameters: {n_parameters}")

    if config.lr_parameter_groups is not None:
        lr_pg = config.lr_parameter_groups
        parameters = []

        # body network
        d = dict(params=model.body.parameters())
        if hasattr(lr_pg, "body"):
            d["lr"] = lr_pg.body
        parameters.append(d)

        # Task networks
        if hasattr(lr_pg, "heads"):
            for k, v in model.heads.items():
                d = dict(params=v.parameters())
                if hasattr(lr_pg.heads, k):
                    d["lr"] = getattr(lr_pg.heads, k)
                parameters.append(d)
        else:
            parameters.append(model.heads.parameters())
    else:
        parameters = model.parameters()

    return getattr(torch.optim, config.name)(parameters, **config.kwargs)


def init_dataloader(
        ds_config: brainnet.config.DatasetParameters,
        dl_config: brainnet.config.DataloaderParameters
    ):
    return {
        subset: brainsynth.dataset.setup_dataloader(config, vars(dl_config))
        for subset,config in vars(ds_config).items()
    }


def init_criterion(config: brainnet.config.CriterionParameters):
    return {subset: brainnet.Criterion(v) for subset,v in vars(config).items()}

def init_synthesizer(config: brainnet.config.SynthesizerParameters):
    return {subset: brainsynth.Synthesizer(v) for subset,v in vars(config).items()}
