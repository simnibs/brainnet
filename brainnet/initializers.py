import torch

import brainsynth
import brainnet.config
import brainnet.modules

def init_model(config: brainnet.config.BrainNetParameters):
    # Device is needed as arg for topofit for now...
    device = torch.device(config.device)
    model_cls = getattr(brainnet.modules, config.model)
    model = model_cls(**vars(config))
    model.to(device)
    return model


def init_optimizer(config, model):

    np_body = sum(p.numel() for p in model.body.parameters() if p.requires_grad)
    print("Number of trainable parameters")
    print(f"  body           {np_body:10d}")
    # print(f"  heads")
    # for h,v in model.heads.items():
    #     np_h = sum(p.numel() for p in v.parameters() if p.requires_grad)
    #     print(f"    {h:10s}   {np_h:10d}")
    n_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total            {n_parameters:10d}")

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
        dl_config: brainnet.config.DataloaderParameters,
    ):
    dataloader = {
        subset: brainsynth.dataset.setup_dataloader(config, vars(dl_config))
        for subset,config in vars(ds_config).items()
    }
    print("Dataloaders")
    for k,v in dataloader.items():
        print(f"  {k:10s} : {len(v):6d}")
    return dataloader

def init_criterion(config: brainnet.config.CriterionParameters):
    return {subset: brainnet.Criterion(v) for subset,v in vars(config).items()}

def init_synthesizer(config: brainnet.config.SynthesizerParameters):
    return {subset: None if v is None else brainsynth.Synthesizer(v) for subset,v in vars(config).items()}
