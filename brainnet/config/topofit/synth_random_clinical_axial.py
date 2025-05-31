builder_kw = dict(
    resolution_transforms_kw=dict(
        resolution_sampler="RandClinicalSlice",
        resolution_sampler_kw=dict(slice_idx=2),  # z (axial)
    )
)
DEFAULTS = dict(
    contrast="synth",
    resolution="random",
    run_suffix="clinical-axial",
    builder_train_kw=builder_kw,
    builder_validation_kw=builder_kw,
    load_body_from_checkpoint="/mnt/scratch/personal/jesperdn/results/TopoFit-Features/synth-random/checkpoint/state_checkpoint_00400.pt",
)

PHASES = {
    "Resolution 4.1": dict(TOPOFIT_ORDER_OUT=4, UNET_FREEZE=True, max_epochs=100),
    "Resolution 4.2": dict(TOPOFIT_ORDER_OUT=4, load_checkpoint=100, max_epochs=200),
    "Resolution 5": dict(TOPOFIT_ORDER_OUT=5, load_checkpoint=200, max_epochs=400),
    "Resolution 6": dict(TOPOFIT_ORDER_OUT=6, load_checkpoint=400, max_epochs=800),
}

# If override is defined, `phases` will be ignored and `override` will be run
# instead. This is just a convenience for development.
# OVERRIDE = dict(TOPOFIT_ORDER_OUT=4, max_epochs=1)
