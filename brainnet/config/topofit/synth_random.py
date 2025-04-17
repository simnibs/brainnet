DEFAULTS = dict(
    contrast="synth",
    resolution="random",
    run_suffix="axial",
    load_body_from_checkpoint="/mnt/scratch/personal/jesperdn/results/TopoFit-UNet/synth_random/checkpoint/state_checkpoint_00400.pt",
)

PHASES = {
    "Resolution 4 (freeze)": dict(
        TOPOFIT_ORDER_OUT=4, UNET_FREEZE=True, max_epochs=100
    ),
    "Resolution 4": dict(TOPOFIT_ORDER_OUT=4, load_checkpoint=100, max_epochs=200),
    "Resolution 5": dict(TOPOFIT_ORDER_OUT=5, load_checkpoint=200, max_epochs=600),
    "Resolution 6": dict(TOPOFIT_ORDER_OUT=6, load_checkpoint=600, max_epochs=800),
}
# If override is defined, `phases` will be ignored and `override` will be run
# instead. This is just a convenience for development.
# OVERRIDE = dict(TOPOFIT_ORDER_OUT=4, max_epochs=1)
