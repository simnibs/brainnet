DEFAULTS = dict(
    contrast="synth",
    resolution="random",
    run_suffix="reg",
    datasets=["ADNI-GO2"],
    images_train=["flair"],
    images_validation=["flair"],
    builder_train="PredictionBuilder",
    builder_validation="PredictionBuilder",
    # selectable_images_train="flair",
    # selectable_images_validation="flair",
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
