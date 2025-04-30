DEFAULTS = dict(
    contrast="t1w",
    resolution="1mm",
    run_suffix="32-spring",
    UNET_DECODER_CHANNELS={
        ("t1w", "1mm"): [[128], [64], [32], [32]],
    },
)

PHASES = {
    "Resolution 4": dict(TOPOFIT_ORDER_OUT=4, max_epochs=200),
    "Resolution 5": dict(TOPOFIT_ORDER_OUT=5, load_checkpoint=200, max_epochs=400),
    "Resolution 6": dict(TOPOFIT_ORDER_OUT=6, load_checkpoint=400, max_epochs=600),
}

# If override is defined, `phases` will be ignored and `override` will be run
# instead. This is just a convenience for development.
# OVERRIDE = dict(TOPOFIT_ORDER_OUT=4, max_epochs=200)
