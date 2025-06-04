DEFAULTS = dict(
    contrast="t1w",
    resolution="1mm",
    # run_suffix="noUC",
)

PHASES = {
    "Resolution 4": dict(TOPOFIT_ORDER_OUT=4, max_epochs=200),
    # resume_from_run="t1w-1mm",
    "Resolution 5": dict(TOPOFIT_ORDER_OUT=5, load_checkpoint=200, max_epochs=400),
    "Resolution 6": dict(TOPOFIT_ORDER_OUT=6, load_checkpoint=400, max_epochs=800),
    # "Resolution 6+": dict(TOPOFIT_ORDER_OUT=6, load_checkpoint=600, max_epochs=800),
}

# If override is defined, `phases` will be ignored and `override` will be run
# instead. This is just a convenience for development.
# OVERRIDE = dict(TOPOFIT_ORDER_OUT=4, load_checkpoint=0, max_epochs=200)
