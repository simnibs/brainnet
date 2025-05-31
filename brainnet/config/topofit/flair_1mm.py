DEFAULTS = dict(
    contrast="flair",
    resolution="1mm",
    dataset=["ADNI3", "AIBL"],
    # run_suffix="noUC",
    # evaluate_on_every=1,
    # save_example_on_every=1,
)

PHASES = {
    "Resolution 4": dict(TOPOFIT_ORDER_OUT=4, max_epochs=200),
    "Resolution 5": dict(TOPOFIT_ORDER_OUT=5, load_checkpoint=200, max_epochs=400),
    "Resolution 6": dict(TOPOFIT_ORDER_OUT=6, load_checkpoint=400, max_epochs=800),
}

# If override is defined, `phases` will be ignored and `override` will be run
# instead. This is just a convenience for development.
# OVERRIDE = dict(TOPOFIT_ORDER_OUT=4, load_checkpoint=0, max_epochs=200)
