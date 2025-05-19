DEFAULTS = dict(
    contrast="synth",
    resolution="1mm",
    # run_suffix="neglogprob-std",
)

PHASES = {
    "Phase 1": dict(max_epochs=400),
}

# If override is defined, `phases` will be ignored and `override` will be run
# instead. This is just a convenience for development.
# OVERRIDE = dict(TOPOFIT_ORDER_OUT=4, load_checkpoint=0, max_epochs=200)
