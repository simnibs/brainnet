DEFAULTS = dict(
    contrast="synth",
    resolution="random",
    run_suffix="reg",
    pretrained_run="t1w-1mm-reg",
    pretrained_checkpoint=800,
)

PHASES = {"Phase 1": dict(max_epochs=400)}

# If override is defined, `phases` will be ignored and `override` will be run
# instead. This is just a convenience for development.
# OVERRIDE = dict(TOPOFIT_ORDER_OUT=4, load_checkpoint=0, max_epochs=200)
