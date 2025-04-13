
DEFAULTS = dict(contrast="synth", resolution="1mm", run_suffix="TEST")

PHASES = {
    "Resolution 4" : dict(TOPOFIT_ORDER_OUT=4, max_epochs=200),
    "Resolution 5" : dict(TOPOFIT_ORDER_OUT=5, load_checkpoint=200, max_epochs=400),
    "Resolution 6" : dict(TOPOFIT_ORDER_OUT=6, load_checkpoint=400, max_epochs=600),
}
PHASES = {
    "Resolution 4" : dict(TOPOFIT_ORDER_OUT=4, max_epochs=1),
    "Resolution 5" : dict(TOPOFIT_ORDER_OUT=5, load_checkpoint=1, max_epochs=2),
    # "Resolution 6" : dict(TOPOFIT_ORDER_IN=6, load_checkpoint=400, max_epochs=600),
}
# If override is defined, `phases` will be ignored and `override` will be run
# instead. This is just a convenience for development.
# OVERRIDE = dict(TOPOFIT_ORDER_OUT=4, max_epochs=200)