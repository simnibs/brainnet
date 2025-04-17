DEFAULTS = dict(
    contrast="synth",
    resolution="1mm",
    # run_suffix="TEST",
)

PHASES = {
    "Phase 1": dict(max_epochs=1000),
    "Phase 2": dict(load_checkpoint=1000, max_epochs=2000),
}

# If override is defined, `phases` will be ignored and `override` will be run
# instead. This is just a convenience for development.
# OVERRIDE = dict(
#     load_checkpoint=2000,
#     max_epochs=3000,
#     # save_example_on_every=1, evaluate_on_every=1
# )
