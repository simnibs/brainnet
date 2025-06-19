DEFAULTS = dict(
    contrast="synth",
    resolution="random",
)

PHASES = {"Phase 1": dict(max_epochs=2000)}

# If override is defined, `phases` will be ignored and `override` will be run
# instead. This is just a convenience for development.
# OVERRIDE = dict(
#     load_checkpoint=400,
#     max_epochs=800,
#     # save_example_on_every=1, evaluate_on_every=1
# )
