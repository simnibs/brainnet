DEFAULTS = dict(
    contrast="synth",
    resolution="050mm",
    # wandb_run_id="sak5xt99",
    # datasets=["ABIDE", "OASIS3"],
    # run_suffix="reg",  # WMGM-only
    # evaluate_on_every=1,
    # save_example_on_every=1,
    fov_out_size=(192, 400, 304),  # single hemi
    iterative_hemisphere_prediction=True,
    load_random_hemisphere=True,
)

PHASES = {
    # resume_from_run="t1w-1mm",
    # "Resolution 4": dict(TOPOFIT_ORDER_OUT=4, max_epochs=200),
    # "Resolution 5": dict(TOPOFIT_ORDER_OUT=5, load_checkpoint=200, max_epochs=400),
    "Resolution 6": dict(TOPOFIT_ORDER_OUT=6, load_checkpoint=500, max_epochs=800),
}

# If override is defined, `phases` will be ignored and `override` will be run
# instead. This is just a convenience for development.
# OVERRIDE = dict(TOPOFIT_ORDER_OUT=4, load_checkpoint=0, max_epochs=200)
