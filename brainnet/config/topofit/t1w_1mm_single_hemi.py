DEFAULTS = dict(
    contrast="t1w",
    resolution="1mm",
    run_suffix="single-hemi",
    network="TopoFitSingleHemi",
    fov_out_size=(144, 224, 176),
    # fov_out_size=(128, 208, 176),
    load_random_hemisphere=True,
    images_train=["lp_dist_map", "rp_dist_map", "t1w"],
    images_validation=["lp_dist_map", "rp_dist_map", "t1w"],
    builder_train="ExvivoSelectIso",
    builder_validation="ExvivoSelectIso",
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
