DEFAULTS = dict(
    contrast="synth",
    resolution="random",
    run_suffix="exvivo",
    network="TopoFitSingleHemi",
    fov_out_size=(144, 224, 176),
    load_random_hemisphere=True,
    # "t1w", "t2w", "flair"
    images_train=["generation_labels_dist", "lp_dist_map", "rp_dist_map", "t1w"],
    images_validation=["lp_dist_map", "rp_dist_map", "t1w"],
    builder_train="ExvivoSynth",
    preprocessor_train_kwargs=dict(photo_mode_prob=0.1),
    preprocessor_validation_kwargs=dict(photo_mode_prob=0.1),
    # builder_train="ExvivoSynthLinearComb",
    # selectable_images_train=("t1w", "t2w", "flair"),
    builder_validation="ExvivoSelect",
    pretrained_run="t1w-1mm-single-hemi",
    pretrained_checkpoint=800,
    # evaluate_on_every=1,
    # save_example_on_every=1,
)

PHASES = {"Phase 1": dict(max_epochs=400)}

# If override is defined, `phases` will be ignored and `override` will be run
# instead. This is just a convenience for development.
# OVERRIDE = dict(TOPOFIT_ORDER_OUT=4, load_checkpoint=0, max_epochs=200)
