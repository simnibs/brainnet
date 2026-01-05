DEFAULTS = dict(
    contrast="synth",
    resolution="random",
    run_suffix="exvivo",
    # epoch_length_train=1,
    # epoch_length_val=1,
    # model_kwargs=dict(hemispheres="lh"),
    fov_out_size=(192, 224, 192),
    load_random_hemisphere=0.5,
    # train
    builder_train="ExvivoCropSynth",
    images_train=["generation_labels_dist", "lp_dist_map", "rp_dist_map"],
    preprocessor_train_kwargs=dict(photo_mode_prob=0.1),
    # validation
    builder_validation="ExvivoCropSelect",
    images_validation=["generation_labels_dist", "lp_dist_map", "rp_dist_map", "t1w"],
    preprocessor_validation_kwargs=dict(photo_mode_prob=0.1),
    # builder_train="ExvivoSynthLinearComb",
    # selectable_images_train=("t1w", "t2w", "flair"),
    # evaluate_on_every=1,
    # save_example_on_every=1,
)

PHASES = {"Phase 1": dict(max_epochs=2000)}

# If override is defined, `phases` will be ignored and `override` will be run
# instead. This is just a convenience for development.
# OVERRIDE = dict(
#     load_checkpoint=400,
#     max_epochs=800,
#     # save_example_on_every=1, evaluate_on_every=1
# )
