DEFAULTS = dict(
    contrast="synth",
    resolution="random",
    run_suffix="exvivo",
    network="TopoFitSingleHemi",
    fov_out_size=(144, 224, 176),
    # fov_out_size=(128, 208, 176),
    load_random_hemisphere=True,
    images_train=["generation_labels_dist", "lp_dist_map", "rp_dist_map"],
    images_validation=["lp_dist_map", "rp_dist_map", "t1w"],
    builder_train="ExvivoSynth",
    preprocessor_train_kwargs=dict(photo_mode_prob=0.1),
    preprocessor_validation_kwargs=dict(photo_mode_prob=0.1),
    builder_validation="ExvivoSelect",
    # evaluate_on_every=1,
    # save_example_on_every=1,
    # load_body_from_checkpoint="/mnt/projects/CORTECH/nobackup/jesper/results/TopoFit-Features/synth-random-lh/checkpoint/state_checkpoint_00400.pt",
    load_body_from_checkpoint="/mnt/scratch/personal/jesperdn/results/TopoFit-Features/synth-random-single-hemi/checkpoint/state_checkpoint_00400.pt",
)

PHASES = {
    # "Resolution 4": dict(TOPOFIT_ORDER_OUT=4, max_epochs=200),
    "Resolution 4.1": dict(TOPOFIT_ORDER_OUT=4, UNET_FREEZE=True, max_epochs=100),
    "Resolution 4.2": dict(TOPOFIT_ORDER_OUT=4, load_checkpoint=100, max_epochs=200),
    "Resolution 5": dict(TOPOFIT_ORDER_OUT=5, load_checkpoint=200, max_epochs=400),
    "Resolution 6": dict(TOPOFIT_ORDER_OUT=6, load_checkpoint=400, max_epochs=800),
}

# If override is defined, `phases` will be ignored and `override` will be run
# instead. This is just a convenience for development.
# OVERRIDE = dict(TOPOFIT_ORDER_OUT=4, max_epochs=1)

# pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu12.4.html
