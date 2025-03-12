import sys
import torch


from pathlib import Path
from ignite.engine import Events
from brainnet import config

from ignite.engine import Engine

import brainnet.config
import brainnet.train.utilities
from brainnet import event_handlers
import brainnet.initializers
from brainsynth.transforms.spatial import RandResolution
from brainsynth.transforms.resolution_sampler import RandClinicalSlice
from brainsynth.transforms.contrast import IntensityNormalization
import brainsynth.config

from brainnet.modules.image import SuperResolution


class SupervisedStep:
    def __init__(
        self,
        model: brainnet.BrainNet,
    ) -> None:
        self.model = model
        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.L1Loss()
        self.device = self.model.device
        self.in_res = (1,1,1)
        self.intensity_norm = IntensityNormalization()


class SupervisedTrainingStep(SupervisedStep):
    def __init__(
        self,
        model,
        optimizer,
        gradient_accumulation_steps: int = 1,
        enable_amp: bool = False
    ) -> None:
        super().__init__(model)
        self.optimizer = optimizer
        self.enable_amp = enable_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.enable_amp:
            self.grad_scaler = torch.amp.GradScaler("cuda")

    def __call__(self, engine, batch) -> tuple:
        self.model.train()

        images, _, _ = batch

        if images["t1w"].shape[-1] < 180:
            i = images["t1w"].shape[-1]-images["t1w"].shape[-1]%6
        else:
            i=180
        labels = images["generation_labels_dist"][:,:,:176,:208,:i].to(self.device)
        mask = labels > 0
        mask = mask.reshape(1,1,-1)

        y_true = images["t1w"][:,:,:176,:208,:i].to(self.device)
        # y_true = self.intensity_norm(y_true[0])[None]
        ymax = y_true.amax()
        ymin = y_true.amin()
        y_true = (y_true-ymin)/(ymax-ymin)

        # y_true = torch.rand(self.shape, device=self.device)
        res_sampler = RandResolution(tuple(y_true.shape[2:]), self.in_res, device=self.device)
        # res_sampler = RandClinicalSlice(5.9, 6.1)
        image_ds = res_sampler(y_true[0])[None]

        in_img = image_ds#[:,:1]
        # in_dist = image_ds[:, 1:]
        # in_dist = torch.ones_like(in_dist) # remove distance info

        image_ds_us = res_sampler.resize(image_ds)[None]

        # Only wrap forward pass and loss computation. Backward uses the same
        # types as inferred during forward
        with torch.autocast(self.device.type, enabled=self.enable_amp):
            y_pred = self.model(in_img)

            loss = self.criterion(y_pred.reshape(1,1,-1)[mask], y_true.reshape(1,1,-1)[mask])
            #loss_linear = self.criterion(image_ds_us.reshape(1,1,-1)[mask], y_true.reshape(1,1,-1)[mask])

            total_loss = loss
            total_loss /= self.gradient_accumulation_steps

        # exit if loss diverges
        if total_loss > 1e6 or torch.isnan(total_loss):
            raise RuntimeError(f"Loss diverged (loss = {total_loss})")

        if self.enable_amp:
            self.grad_scaler.scale(total_loss).backward()
            if engine.state.iteration % self.gradient_accumulation_steps == 0:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
        else:
            total_loss.backward()  # backpropagate loss
            if engine.state.iteration % self.gradient_accumulation_steps == 0:
                self.optimizer.step()  # update parameters
                # Reset gradients in optimizer. Otherwise gradients would
                # accumulate across multiple passes (whenever .backward is
                # called)
                self.optimizer.zero_grad()

        # these are stored in engine.state.output
        return dict(image=dict(l1=loss.item())), image_ds_us, dict(image=y_pred), dict(image=y_true)

class EvaluationStep(SupervisedStep):
    def __init__(self, model, enable_amp: bool = False):
        super().__init__(model)
        self.enable_amp = enable_amp

    def __call__(self, engine, batch):
        self.model.eval()

        images, _, _ = batch
        if images["t1w"].shape[-1] < 180:
            i = images["t1w"].shape[-1]-images["t1w"].shape[-1]%6
        else:
            i=180

        labels = images["generation_labels_dist"][:,:,:176,:208,:i].to(self.device)
        mask = labels > 0
        mask = mask.reshape(1,1,-1)

        y_true = images["t1w"][:,:,:176,:208,:i].to(self.device)
        # y_true = self.intensity_norm(y_true[0])[None]
        ymax = y_true.amax()
        ymin = y_true.amin()
        y_true = (y_true-ymin)/(ymax-ymin)

        # y_true = torch.rand(self.shape, device=self.device)
        res_sampler = RandResolution(tuple(y_true.shape[2:]), self.in_res, device=self.device)
        image_ds = res_sampler(y_true[0])[None]

        image_ds_us = res_sampler.resize(image_ds)[None]

        in_img = image_ds#[:,:1]
        # in_dist = image_ds[:, 1:]
        # in_dist = torch.ones_like(in_dist) # remove distance info

        with torch.autocast(self.device.type, enabled=self.enable_amp):
            with torch.inference_mode():

                # Only wrap forward pass and loss computation. Backward uses the same
                # types as inferred during forward
                y_pred = self.model(in_img)

                loss = self.criterion(y_pred.reshape(1,1,-1)[mask], y_true.reshape(1,1,-1)[mask])
                loss_linear = self.criterion(image_ds_us.reshape(1,1,-1)[mask], y_true.reshape(1,1,-1)[mask])

        return dict(image=dict(l1=loss.item(), l1_linear=loss_linear.item())), image_ds_us, dict(image=y_pred), dict(image=y_true)



# class Upsampler(torch.nn.Module):
#     def __init__(self, device, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.device = torch.device(device)

#         i = 5
#         in_ch = 1
#         out_ch = 8

#         self.res_xy = self.make_conv_block(in_ch, out_ch, i, 2)
#         self.res_xz = self.make_conv_block(in_ch, out_ch, i, 1)
#         self.res_yz = self.make_conv_block(in_ch, out_ch, i, 0)


#     def make_conv_block(self, in_ch, out_ch, ks, dim):
#         kernel_size = tuple(1 if i == dim else ks for i in range(3))
#         return torch.nn.Sequential(
#             torch.nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding="same"),
#             torch.nn.InstanceNorm3d(out_ch),
#             torch.nn.PReLU(),
#             # torch.nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size, padding="same"),
#             # torch.nn.InstanceNorm3d(out_ch),
#             # torch.nn.PReLU(),
#             torch.nn.Conv3d(out_ch, in_ch, kernel_size=kernel_size, padding="same"),
#         )

#     def forward(self, image, dist):
#         # if vox_size[0] > 1.0:
#         upxy = self.res_xy(image) # axial
#         # if vox_size[1] > 1.0:
#         upxz = self.res_xz(image) # coronal
#         # if vox_size[2] > 1.0:
#         upyz = self.res_yz(image) # sagittal

#         res = torch.cat((upxy, upxz, upyz), dim=1)

#         # total_dist : for each voxel, the euclidean distance to the closest
#         #              voxel in the original image (C=1)
#         # dist :       for each voxel, the distance to the closest voxel IN
#         #               THAT dimension (C=3)
#         #

#         # weigh the input image and the residual according to the overall dist
#         # to closest voxel
#         # The input always has a weight of 1; the total dist then determines
#         # the trade-off: if total_dist = 0 then only the input is considered,
#         # if total = 3, the 0.25 * input + 0.75 * residual
#         total_dist = dist.pow(2).sum(1, keepdim=True).sqrt()
#         total_dist_denom = total_dist + 1.0

#         # weigh the convolutions in each plane (axial, coronal, sagittal) like
#         # this
#         w_index = ((0,1), (0,2), (1,2))
#         dist_weights = torch.cat(
#             tuple(dist[:, ind].sum(1, keepdim=True) for ind in w_index), dim=1
#         )
#         # dist_weights[:, 0] = 0.0
#         # dist_weights[:, 1] = 0.0
#         # dist_weights[:, 2] = 0.0
#         normalizer = dist_weights.sum(1, keepdim=True)
#         normalizer[normalizer == 0.0] = 1.0
#         dist_weights /= normalizer

#         w_res = torch.sum(res * dist_weights, dim=1)

#         image_up = (1.0 / total_dist_denom) * image + total_dist/total_dist_denom * w_res

#         return image_up

def make_train_setup():

    project: str = "Upsampling"
    run: str = "Test-01"
    tags = []
    run_id: None | str = None  # f"{run}-00"
    resume_from_run: None | str = run # None # run
    device: str | torch.device = torch.device("cuda:0")

    out_dir: Path = Path("/mnt/scratch/personal/jesperdn/results")
    model_dir = out_dir


    data_dir: Path = Path("/mnt/projects/CORTECH/nobackup/training_data")
    out_dir: Path = Path("/mnt/scratch/personal/jesperdn/results")
    model_dir = out_dir
    # model_dir: Path = Path("/mnt/projects/CORTECH/nobackup/jesper/models")

    # =============================================================================
    # TRAINING MODE
    # =============================================================================

    # Use COBRE and MCIC as test sets
    # ISBI2015 are not great from FS
    datasets = [
        "ABIDE",
        "ADHD200",
        "ADNI3",
        "AIBL",
        "Buckner40",
        "Chinese-HCP",
        # "COBRE",
        "HCP",
        # "ISBI2015",
        # "MCIC",
        "OASIS3",
    ]


    subject_subset_exclude = "exclude"

    images_train = ["brain_dist_map", "generation_labels_dist", "t1w"]
    images_train_sel = None
    images_val = ["brain_dist_map", "generation_labels_dist", "t1w"] # ["t1w"]
    images_val_sel = ["t1w"]
    subject_subset_train = "train"
    subject_subset_val = "validation"


    # =============================================================================
    # TRAINING PARAMETERS
    # =============================================================================

    cfg_train = config.TrainParameters(
        max_epochs=5000,
        epoch_length_train=100,
        epoch_length_val=25,
        gradient_accumulation_steps=1,
        # evaluate_on=Events.EPOCH_COMPLETED(every=1),
        enable_amp=True,
    )

    cfg_dataloader = config.DataloaderParameters()

    # =============================================================================
    # DATASETS
    # =============================================================================

    cfg_dataset = config.DatasetParameters(
        train=brainsynth.config.DatasetConfig(
            root_dir=data_dir / "full",
            subject_dir=data_dir / "subject_splits",
            subject_subset=subject_subset_train,
            datasets=datasets,
            images=images_train,
            target_vertices=None,
            template_surface=None,
            exclude_subjects=subject_subset_exclude,
        ),
        validation=brainsynth.config.DatasetConfig(
            root_dir=data_dir / "full",
            subject_dir=data_dir / "subject_splits",
            subject_subset=subject_subset_val,
            datasets=datasets,
            images=images_val,
            target_vertices=None,
            template_surface=None,
            exclude_subjects=subject_subset_exclude,
        ),
    )


    cfg_criterion = None
    cfg_model = None

    mode_contrast = "synth"  # synth, t1w, t2w, flair
    mode_resolution = "random"  # 1mm, random
    random_skullstrip = True
    out_size = [176, 208, 176]
    out_center_str = "brain"

    match mode_resolution:
        case "1mm":
            builder_res = "Iso"
        case "random":
            builder_res = ""
        case _:
            raise ValueError

    builder_contrast = "Synth" if mode_contrast == "synth" else "Select"
    if builder_contrast == "Synth" or random_skullstrip:
        # synth has skullstrip anyway
        builder_train = f"Only{builder_contrast}{builder_res}"
    else:
        builder_train = f"Only{builder_contrast}NoSkullStrip{builder_res}"
    builder_validation = f"OnlySelectNoSkullStrip{builder_res}"


    cfg_synth = config.SynthesizerParameters(
        train=brainsynth.config.SynthesizerConfig(
            builder=builder_train,
            out_size=out_size,
            out_center_str=out_center_str,
            # segmentation_labels = "brainseg"
            # photo_mode = False
            # photo_spacing_range = [2.0, 7.0]
            # photo_thickness = 0.001
            selectable_images=images_train_sel,
            device=device,
        ),
        validation=brainsynth.config.SynthesizerConfig(
            builder=builder_validation,
            out_size=out_size,
            out_center_str=out_center_str,
            # segmentation_labels = "brainseg"
            # photo_mode = False
            # photo_spacing_range = [2.0, 7.0]
            # photo_thickness = 0.001
            selectable_images=images_val_sel,  # images_val
            device=device,
        ),
    )
    cfg_optimizer = config.OptimizerParameters("AdamW", dict(lr=1.0e-4))

    cfg_results = config.ResultsParameters(
        out_dir=out_dir / project / run,
        load_from_dir=model_dir / project / resume_from_run
        if resume_from_run is not None
        else None,
        # save_example_on=Events.EPOCH_COMPLETED(every=1)
    )


    # =============================================================================
    # WANDB
    # =============================================================================

    cfg_wandb = config.WandbParameters(
        enable=True,
        project=project,
        name=run,
        wandb_dir=out_dir / "wandb",
        log_on=cfg_train.evaluate_on,
        run_id=run_id,
        tags=tags,
    )

    return config.TrainSetup(
        project,
        run,
        device,
        cfg_criterion,
        cfg_dataloader,
        cfg_dataset,
        cfg_model,
        cfg_optimizer,
        cfg_results,
        cfg_synth,
        cfg_train,
        cfg_wandb,
    )


def train():

    print("Setting up training...")

    sep_line = 79 * "="

    train_setup = make_train_setup()

    # Overwrite args from command line if provided
    train_setup.train_params.load_checkpoint = 100 # args.load_checkpoint
    train_setup.train_params.max_epochs = 200
    train_setup.wandb.enable = True

    # def dataloader():
    #     while True:
    #         yield None

    # dl = dataloader()
    dataloader = brainnet.initializers.init_dataloader(
        train_setup.dataset, train_setup.dataloader
    )
    model = SuperResolution(device=train_setup.device)
    model = model.to(train_setup.device)

    optimizer = brainnet.initializers.init_optimizer(train_setup.optimizer, model)

    # =============================================================================
    # TRAINING
    # =============================================================================

    train_step = SupervisedTrainingStep(
        model,
        optimizer,
        train_setup.train_params.gradient_accumulation_steps,
        enable_amp=train_setup.train_params.enable_amp,
    )
    trainer = Engine(train_step)

    # The order in which the events are added to the engine is important!

    # Aggregate average loss over epoch
    brainnet.train.utilities.add_metric_to_engine(trainer)
    brainnet.train.utilities.add_terminal_logger(trainer)

    # Add evaluation
    kwargs = dict(
        engine=trainer,
        evaluate_on=train_setup.train_params.evaluate_on,
        epoch_length=train_setup.train_params.epoch_length_val,
    )

    evaluators = dict(
        validation=brainnet.train.utilities.add_evaluation_event(
            EvaluationStep(
                model,
                train_setup.train_params.enable_amp,
            ),
            dataloader=dataloader["validation"],
            logger=event_handlers.MetricLogger(key="loss", name="validation"),
            **kwargs,
        ),
    )

    brainnet.train.utilities.add_wandb_logger(trainer, evaluators, train_setup.wandb)

    # Include this in the checkpoint
    to_save = dict(
        model=model,
        optimizer=optimizer,
        engine=trainer,
    )
    if train_setup.train_params.enable_amp:
        to_save["grad_scaler"] = train_step.grad_scaler

    brainnet.train.utilities.add_model_checkpoint(trainer, to_save, train_setup.results)
    brainnet.train.utilities.write_example_to_disk(trainer, evaluators, train_setup.results)
    brainnet.train.utilities.load_checkpoint_from_setup(to_save, train_setup)

    print("Setup completed. Starting training at epoch ...")

    print(sep_line)
    print(f"Project         {train_setup.project:30s}")
    print(f"Run             {train_setup.run:30s}")
    print(f"Output dir      {train_setup.results.out_dir}")
    print(f"Wandb enabled   {train_setup.wandb.enable}")
    print(sep_line)

    # Start the training
    epoch_length = 100
    # trainer.state.epoch_length = epoch_length
    trainer.run(
        dataloader["train"],
        epoch_length=epoch_length,
        max_epochs=train_setup.train_params.max_epochs,
    )


if __name__ == "__main__":
    #args = brainnet.train.utilities.parse_args(sys.argv)
    train()
