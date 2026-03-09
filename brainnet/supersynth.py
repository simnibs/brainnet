import os
import numpy as np
import torch
from torch.nn.functional import (
    max_pool3d,
    conv3d,
    group_norm,
    leaky_relu,
    interpolate,
)


def get_label_lists_etc():
    # Just defines a bunch of constants
    # fmt: off
    label_list_segmentation_whole_freesurfer = [0, 14, 15, 16, 24, 77, 85, 99, 901, 902, 906, 907, 908, 909, 911, 912, 914, 915, 916, 930, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 819, 821, 843, 865, 869, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 820, 822, 844, 866, 870]
    label_list_segmentation_exvivo_freesurfer = [0, 14, 15, 16, 77, 85, 99, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 819, 821, 843, 865, 869, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 820, 822, 844, 866, 870]
    label_list_segmentation_cerebrum_freesurfer = [0, 77, 85, 99, 2, 3, 4, 5, 10, 11, 12, 13, 17, 18, 26, 819, 821, 843, 865, 869, 41, 42, 43, 44, 49, 50, 51, 52, 53, 54, 58, 820, 822, 844, 866, 870]
    label_list_segmentation_hemi_freesurfer_left = [0, 2, 3, 4, 5, 10, 11, 12, 13, 17, 18, 26, 77, 99, 819, 821, 843, 865, 869]
    label_list_segmentation_hemi_freesurfer_right = [0, 41, 42, 43, 44, 49, 50, 51, 52, 53, 54, 58, 77, 99, 820, 822, 844, 866, 870]
    label_list_segmentation_whole = [0, 11, 12, 13, 16, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 17, 47, 49, 51, 53, 55, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 48, 50, 52, 54, 56]
    label_list_segmentation_hemis = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    label_list_segmentation_exvivo = [0, 11, 12, 13, 31, 32, 33, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 17, 34, 36, 38, 40, 42, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 37, 39, 41, 43]
    # fmt: on

    n_neutral_labels_whole = 20
    n_neutral_labels_hemis = len(label_list_segmentation_hemis)
    n_neutral_labels_exvivo = 7
    n_neutral_labels_cerebrum = 4
    n_labels_whole = len(label_list_segmentation_whole)
    n_labels_hemis = len(label_list_segmentation_hemis)
    n_labels_exvivo = len(label_list_segmentation_exvivo)
    n_labels_cerebrum = len(label_list_segmentation_cerebrum_freesurfer)
    nlat = int((n_labels_whole - n_neutral_labels_whole) / 2.0)
    vflip_invivo = np.concatenate(
        [
            np.array(range(n_neutral_labels_whole)),
            np.array(range(n_neutral_labels_whole + nlat, n_labels_whole)),
            np.array(range(n_neutral_labels_whole, n_neutral_labels_whole + nlat)),
        ]
    )

    nlat = int((len(label_list_segmentation_exvivo) - n_neutral_labels_exvivo) / 2.0)
    vflip_exvivo = np.concatenate(
        [
            np.array(range(n_neutral_labels_exvivo)),
            np.array(
                range(
                    n_neutral_labels_exvivo + nlat, len(label_list_segmentation_exvivo)
                )
            ),
            np.array(range(n_neutral_labels_exvivo, n_neutral_labels_exvivo + nlat)),
        ]
    )
    nlat = int(
        (len(label_list_segmentation_cerebrum_freesurfer) - n_neutral_labels_cerebrum)
        / 2.0
    )
    vflip_cerebrum = np.concatenate(
        [
            np.array(range(n_neutral_labels_cerebrum)),
            np.array(
                range(
                    n_neutral_labels_cerebrum + nlat,
                    len(label_list_segmentation_cerebrum_freesurfer),
                )
            ),
            np.array(
                range(n_neutral_labels_cerebrum, n_neutral_labels_cerebrum + nlat)
            ),
        ]
    )
    # fmt: off
    list_to_kill_photo_whole = [5, 6, 11, 12, 13, 16, 22, 23, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46]
    # fmt: on

    return (
        label_list_segmentation_whole_freesurfer,
        label_list_segmentation_exvivo_freesurfer,
        label_list_segmentation_cerebrum_freesurfer,
        label_list_segmentation_hemi_freesurfer_left,
        label_list_segmentation_hemi_freesurfer_right,
        label_list_segmentation_whole,
        label_list_segmentation_hemis,
        label_list_segmentation_exvivo,
        n_neutral_labels_whole,
        n_neutral_labels_hemis,
        n_neutral_labels_exvivo,
        n_neutral_labels_cerebrum,
        n_labels_whole,
        n_labels_hemis,
        n_labels_exvivo,
        n_labels_cerebrum,
        vflip_invivo,
        vflip_exvivo,
        vflip_cerebrum,
        list_to_kill_photo_whole,
    )


class frugal_models:
    def __init__(self, model_file, device):
        self.device = device
        if device == "cpu":
            cp = torch.load(model_file, map_location=torch.device("cpu"))
        else:
            cp = torch.load(model_file)

        # SuperSynth part
        self.ssynth_bbone = cp["backbone_state_dict"]
        for key in self.ssynth_bbone.keys():
            self.ssynth_bbone[key] = self.ssynth_bbone[key].to(device)
        self.ssynth_final_conv_names = [
            "reg",
            "seg",
            "T1",
            "T2",
            "FLAIR",
            "LP",
            "LW",
            "RP",
            "RW",
        ]
        self.ssynth_final_conv_weight = {}
        self.ssynth_final_conv_bias = {}
        for name in self.ssynth_final_conv_names:
            self.ssynth_final_conv_weight[name] = cp[name + "_state_dict"]["weight"].to(
                device
            )
            self.ssynth_final_conv_bias[name] = cp[name + "_state_dict"]["bias"].to(
                device
            )

        # AutoQC part (Billot et al, PNAS, 2023)
        for key in cp.keys():
            if key.endswith("_state_dict") is False:
                setattr(self, key, cp[key].to(device))
        # fmt: off
        labels_segmentation = np.array(
            [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52,
             53, 54, 58, 77, 85, 99, 819, 820, 821, 822, 843, 844, 865, 866, 869, 870, 901, 902, 906, 907, 908, 909,
             911, 912, 914, 915, 916, 930], dtype=np.int32)
        labels_qc = np.array(
            [0, 1, 2, 3, 3, 4, 4, 6, 1, 7, 7, 3, 3, 5, 8, 8, 0, 1, 1, 2, 3, 3, 4, 4, 6, 1, 7, 7, 8, 8, 1, 1, 1, 0, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        # fmt: on
        self.lut = torch.zeros(1000, dtype=torch.int32, device=device)
        for i in range(len(labels_qc)):
            self.lut[labels_segmentation[i]] = labels_qc[i]
        self.onehotlut = torch.eye(9, dtype=torch.float32, device=device)

        # Clean up
        del cp

    def ssynth_inference(self, input):
        enc_feat_maps = []
        for level in range(5):
            x = input if level == 0 else max_pool3d(enc_feat_maps[-1], 2)
            x = conv3d(
                x,
                self.ssynth_bbone[
                    "encoders." + str(level) + ".basic_module.conv1.weight"
                ],
                self.ssynth_bbone[
                    "encoders." + str(level) + ".basic_module.conv1.bias"
                ],
            )
            x = conv3d(
                x,
                self.ssynth_bbone[
                    "encoders." + str(level) + ".basic_module.conv2.conv.weight"
                ],
                padding=1,
            )
            x = group_norm(
                x,
                8,
                weight=self.ssynth_bbone[
                    "encoders." + str(level) + ".basic_module.conv2.groupnorm.weight"
                ],
                bias=self.ssynth_bbone[
                    "encoders." + str(level) + ".basic_module.conv2.groupnorm.bias"
                ],
            )  # , eps=1e-05)
            x = leaky_relu(x, negative_slope=0.01, inplace=True)
            x = conv3d(
                x,
                self.ssynth_bbone[
                    "encoders." + str(level) + ".basic_module.conv3.conv.weight"
                ],
                padding=1,
            )
            x = group_norm(
                x,
                8,
                weight=self.ssynth_bbone[
                    "encoders." + str(level) + ".basic_module.conv3.groupnorm.weight"
                ],
                bias=self.ssynth_bbone[
                    "encoders." + str(level) + ".basic_module.conv3.groupnorm.bias"
                ],
            )  # , eps=1e-05)
            # next line is a bit inefficient but saves memory!
            x += conv3d(
                input if level == 0 else max_pool3d(enc_feat_maps[-1], 2),
                self.ssynth_bbone[
                    "encoders." + str(level) + ".basic_module.conv1.weight"
                ],
                self.ssynth_bbone[
                    "encoders." + str(level) + ".basic_module.conv1.bias"
                ],
            )
            x = leaky_relu(x, negative_slope=0.1, inplace=True)
            enc_feat_maps.append(x)
            del x
            torch.cuda.empty_cache()

        for level in range(4):
            # Here's the real memory bottleneck that we overcome with ugly code
            idx = enc_feat_maps[-2].shape[1]
            nc = 24  # how many channels at a time
            enc_feat_maps[-2] = conv3d(
                enc_feat_maps[-2],
                self.ssynth_bbone[
                    "decoders." + str(level) + ".basic_module.SingleConv1.conv.weight"
                ][:, :idx],
                padding=1,
            )
            newshape = enc_feat_maps[-2].shape[2:]
            for c in range(0, enc_feat_maps[-1].shape[1], nc):
                enc_feat_maps[-2] += conv3d(
                    interpolate(
                        enc_feat_maps[-1][:, c : c + nc], size=newshape, mode="nearest"
                    ),
                    self.ssynth_bbone[
                        "decoders."
                        + str(level)
                        + ".basic_module.SingleConv1.conv.weight"
                    ][:, idx + c : idx + nc + c],
                    padding=1,
                )

            del enc_feat_maps[-1]  # now, enc_feat_maps[-2] is enc_feat_maps[-1]
            torch.cuda.empty_cache()
            enc_feat_maps[-1] = group_norm(
                enc_feat_maps[-1],
                8,
                weight=self.ssynth_bbone[
                    "decoders."
                    + str(level)
                    + ".basic_module.SingleConv1.groupnorm.weight"
                ],
                bias=self.ssynth_bbone[
                    "decoders."
                    + str(level)
                    + ".basic_module.SingleConv1.groupnorm.bias"
                ],
                eps=1e-05,
            )
            enc_feat_maps[-1] = leaky_relu(
                enc_feat_maps[-1], negative_slope=0.01, inplace=True
            )
            enc_feat_maps[-1] = conv3d(
                enc_feat_maps[-1],
                self.ssynth_bbone[
                    "decoders." + str(level) + ".basic_module.SingleConv2.conv.weight"
                ],
                padding=1,
            )
            enc_feat_maps[-1] = group_norm(
                enc_feat_maps[-1],
                8,
                weight=self.ssynth_bbone[
                    "decoders."
                    + str(level)
                    + ".basic_module.SingleConv2.groupnorm.weight"
                ],
                bias=self.ssynth_bbone[
                    "decoders."
                    + str(level)
                    + ".basic_module.SingleConv2.groupnorm.bias"
                ],
                eps=1e-05,
            )
            enc_feat_maps[-1] = leaky_relu(
                enc_feat_maps[-1], negative_slope=0.01, inplace=True
            )

        outputs = {}
        for name in self.ssynth_final_conv_names:
            outputs[name] = conv3d(
                enc_feat_maps[0],
                self.ssynth_final_conv_weight[name],
                self.ssynth_final_conv_bias[name],
            )

        del enc_feat_maps
        torch.cuda.empty_cache()

        return outputs


def supersynth_init():
    device = "cuda"
    # Set up CPU/GPU and threads
    threads = os.cpu_count()
    torch.set_num_threads(threads)

    model_file = (
        "/mnt/projects/CORTECH/nobackup/jesper/supersynth/SuperSynth_August_2025.pth"
    )

    flipping = False
    mode = "exvivo"

    # some constants
    (
        label_list_segmentation_whole_freesurfer,
        label_list_segmentation_exvivo_freesurfer,
        label_list_segmentation_cerebrum_freesurfer,
        label_list_segmentation_hemi_freesurfer_left,
        label_list_segmentation_hemi_freesurfer_right,
        label_list_segmentation_whole,
        label_list_segmentation_hemis,
        label_list_segmentation_exvivo,
        n_neutral_labels_whole,
        n_neutral_labels_hemis,
        n_neutral_labels_exvivo,
        n_neutral_labels_cerebrum,
        n_labels_whole,
        n_labels_hemis,
        n_labels_exvivo,
        n_labels_cerebrum,
        vflip_invivo,
        vflip_exvivo,
        vflip_cerebrum,
        list_to_kill_photo_whole,
    ) = get_label_lists_etc()

    mask_photo_or_cerebrum_whole = torch.ones(
        len(label_list_segmentation_whole), dtype=torch.bool, device=device
    )
    for label in range(len(label_list_segmentation_whole)):
        if (
            np.sum(
                np.array(list_to_kill_photo_whole)
                == label_list_segmentation_whole[label]
            )
            > 0
        ):
            mask_photo_or_cerebrum_whole[label] = False
    v_left = []
    for lab in label_list_segmentation_hemi_freesurfer_left:
        v_left.append(
            np.where(np.array(label_list_segmentation_whole_freesurfer) == lab)[0][0]
        )
    v_left = torch.tensor(v_left, device=device, dtype=torch.int32)
    v_right = []
    for lab in label_list_segmentation_hemi_freesurfer_right:
        v_right.append(
            np.where(np.array(label_list_segmentation_whole_freesurfer) == lab)[0][0]
        )
    v_right = torch.tensor(v_right, device=device, dtype=torch.int32)
    mask_exvivo_whole = torch.ones(
        len(label_list_segmentation_whole), dtype=torch.bool, device=device
    )
    for label in range(len(label_list_segmentation_whole)):
        if (
            np.sum(
                np.array(label_list_segmentation_exvivo_freesurfer)
                == label_list_segmentation_whole_freesurfer[label]
            )
            == 0
        ):
            mask_exvivo_whole[label] = False

    # Load FreeSurfer labels
    d = {}
    with open(os.getenv("FREESURFER_HOME") + "/FreeSurferColorLUT.txt") as f:
        for label in f:
            if label.strip() and not label.lstrip().startswith("#"):
                p = label.split()
                if len(p) > 1 and p[0].isdigit():
                    d[int(p[0])] = p[1]
    FSlabelNames = [None] * (max(d) + 1) if d else []
    for k, v in d.items():
        FSlabelNames[k] = v

    return frugal_models(model_file, device)


def supersynth_pred(model, image, mode="exvivo"):
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model.ssynth_inference(image)

            # im, aff = torch_resize(im, aff, 1.0, device)
            # im, aff = align_volume_to_ref(
            #     im, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3
            # )
            # im /= im.max()
            # W = (np.ceil(np.array(im.shape) / 16.0) * 16).astype("int")
            # idx = np.floor((W - im.shape) / 2).astype("int")
            # S = torch.zeros(*W, dtype=torch.float32, device=device)
            # S[
            #     idx[0] : idx[0] + im.shape[0],
            #     idx[1] : idx[1] + im.shape[1],
            #     idx[2] : idx[2] + im.shape[2],
            # ] = im

            LP = pred["LP"]
            LW = pred["LW"]
            RP = pred["RP"]
            RW = pred["RW"]

            """
            # SEGMENTATION

            seg = pred["seg"]
            if mode == "cerebrum":
                seg = seg[:, mask_photo_or_cerebrum_whole]
            elif mode == "left-hemi":
                seg = seg[:, v_left]
            elif mode == "right-hemi":
                seg = seg[:, v_right]
            elif mode == "exvivo":
                seg = seg[:, mask_exvivo_whole]
            elif mode == "invivo":
                pass
            else:
                raise Exception("mode not supported: " + mode)
            seg = softmax(seg, dim=1)  # segmentations are activations, at this point

            # if flipping goes here...

            # Discretize segmentations
            if mode == "cerebrum":
                seg_discrete = torch.tensor(
                    label_list_segmentation_whole_freesurfer, device=device
                )[mask_photo_or_cerebrum_whole][torch.argmax(seg, 0)]
            elif mode == "left-hemi":
                seg_discrete = torch.tensor(
                    label_list_segmentation_hemi_freesurfer_left, device=device
                )[torch.argmax(seg, 0)]
            elif mode == "right-hemi":
                seg_discrete = torch.tensor(
                    label_list_segmentation_hemi_freesurfer_right, device=device
                )[torch.argmax(seg, 0)]
            elif mode == "exvivo":
                seg_discrete = torch.tensor(
                    label_list_segmentation_exvivo_freesurfer, device=device
                )[torch.argmax(seg, 0)]
            elif mode == "invivo":
                seg_discrete = torch.tensor(
                    label_list_segmentation_whole_freesurfer, device=device
                )[torch.argmax(seg, 0)]
            else:
                raise Exception("mode not supported: " + mode)
            """

            # print("   Postprocessing cortical ribbon")
            a = 2
            max_surf_distance = 3.0
            LW = torch.clamp(LW, min=-max_surf_distance, max=max_surf_distance)
            RW = torch.clamp(RW, min=-max_surf_distance, max=max_surf_distance)
            LP = torch.clamp(LP, min=-max_surf_distance, max=max_surf_distance)
            RP = torch.clamp(RP, min=-max_surf_distance, max=max_surf_distance)
            ribbonL = 70 * (1 - (torch.tanh(a * (LW + 0.3)) + 1) / 2) + 40 * (
                1 - (torch.tanh(a * LP) + 1) / 2
            )
            ribbonR = 70 * (1 - (torch.tanh(a * (RW + 0.3)) + 1) / 2) + 40 * (
                1 - (torch.tanh(a * RP) + 1) / 2
            )
            if mode == "left-hemi":
                ribbon = ribbonL
            elif mode == "right-hemi":
                ribbon = ribbonR
            else:
                ribbon = torch.maximum(ribbonL, ribbonR)

            """
            # get masks for fiting deformations and postprocessing segmentations
            M = (
                (seg_discrete > 0) & (seg_discrete != 24) & (seg_discrete < 900)
            )  # useful for later

            M = get_largest_connected_component(M.detach().cpu().numpy())
            Mdilated = binary_dilation(M, iterations=2)

            M = torch.tensor(M, device=device, dtype=torch.bool)
            Mdilated = torch.tensor(Mdilated, device=device, dtype=torch.bool)

            if mode != "invivo":
                seg_discrete[~M] = 0
            ribbon[~Mdilated] = 0

            # postprocess soft segmentations and compute volumes
            seg[0][~Mdilated] = 1
            for label in range(seg.shape[0]):
                seg[label][~Mdilated] = 0
            """

    return ribbon.clip(0, 255)  # normalize to 0-1 !


# if flipping:
#     print("   Pushing flipped data through the CNN")
#     image = torch.flip(image, [2])
#     pred = model.ssynth_inference(image)

#     LP = (
#         0.5 * LP
#         + 0.5
#         * torch.flip(pred["RP"][0, 0, ...], [0])[
#             idx[0] : idx[0] + im.shape[0],
#             idx[1] : idx[1] + im.shape[1],
#             idx[2] : idx[2] + im.shape[2],
#         ]
#     )
#     LW = (
#         0.5 * LW
#         + 0.5
#         * torch.flip(pred["RW"][0, 0, ...], [0])[
#             idx[0] : idx[0] + im.shape[0],
#             idx[1] : idx[1] + im.shape[1],
#             idx[2] : idx[2] + im.shape[2],
#         ]
#     )
#     RP = (
#         0.5 * RP
#         + 0.5
#         * torch.flip(pred["LP"][0, 0, ...], [0])[
#             idx[0] : idx[0] + im.shape[0],
#             idx[1] : idx[1] + im.shape[1],
#             idx[2] : idx[2] + im.shape[2],
#         ]
#     )
#     RW = (
#         0.5 * RW
#         + 0.5
#         * torch.flip(pred["LW"][0, 0, ...], [0])[
#             idx[0] : idx[0] + im.shape[0],
#             idx[1] : idx[1] + im.shape[1],
#             idx[2] : idx[2] + im.shape[2],
#         ]
#     )
#     activations = torch.flip(pred["seg"][0, ...], [1])[
#         :,
#         idx[0] : idx[0] + im.shape[0],
#         idx[1] : idx[1] + im.shape[1],
#         idx[2] : idx[2] + im.shape[2],
#     ]
#     if mode == "cerebrum":
#         activations = activations[mask_photo_or_cerebrum_whole, ...]
#         activations = activations[vflip_cerebrum, ...]
#     elif mode == "left-hemi":
#         activations = activations[v_right, ...]
#     elif mode == "right-hemi":
#         activations = activations[v_left, ...]
#     elif mode == "exvivo":
#         activations = activations[mask_exvivo_whole, ...]
#         activations = activations[vflip_exvivo, ...]
#     else:  # 'invivo':
#         activations = activations[vflip_invivo, ...]
#     seg = 0.5 * seg + 0.5 * softmax(activations, dim=0)
