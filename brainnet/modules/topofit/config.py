"""

Differences from the original TopoFit implementation

- we learn the slope using PReLU rather than ReLU with fixed slope.
- we learn the "Euler step size" for the spatial deformation rather than
  specifying it manually

"""

class UnetParameters:
    # manual specification
    channels: int | dict = dict(
        # encoder=[64, 96, 128], ubend=160, decoder=[128, 96, 64]
        # encoder=[96, 128, 160], ubend=192, decoder=[160, 128, 96]
        encoder=[96, 96, 96], ubend=96, decoder=[96, 96, 96]
    )
    multiplier: int = 1

    # automatic
    # channels: Union[int, dict] = 32 # initial number of channels; then 32 * multiplier**level
    # multiplier: int = 2

    # maximum number of levels to recurse in Unet
    n_levels: int = 4

    # number of convolutions per unet level        # GRAPH

    n_conv: int = 1
    reduction: str = "amax" # pool/unpool reduction operation
    conv_module: str = "EdgeConv" # {GraphConv, EdgeConv}

class UnetDeformParameters:
    unet = UnetParameters()

    # Maximum resolution for each deformation block
    # e.g., with `n_levels = 4` the u-nets will look like
    #
    # 0 :       0
    # 1 :     1-0-1
    # 2 :   2-1-0-1-2
    # 3 : 3-2-1-0-1-2-3
    # 4 : 4-3-2-1-2-3-4
    # 5 : 5-4-3-2-3-4-5
    # 6 : 6-5-4-3-4-5-6
    resolutions: list[int] = [0, 1, 2, 3, 4, 5, 6]

    # scaling of the deformation vector
    # euler_step_size: list[float] = [1, 1, 1, 1, 1, 1, 1]
    # euler_step_size: list[float] = [10, 10, 10, 10, 1, 1, 1]
    # euler_step_size: list[float] = [0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]

    # number of iterations per resolution
    euler_iterations: list[int] = [1, 1, 1, 1, 1, 2, 1]

    conv_module: str = "EdgeConv" # {GraphConv, EdgeConv}


class LinearDeformParameters:
    # white -> layer 4 -> pial
    #     (5 it)     (5 it)
    channels: list[int] = [32]
    n_iterations: int = 5


class TopoFitModelParameters:
    # GraphDeformation:
    #     include_positions: False # whether to include vertex positions to each graph block

    unet_deform = UnetDeformParameters()
    linear_deform = LinearDeformParameters()

# class TopoFitTrainParameters:

#     steps_per_epoch: int = 100
#     validate_every_n_epoch: int = 20
#     checkpoint_every_n_epoch: int = 100

#     initial_lr: float = 1.0e-4