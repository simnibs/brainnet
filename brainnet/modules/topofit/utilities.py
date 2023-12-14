import torch


def grid_sample(image: torch.Tensor, vertices: torch.Tensor):
    """

    Parameters
    ----------
    image :
        image shape is (N, C, W, H, D)
    vertices :
        vertices shape is (N, 3, M)

    Returns
    -------
    samples :
        samples shape (N, C, M)


    """
    # vertices are in voxel coordinates
    # Transform vertices from (0, shape) to (-half_shape, half_shape), then
    # normalize to [-1, 1]
    half_shape = (torch.as_tensor(image.shape[-3:], device=image.device) - 1) / 2
    points = (vertices.mT - half_shape) / half_shape # N,3,M -> N,M,3

    # samples is N,C,D,H,W where C is from `image` and D,H,W are from `points`
    samples = torch.nn.functional.grid_sample(
        image.swapaxes(2,4),        # N,C,W,H,D -> N,C,D,H,W
        points[:, :, None, None],   # N,M,3     -> N,D,H,W,3 where D=M; H=W=1
        align_corners=True
    )
    return samples[..., 0, 0] # squeeze out H, W
