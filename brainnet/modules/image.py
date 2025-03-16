import itertools

import torch
from brainnet.modules.blocks import ConvolutionBlock

class SuperResolution(torch.nn.Module):
    def __init__(self, in_channels: int = 1, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(device)

        self.transform = torch.nn.Sequential(
            ConvolutionBlock(3, in_channels, 32, norm=False),
            ConvolutionBlock(3, 32, 32, norm=False),
            ConvolutionBlock(3, 32, 32, norm=False),
            ConvolutionBlock(3, 32, 32, norm=False),
        )
        self.transform_final = ConvolutionBlock(3, 32, 6, norm=False, activation=False)
        self.subpixel_conv = SubpixelConvolution(up_factor=6, up_dims=1, spatial_dims=3)

    def forward(self, image):
        features = self.transform(image)
        features = self.transform_final(features)
        subpixel = self.subpixel_conv(features)
        return subpixel

# class DenseConvolution(torch.nn.Module):
#     def __init__(self, in_channels: int = 1,*args, **kwargs):
#         super().__init__(*args, **kwargs)

#         n_levels = 5
#         out_channels = 16

#         self.image_to_feature = ConvolutionBlock(3, in_channels, out_channels, norm=False)
#         self.transform = torch.nn.ModuleList()
#         in_channels = out_channels
#         for i in range(1,n_levels+1):
#             in_channels *= i
#             self.transform.append(ConvolutionBlock(3, in_channels, out_channels, norm=False))


#         ConvolutionBlock(3, 1, 16, norm=False)

#         ConvolutionBlock(3, 16, 16, norm=False)
#         ConvolutionBlock(3, 32, 16, norm=False)
#         ConvolutionBlock(3, 48, 16, norm=False)
#         ConvolutionBlock(3, 64, 16, norm=False)

#     def forward(self, image):
#         all_features = self.image_to_feature(image)
#         for conv in self.transform:
#             all_features = torch.cat((all_features, conv(all_features)), dim=1)

class SubpixelConvolution(torch.nn.Module):
    def __init__(self, up_factor: int = 2, spatial_dims: int = 3, up_dims: int | None = None, *args, **kwargs) -> None:
        """Sub-pixel convolution from Shi (2016).

        up_factor=2; spatial_dims=2, up_dims=2
            Upsample the last two (all) dimensions of a 2D array by a factor of 2.
        up_factor=2; spatial_dims=3, up_dims=3
            Upsample the last three (all) dimensions of a 3D array by a factor
            of 2.
        up_factor=2; spatial_dims=3, up_dims=2
            Upsample the last two dimensions of a 3D array by a factor of 2.


        Parameters
        ----------
        up_factor : int, optional
            Upscaling factor.
        spatial_dims : int, optional
            Number of spatial dimensions of the input.
        up_dims : int, optional
            Number of dimensions to up-scale. If up_dims < spatial_dims, only
            the *last* `up_dims` number of spatial dimensions are upsampled.

        Examples
        --------
        spatial_dims = 2; up_dims = 1

            x = torch.rand((1,2,3,3))
            y = torch.zeros((1,1,3,6))
            y[..., 0::2] = x[:,0]
            y[..., 1::2] = x[:,1]

        spatial_dims = 2; up_dims = 2

            x = torch.rand((1,4,3,3))
            y = torch.zeros((1,1,6,6))
            y[..., 0::2, 0::2] = x[:,0]
            y[..., 0::2, 1::2] = x[:,1]
            y[..., 1::2, 0::2] = x[:,2]
            y[..., 1::2, 1::2] = x[:,3]

        spatial_dims = 3; up_dims = 3

            x = torch.rand((1,8,2,2,2))
            y = torch.zeros((1,1,4,4,4))
            y[..., 0::2, 0::2, 0::2] = x[:,0]
            y[..., 0::2, 0::2, 1::2] = x[:,1]
            y[..., 0::2, 1::2, 0::2] = x[:,2]
            y[..., 0::2, 1::2, 1::2] = x[:,3]
            y[..., 1::2, 0::2, 0::2] = x[:,4]
            y[..., 1::2, 0::2, 1::2] = x[:,5]
            y[..., 1::2, 1::2, 0::2] = x[:,6]
            y[..., 1::2, 1::2, 1::2] = x[:,7]

        References
        ----------
        Shi (2016). Real-Time Single Image and Video Super-Resolution Using an
            Efficient Sub-Pixel Convolutional Neural Network.
            https://arxiv.org/pdf/1609.05158
        Georgescu (2023). Convolutional Neural Networks with Intermediate Loss
            for 3D Super-Resolution of CT and MRI Scans.
            https://arxiv.org/pdf/2001.01330

        Bouter (2022). Deep learning-based single image super-resolution for
            low-field MR brain images.
            https://www.nature.com/articles/s41598-022-10298-6
        Tong (2017). Image Super-Resolution Using Dense Skip Connections.
            https://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf
        """
        super().__init__(*args, **kwargs)
        up_dims = spatial_dims if up_dims is None else up_dims
        self.up_scales = tuple(up_factor if i < up_dims else 1 for i in range(spatial_dims-1,-1,-1))
        offsets = tuple(itertools.product(*[tuple(range(s)) for s in self.up_scales]))
        self.slices = tuple(tuple(slice(j,None,s) for j,s in zip(i,self.up_scales)) for i in offsets )


    def forward(self, image):
        """

        up_dims = 3

        image = B,C,H,W,D
        out = B,1,H*i,W*j,D*k




        Parameters
        ----------
        image : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        B,C,*SpaDim = image.size()
        up_size = (B,1,*tuple(i*j for i,j in zip(self.up_scales, SpaDim)))

        y = torch.zeros(up_size, dtype=image.dtype, device=image.device)

        for i,sl in enumerate(self.slices):
            y[..., *sl] = image[:, i]

        return y



# x = torch.rand((1,2,3,3))
# y = torch.zeros((1,1,3,6))
# y[..., 0::2] = x[:,0]
# y[..., 1::2] = x[:,1]


# x = torch.rand((1,2,3,3))
# y = torch.zeros((1,1,6,3))
# y[:, :, 0::2, :] = x[:,0]
# y[:, :, 1::2, :] = x[:,1]


# x = torch.rand((1,4,3,3))
# y = torch.zeros((1,1,6,6))
# y[..., 0::2, 0::2] = x[:,0]
# y[..., 0::2, 1::2] = x[:,1]
# y[..., 1::2, 0::2] = x[:,2]
# y[..., 1::2, 1::2] = x[:,3]


# x = torch.zeros((1,4,3,3))
# x[:,1] = 1
# x[:,2] = 2
# x[:,3] = 3
# y = torch.zeros((1,1,6,6))
# y[..., 0::2, 0::2] = 0
# y[..., 0::2, 1::2] = 1
# y[..., 1::2, 0::2] = 2
# y[..., 1::2, 1::2] = 3
# tensor([[[[0.8261, 0.2330, 0.7862, 0.4855, 0.2718, 0.0514],
#           [0.3205, 0.6045, 0.6260, 0.9715, 0.8284, 0.5922],
#           [0.0045, 0.8607, 0.4609, 0.7674, 0.2831, 0.0846],
#           [0.1344, 0.0514, 0.8431, 0.8518, 0.5874, 0.0971],
#           [0.4749, 0.4500, 0.3447, 0.7561, 0.6612, 0.8569],
#           [0.0617, 0.8129, 0.5612, 0.9751, 0.3276, 0.7836]]]])

# x.reshape(1,1,6,6)


# x = torch.zeros((1,8,2,2,2))
# for i in range(8):
#     x[:,i] = i

# y = torch.zeros((1,1,4,4,4))

# for i in range(2):
#     for j in range(2):
#         for k in range(2):
#             y[..., i::2, j::2, k::2] = x[:, int(f"{i}{j}{k}", 2)]