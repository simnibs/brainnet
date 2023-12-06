#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


#define SET_POINT(target, array, index)     \
    target[0] = array[index];      \
    target[1] = array[index + 1];  \
    target[2] = array[index + 2];


#define SQUARE_DISTANCE(distance, p1, p2)  \
    scalar_t delta;             \
    distance = 0;               \
    delta = p1[0] - p2[0];      \
    distance += delta * delta;  \
    delta = p1[1] - p2[1];      \
    distance += delta * delta;  \
    delta = p1[2] - p2[2];      \
    distance += delta * delta;


template <typename scalar_t>
__global__ void NearestNeighborKernel(
    const scalar_t* __restrict__ point_set_1,
    const scalar_t* __restrict__ point_set_2,
    const int64_t* __restrict__ num_points_1,
    const int64_t* __restrict__ num_points_2,
    int64_t* __restrict__ nearest,
    const size_t L1,
    const size_t L2,
    const size_t B)
{
    scalar_t point_a[3];
    scalar_t point_b[3];

    const int64_t chunks_per_set = (1 + (L1 - 1) / blockDim.x);
    const int64_t chunks = B * chunks_per_set;

    for (int64_t chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {

        const int64_t batch = chunk / chunks_per_set;
        const int64_t start = blockDim.x * (chunk % chunks_per_set);
        int64_t current_index = start + threadIdx.x;

        if (current_index >= num_points_1[batch]) continue;

        SET_POINT(point_a, point_set_1, batch * L1 * 3 + current_index * 3)

        scalar_t min_distance = 0;
        int64_t min_index = 0;

        int64_t npoints = num_points_2[batch];
        for (int64_t i = 0; i < npoints; ++i) {

            scalar_t distance;
            SET_POINT(point_b, point_set_2, batch * L2 * 3 + i * 3)
            SQUARE_DISTANCE(distance, point_a, point_b)

            if ((distance < min_distance) || (i ==0)) {
                min_distance = distance;
                min_index = i;
            }
        }

        nearest[batch * L1 + current_index] = min_index;
    }
}


at::Tensor NearestNeighbor(
    const at::Tensor& point_set_1,
    const at::Tensor& point_set_2,
    const at::Tensor& num_points_1,
    const at::Tensor& num_points_2)
{
    at::TensorArg
        p1{point_set_1,  "point_set_1",  1},
        p2{point_set_2,  "point_set_2",  2},
        n1{num_points_1, "num_points_1", 3},
        n2{num_points_2, "num_points_2", 4};

    at::CheckedFrom c = "NearestNeighbor";
    at::checkAllSameGPU(c, {p1, p2, n1, n2});
    at::checkAllSameType(c, {p1, p2});

    at::cuda::CUDAGuard device_guard(point_set_1.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const auto B  = point_set_1.size(0);
    const auto L1 = point_set_1.size(1);
    const auto L2 = point_set_2.size(1);

    TORCH_CHECK(point_set_2.size(0) == B, "Point sets must have the same batch size");
    TORCH_CHECK(point_set_1.size(2) == 3, "Point sets must be 3D");
    TORCH_CHECK(point_set_2.size(2) == 3, "Point sets must be 3D");
    TORCH_CHECK(num_points_1.size(0) == B, "Num points must have the same size");
    TORCH_CHECK(num_points_2.size(0) == B, "Num points must have the same size");

    auto dtype = num_points_1.options().dtype(at::kLong);
    auto nearest = at::zeros({B, L1}, dtype);

    const size_t threads = 256;
    const size_t blocks = 256;
    AT_DISPATCH_FLOATING_TYPES(point_set_1.scalar_type(), "NearestNeighborKernel", [&] {
        NearestNeighborKernel<scalar_t><<<blocks, threads, 0, stream>>>(
            point_set_1.contiguous().data_ptr<scalar_t>(),
            point_set_2.contiguous().data_ptr<scalar_t>(),
            num_points_1.data_ptr<int64_t>(),
            num_points_2.data_ptr<int64_t>(),
            nearest.data_ptr<int64_t>(),
            L1, L2, B);
    });

    AT_CUDA_CHECK(cudaGetLastError());
    return nearest;
}
