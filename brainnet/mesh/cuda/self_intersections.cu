#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


__constant__ const float EPSILON = 0.000001;


#define FABS(x) fabs(x)


#define VEC3_CROSS(dest, v1, v2)             \
    dest[0] = v1[1] * v2[2] - v1[2] * v2[1]; \
    dest[1] = v1[2] * v2[0] - v1[0] * v2[2]; \
    dest[2] = v1[0] * v2[1] - v1[1] * v2[0];


#define VEC3_DOT(v1, v2) (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])


#define VEC3_SUB(dest, v1, v2) dest[0] = v1[0] - v2[0]; dest[1] = v1[1] - v2[1]; dest[2] = v1[2] - v2[2];


#define VEC3_ADD(dest, v1, v2) dest[0] = v1[0] + v2[0]; dest[1] = v1[1] + v2[1]; dest[2] = v1[2] + v2[2];


#define VEC3_MULT(dest, v, factor) dest[0] = factor * v[0]; dest[1] = factor * v[1]; dest[2] = factor * v[2];


#define VEC3_DIV(dest, v, factor) dest[0] = v[0] / factor; dest[1] = v[1] / factor; dest[2] = v[2] / factor;


#define VEC3_CONTAINS(v, item) ((item == v[0]) || (item == v[1]) || (item == v[2]))


#define VEC3_PUT(target, index, vector) \
    target[index]     = vector[0];      \
    target[index + 1] = vector[1];      \
    target[index + 2] = vector[2];


#define VEC3_GET(target, array, index)  \
    target[0] = array[index];      \
    target[1] = array[index + 1];  \
    target[2] = array[index + 2];


#define VEC3_SQUARE_DISTANCE(distance, p1, p2)  \
    float delta;             \
    distance = 0;               \
    delta = p1[0] - p2[0];      \
    distance += delta * delta;  \
    delta = p1[1] - p2[1];      \
    distance += delta * delta;  \
    delta = p1[2] - p2[2];      \
    distance += delta * delta;


#define SORT(a, b) \
    if (a > b)     \
    {              \
        float c;   \
        c = a;     \
        a = b;     \
        b = c;     \
    }


#define ISECT(VV0, VV1, VV2, D0, D1, D2, isect0, isect1) \
    isect0 = VV0 + (VV1 - VV0) * D0 / (D0 - D1);         \
    isect1 = VV0 + (VV2 - VV0) * D0 / (D0 - D2);


#define COMPUTE_INTERVALS(VV0, VV1, VV2, D0, D1, D2, D0D1, D0D2, isect0, isect1) \
    if (D0D1 > 0.0f)                                         \
    {                                                        \
        ISECT(VV2, VV0, VV1, D2, D0, D1, isect0, isect1);    \
    }                                                        \
    else if (D0D2 > 0.0f)                                    \
    {                                                        \
        ISECT(VV1, VV0, VV2, D1, D0, D2, isect0, isect1);    \
    }                                                        \
    else if (D1 * D2 > 0.0f || D0 != 0.0f)                   \
    {                                                        \
        ISECT(VV0, VV1, VV2, D0, D1, D2, isect0, isect1);    \
    }                                                        \
    else if (D1 != 0.0f)                                     \
    {                                                        \
        ISECT(VV1, VV0, VV2, D1, D0, D2, isect0, isect1);    \
    }                                                        \
    else if (D2 != 0.0f )                                    \
    {                                                        \
        ISECT(VV2, VV0, VV1, D2, D0, D1, isect0, isect1);    \
    }                                                        \
    else                                                     \
    {                                                        \
        return coplanerTriTri(N1, V0, V1, V2, U0, U1, U2);   \
    }


#define EDGE_EDGE_TEST(V0, U0, U1)                                   \
    Bx = U0[i0] - U1[i0];                                            \
    By = U0[i1] - U1[i1];                                            \
    Cx = V0[i0] - U0[i0];                                            \
    Cy = V0[i1] - U0[i1];                                            \
    f = Ay * Bx - Ax * By;                                           \
    d = By * Cx - Bx * Cy;                                           \
    if ((f > 0 && d >= 0 && d <= f) || (f < 0 && d <= 0 && d >= f))  \
    {                                                                \
        e = Ax * Cy - Ay * Cx;                                       \
        if (f > 0)                                                   \
        {                                                            \
            if (e >= 0 && e <= f) return 1;                          \
        }                                                            \
        else                                                         \
        {                                                            \
            if (e <= 0 && e >= f) return 1;                          \
        }                                                            \
    }


#define EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2) \
{                                              \
    float Ax, Ay, Bx, By, Cx, Cy, e, d, f;     \
    Ax = V1[i0] - V0[i0];                      \
    Ay = V1[i1] - V0[i1];                      \
    EDGE_EDGE_TEST(V0, U0, U1);                \
    EDGE_EDGE_TEST(V0, U1, U2);                \
    EDGE_EDGE_TEST(V0, U2, U0);                \
}


#define POINT_IN_TRI(V0, U0, U1, U2)   \
{                                      \
    float a, b, c, d0, d1, d2;         \
    a = U1[i1] - U0[i1];               \
    b = -(U1[i0] - U0[i0]);            \
    c = -a * U0[i0] - b * U0[i1];      \
    d0 = a * V0[i0] + b * V0[i1] + c;  \
    a = U2[i1] - U1[i1];               \
    b = -(U2[i0] - U1[i0]);            \
    c = -a * U1[i0] - b * U1[i1];      \
    d1 = a * V0[i0] + b * V0[i1] + c;  \
    a = U0[i1] - U2[i1];               \
    b = -(U0[i0] - U2[i0]);            \
    c = -a * U2[i0] - b * U2[i1];      \
    d2 = a * V0[i0] + b * V0[i1] + c;  \
    if (d0 * d1 > 0.0)                 \
    {                                  \
        if (d0 * d2 > 0.0) return 1;   \
    }                                  \
}


__device__ int coplanerTriTri(const float N[3], const float V0[3], const float V1[3], const float V2[3],
                              const float U0[3], const float U1[3], const float U2[3])
{
    float A[3];
    short i0, i1;
    A[0] = FABS(N[0]);
    A[1] = FABS(N[1]);
    A[2] = FABS(N[2]);
    if (A[0] > A[1]) {
        if (A[0] > A[2]) {
            i0 = 1;
            i1 = 2;
        } else {
            i0 = 0;
            i1 = 1;
        }
    } else {
        if (A[2] > A[1]) {
            i0 = 0;
            i1 = 1;
        } else {
            i0 = 0;
            i1 = 2;
        }
    }

    EDGE_AGAINST_TRI_EDGES(V0, V1, U0, U1, U2);
    EDGE_AGAINST_TRI_EDGES(V1, V2, U0, U1, U2);
    EDGE_AGAINST_TRI_EDGES(V2, V0, U0, U1, U2);

    POINT_IN_TRI(V0, U0, U1, U2);
    POINT_IN_TRI(U0, V0, V1, V2);

    return 0;
}


__device__ int triTriIntersect(const float V0[3], const float V1[3], const float V2[3],
                               const float U0[3], const float U1[3], const float U2[3])
{
    float E1[3], E2[3];
    float N1[3], N2[3], d1, d2;
    float du0, du1, du2, dv0, dv1, dv2;
    float D[3];
    float isect1[2], isect2[2];
    float du0du1, du0du2, dv0dv1, dv0dv2;
    short index;
    float vp0, vp1, vp2;
    float up0, up1, up2;
    float b, c, max;

    VEC3_SUB(E1, V1, V0);
    VEC3_SUB(E2, V2, V0);
    VEC3_CROSS(N1, E1, E2);
    d1 = -VEC3_DOT(N1, V0);

    du0 = VEC3_DOT(N1, U0) + d1;
    du1 = VEC3_DOT(N1, U1) + d1;
    du2 = VEC3_DOT(N1, U2) + d1;

    if (FABS(du0) < EPSILON) du0 = 0.0;
    if (FABS(du1) < EPSILON) du1 = 0.0;
    if (FABS(du2) < EPSILON) du2 = 0.0;

    du0du1 = du0 * du1;
    du0du2 = du0 * du2;

    if (du0du1 > 0.0f && du0du2 > 0.0f)
        return 0;

    VEC3_SUB(E1, U1, U0);
    VEC3_SUB(E2, U2, U0);
    VEC3_CROSS(N2, E1, E2);
    d2 = -VEC3_DOT(N2, U0);

    dv0 = VEC3_DOT(N2, V0) + d2;
    dv1 = VEC3_DOT(N2, V1) + d2;
    dv2 = VEC3_DOT(N2, V2) + d2;

    if (FABS(dv0) < EPSILON) dv0 = 0.0;
    if (FABS(dv1) < EPSILON) dv1 = 0.0;
    if (FABS(dv2) < EPSILON) dv2 = 0.0;

    dv0dv1 = dv0 * dv1;
    dv0dv2 = dv0 * dv2;

    if (dv0dv1 > 0.0f && dv0dv2 > 0.0f)
        return 0;

    VEC3_CROSS(D, N1, N2);

    max = FABS(D[0]);
    index = 0;
    b = FABS(D[1]);
    c = FABS(D[2]);
    if (b > max) max = b, index = 1;
    if (c > max) max = c, index = 2;

    vp0 = V0[index];
    vp1 = V1[index];
    vp2 = V2[index];

    up0 = U0[index];
    up1 = U1[index];
    up2 = U2[index];

    COMPUTE_INTERVALS(vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2, isect1[0], isect1[1]);
    COMPUTE_INTERVALS(up0, up1, up2, du0, du1, du2, du0du1, du0du2, isect2[0], isect2[1]);

    SORT(isect1[0], isect1[1]);
    SORT(isect2[0], isect2[1]);

    if (isect1[1] < isect2[0] || isect2[1] < isect1[0]) return 0;
    return 1;
}


__device__ float atomicMaxFloat(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
        }
    return __int_as_float(old);
}


__global__ void ProcessFaceKernel(
    const float* __restrict__ vertices,
    const size_t num_faces,
    const int32_t* __restrict__ faces,
    float* __restrict__ faces_center,
    float* __restrict__ max_distance)
{
    int32_t face_vertices[3];
    float center_position[3];
    float vertex_position[3];
    float thread_max_square_distance = 0;

    __shared__ float shared_distances[256];

    const int32_t chunks = (1 + (num_faces - 1) / blockDim.x);
    for (int32_t chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
        const int32_t current_face_index = blockDim.x * chunk + threadIdx.x;
        if (current_face_index >= num_faces) continue;

        size_t offset = current_face_index * 3;
        VEC3_GET(face_vertices, faces, offset)

        center_position[0] = 0;
        center_position[1] = 0;
        center_position[2] = 0;
        for (unsigned int v = 0; v < 3; ++v) {
            size_t index = face_vertices[v] * 3;
            center_position[0] += vertices[index];
            center_position[1] += vertices[index + 1];
            center_position[2] += vertices[index + 2];
        }
        VEC3_DIV(center_position, center_position, 3)

        float distance;
        for (size_t v = 0; v < 3; ++v) {
            VEC3_GET(vertex_position, vertices, face_vertices[v] * 3)
            VEC3_SQUARE_DISTANCE(distance, center_position, vertex_position)
            if (distance > thread_max_square_distance) thread_max_square_distance = distance;
        }

        VEC3_PUT(faces_center, offset, center_position);
    }

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;

    shared_distances[tid] = sqrt(thread_max_square_distance) * 2;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)  {
        if ((tid < s) && (gid < num_faces)) {
                shared_distances[tid] = max(shared_distances[tid], shared_distances[tid + s]);
            }
        __syncthreads();
    }

    if (tid == 0) atomicMaxFloat(max_distance, shared_distances[0]);
}


__global__ void MarkSelfIntersectionKernel(
    const float* __restrict__ vertices,
    const size_t num_faces,
    const int32_t* __restrict__ faces,
    const float* __restrict__ faces_center,
    const float* __restrict__ max_distance,
    int* __restrict__ faces_intersecting,
    int* __restrict__ num_intersecting)
{
    int32_t face_vertices[3];
    float   face_position[3];
    int32_t neighbor_face_vertices[3];
    float   neighbor_face_position[3];

    const float threshold = *max_distance;

    const int32_t chunks = (1 + (num_faces - 1) / blockDim.x);
    for (int32_t chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {

        const int32_t current_face_index = blockDim.x * chunk + threadIdx.x;
        if (current_face_index >= num_faces) continue;

        int32_t starting_face = current_face_index + 1;
        VEC3_GET(face_vertices, faces, current_face_index * 3)
        VEC3_GET(face_position, faces_center, current_face_index * 3)

        for (int32_t i = starting_face; i < num_faces; ++i) {

            VEC3_GET(neighbor_face_position, faces_center, i * 3)
            if (abs(neighbor_face_position[0] - face_position[0]) > threshold) continue;
            if (abs(neighbor_face_position[1] - face_position[1]) > threshold) continue;
            if (abs(neighbor_face_position[2] - face_position[2]) > threshold) continue;

            VEC3_GET(neighbor_face_vertices, faces, i * 3)
            if (VEC3_CONTAINS(neighbor_face_vertices, face_vertices[0]) ||
                VEC3_CONTAINS(neighbor_face_vertices, face_vertices[1]) ||
                VEC3_CONTAINS(neighbor_face_vertices, face_vertices[2])) continue;

            int intersect = triTriIntersect(
                &vertices[face_vertices[0] * 3],
                &vertices[face_vertices[1] * 3],
                &vertices[face_vertices[2] * 3],
                &vertices[neighbor_face_vertices[0] * 3],
                &vertices[neighbor_face_vertices[1] * 3],
                &vertices[neighbor_face_vertices[2] * 3]);

            if (intersect == 1) {
                faces_intersecting[i] = 1;
                faces_intersecting[current_face_index] = 1;
                atomicAdd(&num_intersecting[0], 1);
                // current_face_index is the face currently being processed
                // accumulate intersecting pairs instead of just an indicator (current_face_index, i)
            }
        }
    }
}



std::tuple<at::Tensor, at::Tensor> MarkSelfIntersectingFaces(
    const at::Tensor& input_vertices,
    const at::Tensor& input_faces)
{
    at::TensorArg
        v{input_vertices, "vertices",  1},
        f{input_faces,    "faces",     2};

    at::CheckedFrom c = "MarkSelfIntersectingFaces";
    at::checkAllSameGPU(c, {v, f});

    at::cuda::CUDAGuard device_guard(input_vertices.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const auto num_faces = input_faces.size(0);
    const auto num_vertices = input_vertices.size(0);

    TORCH_CHECK(input_vertices.size(1) == 3, "Vertices must be 3D");
    TORCH_CHECK(input_faces.size(1) == 3, "Faces must be triangular");

    auto dtype_int   = input_vertices.options().dtype(at::kInt);
    auto dtype_float = input_vertices.options().dtype(at::kFloat);

    auto vertices = input_vertices.contiguous();
    auto faces = input_faces.contiguous();

    // pre-compute centers of each triangle for the nearest-neighbor
    // search and at the same time compute a reasonable max distance
    // threshold for ignoring faces that are too far
    auto faces_center = at::zeros({num_faces, 3}, dtype_float);
    auto max_distance = at::zeros({1}, dtype_float);
    ProcessFaceKernel<<<256, 256>>>(
        vertices.data_ptr<float>(),
        num_faces,
        faces.data_ptr<int32_t>(),
        faces_center.data_ptr<float>(),
        max_distance.data_ptr<float>());

    // mark intersecting faces by finding a set of neighoring faces for
    // each face (within the specified max distance) and check for intersections
    auto faces_intersecting = at::zeros({num_faces}, dtype_int);
    auto num_intersecting = at::zeros({1}, dtype_int);
    MarkSelfIntersectionKernel<<<256, 256, 0, stream>>>(
        vertices.data_ptr<float>(),
        num_faces,
        faces.data_ptr<int32_t>(),
        faces_center.data_ptr<float>(),
        max_distance.data_ptr<float>(),
        faces_intersecting.data_ptr<int>(),
        num_intersecting.data_ptr<int>());

    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(faces_intersecting, num_intersecting);
}

// stuff for mesh-mesh intersection calculations

// __global__ void MarkIntersectionKernel(
//     const float* __restrict__ vertices_0,
//     const float* __restrict__ vertices_1,
//     const size_t num_faces_0,
//     const size_t num_faces_1,
//     const int32_t* __restrict__ faces_0,
//     const int32_t* __restrict__ faces_1,
//     const float* __restrict__ faces_center_0,
//     const float* __restrict__ faces_center_1,
//     const float* __restrict__ max_distance_0,
//     const float* __restrict__ max_distance_1,
//     int* __restrict__ faces_intersecting,
//     int* __restrict__ num_intersecting,
// )
// {
//     int32_t face_vertices[3];
//     float   face_position[3];
//     int32_t neighbor_face_vertices[3];
//     float   neighbor_face_position[3];

//     const float threshold = *max(max_distance_0, max_distance_1);

//     const int32_t chunks = (1 + (num_faces - 1) / blockDim.x);
//     for (int32_t chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {

//         const int32_t current_face_index = blockDim.x * chunk + threadIdx.x;
//         if (current_face_index >= num_faces) continue;

//         int32_t starting_face = current_face_index + 1;
//         VEC3_GET(face_vertices, faces, current_face_index * 3)
//         VEC3_GET(face_position, faces_center, current_face_index * 3)

//         for (int32_t i = starting_face; i < num_faces; ++i) {

//             VEC3_GET(neighbor_face_position, faces_center, i * 3)
//             if (abs(neighbor_face_position[0] - face_position[0]) > threshold) continue;
//             if (abs(neighbor_face_position[1] - face_position[1]) > threshold) continue;
//             if (abs(neighbor_face_position[2] - face_position[2]) > threshold) continue;

//             VEC3_GET(neighbor_face_vertices, faces, i * 3)
//             if (VEC3_CONTAINS(neighbor_face_vertices, face_vertices[0]) ||
//                 VEC3_CONTAINS(neighbor_face_vertices, face_vertices[1]) ||
//                 VEC3_CONTAINS(neighbor_face_vertices, face_vertices[2])) continue;

//             int intersect = triTriIntersect(
//                 &vertices[face_vertices[0] * 3],
//                 &vertices[face_vertices[1] * 3],
//                 &vertices[face_vertices[2] * 3],
//                 &vertices[neighbor_face_vertices[0] * 3],
//                 &vertices[neighbor_face_vertices[1] * 3],
//                 &vertices[neighbor_face_vertices[2] * 3]);

//             if (intersect == 1) {
//                 faces_intersecting[i] = 1;
//                 faces_intersecting[current_face_index] = 1;
//                 atomicAdd(&num_intersecting[0], 1);
//                 // current_face_index is the face currently being processed
//                 // accumulate intersecting pairs instead of just an indicator (current_face_index, i)
//             }
//         }
//     }
// }


// std::tuple<at::Tensor, at::Tensor> MarkIntersectingFaces(
//     const at::Tensor& vertices_0,
//     const at::Tensor& faces_0,
//     const at::Tensor& vertices_1,
//     const at::Tensor& faces_1,
// )
// {
//     at::TensorArg
//         v0{vertices_0, "vertices_0",  1},
//         f0{faces_0,    "faces_0",     2};
//         v1{vertices_1, "vertices_1",  3},
//         f1{faces_1,    "faces_1",     4};

//     at::CheckedFrom c = "MarkSelfIntersectingFaces";
//     at::checkAllSameGPU(c, {v0, f0, v1, f1});

//     at::cuda::CUDAGuard device_guard(vertices_0.device());
//     at::cuda::CUDAGuard device_guard(vertices_1.device());
//     cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//     const auto num_vertices_0 = vertices_0.size(0);
//     const auto num_vertices_1 = vertices_1.size(0);
//     const auto num_faces_0 = faces_0.size(0);
//     const auto num_faces_1 = faces_1.size(0);

//     TORCH_CHECK(vertices_0.size(1) == 3, "Vertices must be 3D");
//     TORCH_CHECK(vertices_1.size(1) == 3, "Vertices must be 3D");
//     TORCH_CHECK(faces_0.size(1) == 3, "Faces must be triangular");
//     TORCH_CHECK(faces_1.size(1) == 3, "Faces must be triangular");

//     auto dtype_int   = vertices_0.options().dtype(at::kInt);
//     auto dtype_float = vertices_0.options().dtype(at::kFloat);

//     auto cvertices_0 = vertices_0.contiguous();
//     auto cvertices_1 = vertices_1.contiguous();
//     auto cfaces_0 = faces_0.contiguous();
//     auto cfaces_1 = faces_1.contiguous();



//     // pre-compute centers of each triangle for the nearest-neighbor
//     // search and at the same time compute a reasonable max distance
//     // threshold for ignoring faces that are too far
//     auto faces_center_0 = at::zeros({num_faces_0, 3}, dtype_float);
//     auto faces_center_1 = at::zeros({num_faces_1, 3}, dtype_float);
//     auto max_distance_0 = at::zeros({1}, dtype_float);
//     auto max_distance_1 = at::zeros({1}, dtype_float);

//     ProcessFaceKernel<<<256, 256>>>(
//         cvertices_0.data_ptr<float>(),
//         num_faces_0,
//         cfaces_0.data_ptr<int32_t>(),
//         faces_center_0.data_ptr<float>(),
//         max_distance_0.data_ptr<float>()
//     );
//     ProcessFaceKernel<<<256, 256>>>(
//         cvertices_0.data_ptr<float>(),
//         num_faces_0,
//         cfaces_0.data_ptr<int32_t>(),
//         faces_center_1.data_ptr<float>(),
//         max_distance_1.data_ptr<float>()
//     );

//     // mark intersecting faces by finding a set of neighoring faces for
//     // each face (within the specified max distance) and check for intersections
//     auto faces_intersecting = at::zeros({num_faces}, dtype_int);
//     auto num_intersecting = at::zeros({1}, dtype_int);
//     MarkIntersectionKernel<<<256, 256, 0, stream>>>(
//         cvertices_0.data_ptr<float>(),
//         cvertices_1.data_ptr<float>(),
//         num_faces_0,
//         num_faces_1,
//         cfaces_0.data_ptr<int32_t>(),
//         cfaces_1.data_ptr<int32_t>(),
//         faces_center_0.data_ptr<float>(),
//         faces_center_1.data_ptr<float>(),
//         max_distance_0.data_ptr<float>(),
//         max_distance_1.data_ptr<float>(),
//         faces_intersecting.data_ptr<int>(),
//         num_intersecting.data_ptr<int>()
//     );

//     AT_CUDA_CHECK(cudaGetLastError());
//     return std::make_tuple(faces_intersecting, num_intersecting);
// }
