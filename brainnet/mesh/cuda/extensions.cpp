#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

at::Tensor NearestNeighbor(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&
);

std::tuple<at::Tensor, at::Tensor> MarkSelfIntersectingFaces(
    const at::Tensor&,
    const at::Tensor&
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "compute_nearest_neighbor",
        &NearestNeighbor,
        py::arg("point_set_0"),
        py::arg("point_set_1"),
        py::arg("point_set_0_size"),
        py::arg("point_set_1_size")
    );
    m.def(
        "compute_self_intersections",
        &MarkSelfIntersectingFaces,
        py::arg("vertices"),
        py::arg("faces")
    );
}
