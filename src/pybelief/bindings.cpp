#include <pybind11/pybind11.h>
#include "belief_propagation.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_cpp_bp, m)
{
    m.doc() = "Belief Propagation Algorithm";

    py::class_<BPResult>(m, "Result", u8R"(Belief propagation result)")
        .def(py::init<bool, std::vector<double>&, std::vector<int>&>())
        .def_readwrite("converged", &BPResult::converged)
        .def_readwrite("posterior_probs", &BPResult::posterior_probs)
        .def_readwrite("hard_decisions", &BPResult::hard_decisions);

    py::class_<BeliefPropagation>(m, "BeliefPropagation", u8R"(Belief propagation algorithm)")
        .def(py::init<py::array_t<std::uint8_t> &, py::array_t<double> &, int, int, double>())
        .def("decode", &BeliefPropagation::decode, u8R"(
        Decode a syndrome vector.
        Args:
            syndromes: A numpy array of syndrome bits.
        Returns:
            A Result object.
        )");
}