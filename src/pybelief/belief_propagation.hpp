#pragma once
#include "graph.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct Result
{
    Result();
    Result(bool converged, std::vector<double> posterior_probs, std::vector<int> hard_decisions);
    bool converged;
    std::vector<double> posterior_probs;
    std::vector<int> hard_decisions;
};

class BeliefPropagation
{
public:
    TannerGraph graph;
    py::array_t<std::uint8_t> parity_check_matrix;
    py::array_t<double> prior_probs;
    int max_iter;
    bool parallel;

    BeliefPropagation();
    BeliefPropagation(py::array_t<std::uint8_t> &parity_check_matrix, py::array_t<double> &prior_probs, int &max_iter, bool &parallel);

    Result decode(const py::array_t<std::uint8_t> &syndromes_);

private:
    std::vector<int> syndromes;
    std::vector<double> soft_probs;
    std::vector<int> hard_decisions;
    std::vector<double> soft_decision();
    std::vector<int> hard_decision();
    bool converge();
};