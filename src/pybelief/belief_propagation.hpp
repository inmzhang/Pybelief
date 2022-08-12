#pragma once
#include "graph.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct BPResult
{
    BPResult(bool converged, std::vector<double> &posterior_probs, std::vector<int> &hard_decisions);
    /**
     * @brief Whether the bp algorithm converged.
     * 
     */
    bool converged;
    /**
     * @brief The marginal probabilities after bp halted, used when converged=false.
     * 
     */
    std::vector<double> posterior_probs;
    /**
     * @brief The decoding results, used when converged=true.
     * 
     */
    std::vector<int> hard_decisions;
};


/**
 * @brief Belief Propagation Algorithm
 * 
 */
class BeliefPropagation
{
public:
    Graph graph;
    py::array_t<std::uint8_t> parity_check_matrix;
    py::array_t<double> prior_probs;
    int max_iter;
    int method;
    double scale;

    /**
     * @brief Construct a new Belief Propagation object
     * 
     * @param parity_check_matrix Parity check matrix representing the tanner graph.
     * @param prior_probs Prior probabilities representing each hyperedges' error rate.
     * @param max_iter Maximum number of iterations before enforcing bp to stop.
     * @param method BP method to run, including 'product-sum-parallel'(1), 'min-sum-parallel'(2), 'product-sum-serial'(3), 'min-sum-serial'(4).
     * @param scale Min-sum scaling factor, used in 'min-sum' method.
     */
    BeliefPropagation(py::array_t<std::uint8_t> &parity_check_matrix, py::array_t<double> &prior_probs, int max_iter, int method, double scale);

    /**
     * @brief Decoding with given syndromes
     * 
     * @param syndromes_ Observed syndromes
     * @return Results of bp
     */
    BPResult decode(const py::array_t<std::uint8_t> &syndromes_);

    std::vector<int> syndromes;
    std::vector<double> soft_probs;
    std::vector<int> hard_decisions;
};