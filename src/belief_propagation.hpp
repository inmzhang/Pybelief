#pragma once
#include "graph.hpp"

struct Result
{
    bool converged;
    std::vector<double> posterior_probs;
    std::vector<int> hard_decisions;

};


class BeliefPropagation
{
public:
    TannerGraph graph;
    std::vector<std::vector<int>> parity_check_matrix;
    std::vector<double> prior_probs;
    int max_iter;
    bool parallel;
    std::vector<double> soft_probs;
    std::vector<int> hard_decisions;

    BeliefPropagation(std::vector<std::vector<int>> &parity_check_matrix, std::vector<double> &prior_probs, int &max_iter, bool &parallel);

    std::vector<double> soft_decision();
    std::vector<int> hard_decision();
    bool converge(std::vector<int> &syndromes);
    Result decode(std::vector<int> &syndromes);
};