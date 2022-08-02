#pragma once
#include "graph.hpp"
#include <tuple>

class BeliefPropagation
{
public:
    TannerGraph graph;
    std::vector<std::vector<int>> parity_check_matrix;
    std::vector<double> prior_probs;
    int max_iter = 5;
    bool parallel;
    std::vector<double> soft_probs;
    std::vector<int> hard_decisions;

    BeliefPropagation(std::vector<std::vector<int>> &parity_check_matrix, std::vector<double> &prior_probs, int max_iter, bool parallel);

    std::vector<double> &soft_decision();
    std::vector<int> &hard_decision();
    bool converge(std::vector<int> &syndromes);
    std::tuple<bool, std::vector<double>, std::vector<int>> &decode(std::vector<int> &syndromes);
};