// #include "belief_propagation.hpp"
#include "graph.hpp"

int main()
{
    std::vector<std::vector<int>> H{{1, 1, 0}, {0, 1, 1}};
    std::vector<double> prior_probs{0.5, 0.5, 0.5};
    TannerGraph graph;

    // BeliefPropagation bp_decoder = BeliefPropagation(H, prior_probs, 5, true);
    // auto [converged, soft_probs, hard_decisions] = bp_decoder.decode({0, 0, 0});
}