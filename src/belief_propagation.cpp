#include "belief_propagation.hpp"
#include <iostream>

BeliefPropagation::BeliefPropagation(
    std::vector<std::vector<int>> &parity_check_matrix,
    std::vector<double> &prior_probs,
    int &max_iter,
    bool &parallel) : parity_check_matrix{parity_check_matrix},
                      prior_probs{prior_probs},
                      max_iter{max_iter},
                      parallel{parallel}
{
    graph.from_parity_check_matrix(parity_check_matrix, prior_probs);
}

std::vector<double> BeliefPropagation::soft_decision()
{
    std::vector<double> decisions;
    for (auto &vnode : graph.v_nodes)
    {
        decisions.push_back(vnode.second.estimate());
    }
    return decisions;
}

int hard(double prob)
{
    return prob > 0 ? 0 : 1;
}

std::vector<int> BeliefPropagation::hard_decision()
{
    std::vector<int> decisions;
    for (auto &vnode : graph.v_nodes)
    {
        decisions.push_back(hard(vnode.second.estimate()));
    }
    return decisions;
}

bool BeliefPropagation::converge(std::vector<int> &syndromes)
{
    for (auto &cnode : graph.c_nodes)
    {
        int check = syndromes[cnode.first];
        for (auto &vnode : cnode.second.neighbors)
        {
            check ^= hard_decisions[vnode.first];
        }
        if (check)
        {
            return false;
        }
    }
    return true;
}

Result BeliefPropagation::decode(std::vector<int> &syndromes)
{
    // Initialize
    for (auto &vnode : graph.v_nodes)
    {
        vnode.second.initialize();
    }
    for (auto &cnode : graph.c_nodes)
    {   
        cnode.second.set_syndrome(syndromes[cnode.first]);
        cnode.second.initialize();
    }

    bool converged = false;

    
    for (int i = 0; i < max_iter; i++)
    {
        // Parallel schedule
        if (parallel)
        {
            for (auto &cnode : graph.c_nodes)
            {   
                cnode.second.receive_messages();
            }

            for (auto &vnode : graph.v_nodes)
            {
                vnode.second.receive_messages();
            }
        }
        else
        // Serial schedule
        {
            for (auto &vnode : graph.v_nodes)
            {
                for (auto &neighbor : vnode.second.neighbors)
                {
                    neighbor.second -> receive_messages();
                }

                vnode.second.receive_messages();
            }
        }

        // Hard decision
        hard_decisions = hard_decision();
        // Converge test
        converged = converge(syndromes);
        if (converged)
        {
            break;
        }
    }
    soft_probs = soft_decision();
    Result res = {converged, soft_probs, hard_decisions}; 
    return res;
}