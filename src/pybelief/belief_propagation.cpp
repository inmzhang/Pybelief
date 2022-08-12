#include "belief_propagation.hpp"
#include <cmath>

BPResult::BPResult(bool converged, std::vector<double> &posterior_probs, std::vector<int> &hard_decisions) : converged(converged), posterior_probs(posterior_probs), hard_decisions(hard_decisions) {}

BeliefPropagation::BeliefPropagation(
    py::array_t<std::uint8_t> &parity_check_matrix,
    py::array_t<double> &prior_probs,
    int max_iter,
    int method,
    double scale) : parity_check_matrix{parity_check_matrix},
                    prior_probs{prior_probs},
                    max_iter{max_iter},
                    method{method},
                    scale{scale}
{
    graph.from_parity_check_matrix(parity_check_matrix, prior_probs);
    syndromes.resize(graph.num_rows);
    soft_probs.resize(graph.num_cols);
    hard_decisions.resize(graph.num_cols);
}

BPResult BeliefPropagation::decode(const py::array_t<std::uint8_t> &syndromes_)
{
    // Set syndromes
    auto s = syndromes_.unchecked<1>();
    for (int i = 0; i < s.shape(0); i++)
    {
        syndromes[i] = s(i);
    }

    // Initialize
    for (int j = 0; j < graph.num_cols; j++)
    {
        Node *node = graph.first_in_col(j);
        while (node)
        {
            double p = graph.prior_probs[j];
            node->bit_to_check = std::log((1 - p) / (p));
            node = node->next_in_col;
        }
    }

    bool converged = true;
    double alpha;
    // Message passing
    for (int iter = 1; iter <= max_iter; iter++)
    {
        // ms scaling factor, used in min-sum
        if (scale == 0)
        {
            alpha = 1.0 - std::pow(2, -1 * iter);
        }
        else
        {
            alpha = scale;
        }

        // parallel schedule
        if ((method == 1) || (method == 2))
        {
            /* check to bit messages */
            // product-sum rule
            if (method == 1)
            {
                for (int i = 0; i < graph.num_rows; i++)
                {
                    Node *node = graph.first_in_row(i);
                    double temp = 1.0;
                    while (node)
                    {
                        node->check_to_bit = temp;
                        temp *= std::tanh(node->bit_to_check / 2);
                        node = node->next_in_row;
                    }

                    node = graph.last_in_row(i);
                    temp = 1.0;
                    while (node)
                    {
                        node->check_to_bit *= temp;
                        node->check_to_bit = std::pow(-1, syndromes[i]) * std::log((1 + node->check_to_bit) / (1 - node->check_to_bit));
                        temp *= std::tanh(node->bit_to_check / 2);
                        node = node->prev_in_row;
                    }
                }
            }
            // min-sum rule
            else
            {
                for (int i = 0; i < graph.num_rows; i++)
                {
                    Node *node = graph.first_in_row(i);
                    double temp = 1e308;
                    int sign = 0;
                    if (syndromes[i] == 1)
                    {
                        sign = 1;
                    }
                    while (node)
                    {
                        node->check_to_bit = temp;
                        node -> sgn = sign;
                        if (std::abs(node->bit_to_check) < temp)
                        {
                            temp = std::abs(node->bit_to_check);
                        }
                        if (node->bit_to_check <= 0)
                        {
                            sign += 1;
                        }
                        node = node->next_in_row;
                    }
                    node = graph.last_in_row(i);
                    temp = 1e308;
                    sign = 0;
                    while (node)
                    {
                        if (temp < std::abs(node->check_to_bit))
                        {
                            node->check_to_bit = temp;
                        }
                        node->sgn += sign;
                        node->check_to_bit *= (std::pow(-1, node->sgn) * alpha);
                        if (std::abs(node->bit_to_check) < temp)
                        {
                            temp = std::abs(node->bit_to_check);
                        }
                        if (node->bit_to_check <= 0)
                        {
                            sign += 1;
                        }
                        node = node->prev_in_row;
                    }
                }
            }
            /* bit to check messages */
            for (int j = 0; j < graph.num_cols; j++)
            {
                Node *node = graph.first_in_col(j);
                double p = graph.prior_probs[j];
                double temp = std::log((1 - p) / (p));

                while (node)
                {
                    node->bit_to_check = temp;
                    temp += node->check_to_bit;
                    node = node->next_in_col;
                }

                soft_probs[j] = temp;
                if (temp <= 0)
                {
                    hard_decisions[j] = 1;
                }
                else
                {
                    hard_decisions[j] = 0;
                }

                node = graph.last_in_col(j);
                temp = 0.0;
                while (node)
                {
                    node->bit_to_check += temp;
                    temp += node->check_to_bit;
                    node = node->prev_in_col;
                }
            }
        }
        // serial schedule
        else
        {
            for (int j = 0; j < graph.num_cols; j++)
            {
                /* check to bit messages */

                Node *node = graph.first_in_col(j);
                double temp;
                // product-sum rule
                if (method == 3)
                {
                    while (node)
                    {
                        Node *g = graph.first_in_row(node->row);
                        temp = 1.0;
                        while (g)
                        {
                            if (g->col != j)
                            {
                                temp *= std::tanh(g->bit_to_check / 2);
                            }
                            g = g->next_in_row;
                        }
                        node->check_to_bit = (std::pow(-1, syndromes[node->row]) * std::log((1 + temp) / (1 - temp)));
                        node = node->next_in_col;
                    }
                }
                // min-sum rule
                else
                {
                    while (node)
                    {
                        Node *g = graph.first_in_row(node->row);
                        temp = 1e308;
                        int sign = 0;
                        if (syndromes[node->row] == 1)
                        {
                            sign = 1;
                        }
                        while (g)
                        {
                            if (g->col != j)
                            {
                                if (std::abs(g->bit_to_check) < temp)
                                {
                                    temp = std::abs(g->bit_to_check);
                                }
                                if (g->bit_to_check <= 0)
                                {
                                    sign += 1;
                                }
                            }
                            g = g->next_in_row;
                        }
                        node->check_to_bit = (std::pow(-1, sign) * temp * alpha);
                        node = node->next_in_col;
                    }
                }
                /* bit to check messages */
                node = graph.first_in_col(j);
                double p = graph.prior_probs[j];
                temp = std::log((1 - p) / (p));

                while (node)
                {
                    node->bit_to_check = temp;
                    temp += node->check_to_bit;
                    node = node->next_in_col;
                }

                soft_probs[j] = temp;
                if (temp <= 0)
                {
                    hard_decisions[j] = 1;
                }
                else
                {
                    hard_decisions[j] = 0;
                }

                node = graph.last_in_col(j);
                temp = 0.0;
                while (node)
                {
                    node->bit_to_check += temp;
                    temp += node->check_to_bit;
                    node = node->prev_in_col;
                }
            }
        }

        // Check for convergence
        for (int i = 0; i < graph.num_rows; i++)
        {
            int sign = syndromes[i];
            Node *node = graph.first_in_row(i);
            while (node)
            {
                sign += hard_decisions[node->col];
                node = node -> next_in_row;
            }
            if ((sign % 2) == 1)
            {
                converged = false;
                break;
            }
        }
        if (converged)
        {
            break;
        }
    }
    BPResult res(converged, soft_probs, hard_decisions);
    return res;
}