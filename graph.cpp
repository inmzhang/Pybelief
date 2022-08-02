#include "graph.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>

void Node::register_neighbor(Node &neighbor)
{
    neighbors[neighbor.id] = neighbor;
}

std::vector<int> Node::get_neighbors()
{
    std::vector<int> ns;
    for (auto &neighbor : neighbors)
    {
        ns.push_back(neighbor.first);
    }
    return ns;
}

void Node::receive_messages()
{
    for (auto &neighbor : neighbors)
    {
        received_messages[neighbor.first] = neighbor.second.message(id);
    }
}

void Node::initialize()
{
    for (auto &neighbor : neighbors)
    {
        received_messages[neighbor.first] = 0;
    }
}

std::string Node::repr() const
{
    std::string repr = "";
    repr += "Node " + std::to_string(id) + ": ";
    repr += "neighbors: ";
    for (auto &neighbor : neighbors)
    {
        repr += std::to_string(neighbor.first) + " ";
    }
    return repr;
}


double phi(double x)
{
    return -std::log((std::tanh(x) / 2));
}


double CNode::message(int requester_id)
{
    std::vector<double> messages;
    int sign = 1;
    for (auto &msg : received_messages)
    {
        if (msg.first != requester_id)
        {
            double m = msg.second;
            sign = std::copysign(1, m * sign);
            messages.push_back(phi(std::abs(m)));
        }
    }

    return sign * phi(std::accumulate(messages.begin(), messages.end(), 0.0));
}


VNode::VNode(int id, double prior_prob) : Node(id), prior_prob(prior_prob) 
{
    prior_llr = std::log((1-prior_prob)/prior_prob);
}

double VNode::message(int requester_id)
{
    std::vector<double> messages;
    for (auto &msg : received_messages)
    {
        if (msg.first != requester_id)
        {
            messages.push_back(msg.second);
        }
    }
    return prior_llr + std::accumulate(messages.begin(), messages.end(), 0.0);
}


double VNode::estimate()
{
    double sum = prior_llr;
    for (auto &msg : received_messages)
    {
        sum += msg.second;
    }
    return sum;
}


void TannerGraph::add_cnode()
{
    int id = ++num_cnodes;
    CNode node(id);
    c_nodes[id] = node;
}


void TannerGraph::add_vnode(double prior_prob)
{
    int id = ++num_vnodes;
    VNode node(id, prior_prob);
    v_nodes[id] = node;
}


void TannerGraph::add_edge(int vnode_id, int cnode_id)
{
    auto searchv = v_nodes.find(vnode_id);
    if (searchv == v_nodes.end())
    {
        throw std::invalid_argument("VNode not found");
    }
    auto searchc = c_nodes.find(cnode_id);
    if (searchc == c_nodes.end())
    {
        throw std::invalid_argument("CNode not found");
    }

    auto &vnode = searchv->second;
    auto &cnode = searchc->second;
    vnode.register_neighbor(cnode);
    cnode.register_neighbor(vnode);
}

void TannerGraph::from_parity_check_matrix(std::vector<std::vector<int>> &parity_check_matrix, std::vector<double> &prior_probs)
{      
    // add nodes
    int nc = parity_check_matrix.size();
    int nv = parity_check_matrix[0].size();
    for (int i = 0; i < nc; i++)
    {
        add_cnode();
    }

    for (int i = 0; i < nv; i++)
    {
        add_vnode(prior_probs[i]);
    }

    // add edges
    int row_num = 0;
    for (auto row = parity_check_matrix.begin(); row != parity_check_matrix.end(); ++row)
        {   
            int col_num = 0;
            for (auto col = row->begin(); col != row->end(); ++col)
            {   
                if (*col == 1)
                {
                    add_edge(col_num, row_num);
                }
                col_num++;
            }
            col_num = 0;
            row_num++;
        }
}