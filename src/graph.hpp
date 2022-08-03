#pragma once
#include <map>
#include <string>
#include <vector>
#include <set>
#include <utility>
#include <cmath>

class CNode;
class VNode;

class CNode
{
public:
    int id;
    int syndrome = 0;
    std::map<int, VNode *> neighbors;
    std::map<int, double> received_messages;

    CNode() {}
    CNode(int id);
    void set_syndrome(int syndrome);
    void register_neighbor(VNode* neighbor);
    std::vector<int> get_neighbors();
    void receive_messages();
    double message(int requester_id);
    void initialize();
};

class VNode
{
public:
    int id;
    std::map<int, CNode *> neighbors;
    std::map<int, double> received_messages;

    VNode() {}
    VNode(int id, double prior_prob);
    double prior_prob;
    double prior_llr;
    double estimate();
    void register_neighbor(CNode* neighbor);
    std::vector<int> get_neighbors();
    void receive_messages();
    double message(int requester_id);
    void initialize();
};

class TannerGraph
{
public:
    TannerGraph() {}
    std::map<int, CNode> c_nodes;
    std::map<int, VNode> v_nodes;
    std::set<std::pair<int, int>> edges;
    int num_vnodes = 0;
    int num_cnodes = 0;

    void add_vnode(double prior_prob);
    void add_cnode();
    void add_edge(int vnode_id, int cnode_id);
    void from_parity_check_matrix(std::vector<std::vector<int>> &parity_check_matrix, std::vector<double> &prior_probs);
};
