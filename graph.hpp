#pragma once
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <set>
#include <utility>

struct Node
{
public:
    int id;
    std::map<int, Node> neighbors;
    std::map<int, double> received_messages;

    Node(int id) : id(id) {}
    void register_neighbor(Node &neighbor);
    std::vector<int> get_neighbors();
    void receive_messages();
    void initialize();
    virtual double message(int requester_id);
    std::string repr() const;
};

struct CNode : Node
{
    CNode(int id) : Node(id) {}
    double message(int requester_id);
};

struct VNode : Node
{
    VNode(int id, double prior_prob);
    double prior_prob;
    int event;
    double prior_llr;
    double message(int requester_id);
    double estimate();
};

class TannerGraph
{
public:
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
