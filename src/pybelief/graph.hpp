#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Node 
{
public:
    int row;
    int col;
    Node *next_in_row;
    Node *prev_in_row;
    Node *next_in_col;
    Node *prev_in_col;
    double bit_to_check;
    double check_to_bit;
    int sgn;
    
    Node(int row, int col);
};


class Graph
{
public:
    int num_rows;
    int num_cols;
    std::vector<std::vector<Node*>> Nodes;
    std::vector<double> prior_probs;
    std::vector<Node*> _first_in_row;
    std::vector<Node*> _first_in_col;
    std::vector<Node*> _last_in_row;
    std::vector<Node*> _last_in_col;

    Node *first_in_row(int row);
    Node *first_in_col(int col);
    Node *last_in_row(int row);
    Node *last_in_col(int col);

    void from_parity_check_matrix(py::array_t<std::uint8_t> &parity_check_matrix, py::array_t<double> &prior_probs);
};
