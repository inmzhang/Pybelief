#include "graph.hpp"

Node::Node(int row, int col) : row(row), col(col) {}

Node *Graph::first_in_row(int row) { return _first_in_row[row]; }

Node *Graph::first_in_col(int col) { return _first_in_col[col]; }

Node *Graph::last_in_row(int row) { return _last_in_row[row]; }

Node *Graph::last_in_col(int col) { return _last_in_col[col]; }

void Graph::from_parity_check_matrix(py::array_t<std::uint8_t> &parity_check_matrix, py::array_t<double> &prior_probs_)
{
    auto mat = parity_check_matrix.unchecked<2>();
    auto prior = prior_probs_.unchecked<1>();
    // add nodes
    int nc = mat.shape(0);
    int nv = mat.shape(1);
    num_rows = nc;
    num_cols = nv;
    _first_in_row.resize(num_rows);
    _first_in_col.resize(num_cols);
    _last_in_row.resize(num_rows);
    _last_in_col.resize(num_cols);
    Nodes.resize(nc);
    Node *prev;
    for (int j = 0; j < nv; j++)
    {
        int first_in_col_ = true;
        for (int i = 0; i < nc; i++)
        {
            if (mat(i, j) == 1)
            {
                auto current = new Node(i, j);
                Nodes[i].push_back(current);
                if (first_in_col_)
                {
                    first_in_col_ = false;
                    current->prev_in_col = nullptr;
                    _first_in_col[j] = current;
                }
                else
                {
                    prev->next_in_col = current;
                    current->prev_in_col = prev;
                }
                prev = current;
            }
        }
        prev->next_in_col = nullptr;
        _last_in_col[j] = prev;
    }

    for (int r = 0; r < nc; r++)
    {
        for (auto iter = Nodes[r].begin(); iter != Nodes[r].end(); iter++)
        {
            Node *current = *iter;
            if (iter == Nodes[r].begin())
            {
                current->prev_in_row = nullptr;
                _first_in_row[r] = current;
            }
            else
            {
                prev->next_in_row = current;
                current->prev_in_row = prev;
            }
            prev = current;
        }
        prev->next_in_row = nullptr;
        _last_in_row[r] = prev;
    }

    // set prior probs
    int l = prior.shape(0);
    for (int i = 0; i < l; i++)
    {
        prior_probs.push_back(prior(i));
    }
}