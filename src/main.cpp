#include "belief_propagation.hpp"
#include <iostream>
// #include "graph.hpp"

int main()
{
	std::vector<std::vector<int>> H{{1, 1, 0}, {0, 1, 1}};
	std::vector<double> prior_probs{0.3, 0.3, 0.3};
	std::vector<int> syndromes{0, 1};
	int max_iter = 3;
	bool parallel = false;

	auto bp_decoder = BeliefPropagation(H, prior_probs, max_iter, parallel);
	auto res = bp_decoder.decode(syndromes);

	std::cout << "Converged: " << res.converged << std::endl;
	std::cout << "Soft Probs: " << std::endl;
	for (auto i : res.posterior_probs)
	{
		std::cout << i << " ";
	}
	std::cout << std::endl
			  << "Hard Decisions: " << std::endl;
	for (auto i : res.hard_decisions)
	{
		std::cout << i << " ";
	}
	std::cout << std::endl;
}