from typing import Tuple
import numpy as np
from pybelief._cpp_bp import BeliefPropagation

class bp_decoder:
    """
    A class for constructing belief_propagation decoder
    """    
    def __init__(self, H: np.ndarray, prior_probs: np.ndarray, max_iter: int, method: int, ms_scaling_factor: float) -> None:
        """Constructor for bp_decoder

        Args:
            H (np.ndarray): Parity check matrix representing the tanner graph.
            prior_probs (np.ndarray): Prior probabilities representing each hyperedges' error rate.
            max_iter (int): Maximum number of iterations before enforcing bp to stop.
            method (int): BP method to run, including 'product-sum-parallel'(1), 'min-sum-parallel'(2), 'product-sum-serial'(3), 'min-sum-serial'(4).
            ms_scaling_factor (float): Min-sum scaling factor, used in 'min-sum' method.
        """        
        self.parity_check_matrix = H
        self.prior_probs = prior_probs
        self.max_iter = max_iter
        self.method = method
        self.bp = BeliefPropagation(H, prior_probs, max_iter, method, ms_scaling_factor)

    def decode(self, syndromes: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Decoding with given syndromes

        Args:
            syndromes (np.ndarray): Observed syndromes

        Returns:
            Tuple[bool, np.ndarray, np.ndarray]: converged, posterior probs, decodings
        """        
        res = self.bp.decode(syndromes)
        return res.converged, np.array(res.posterior_probs), np.array(res.hard_decisions)