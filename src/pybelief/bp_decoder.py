from typing import Tuple
import numpy as np
from pybelief._cpp_bp import BeliefPropagation

class bp_decoder:
    def __init__(self, H: np.ndarray, prior_probs: np.ndarray, max_iter: int, method: int, ms_scaling_factor: float) -> None:
        self.parity_check_matrix = H
        self.prior_probs = prior_probs
        self.max_iter = max_iter
        self.method = method
        self.bp = BeliefPropagation(H, prior_probs, max_iter, method, ms_scaling_factor)

    def decode(self, syndromes: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        res = self.bp.decode(syndromes)
        return res.converged, np.array(res.posterior_probs), np.array(res.hard_decisions)