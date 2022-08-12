from pybelief import bp_decoder
import numpy as np

def test_bp_decoder():
    H = np.array([[1, 1, 0], [0, 1, 1]])
    prior_probs = np.array([0.3, 0.3, 0.3])
    max_iter = 5
    method = 1
    bp = bp_decoder(H, prior_probs, max_iter, method, 0)

    syndromes = np.array([1, 1])
    converged, probs, corrections = bp.decode(syndromes)
    assert converged
    assert np.allclose(corrections, np.array([0, 1, 0]))
