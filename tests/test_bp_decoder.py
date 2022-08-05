from pybelief import bp_decoder
import numpy as np

def test_bp_decoder_parallel():
    H = np.array([[1, 1, 0], [0, 1, 1]])
    prior_probs = np.array([0.3, 0.3, 0.3])
    max_iter = 5
    parallel = False
    bp = bp_decoder(H, prior_probs, max_iter, parallel)

    syndromes = np.array([1, 1])
    converged, probs, corrections = bp.decode(syndromes)
    assert converged
    assert np.allclose(corrections, np.array([0, 1, 0]))

def test_bp_decoder_parallel():
    H = np.array([[1, 1, 0], [0, 1, 1]])
    prior_probs = np.array([0.3, 0.3, 0.3])
    max_iter = 5
    parallel = True
    bp = bp_decoder(H, prior_probs, max_iter, parallel)

    syndromes = np.array([1, 1])
    converged, probs, corrections = bp.decode(syndromes)
    assert converged
    assert np.allclose(corrections, np.array([0, 1, 0]))