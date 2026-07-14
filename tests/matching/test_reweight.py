import numpy as np
import pymatching
import pytest

import stim


def _normalized_undirected_edges(edges: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted((int(u), int(v)))) for u, v in edges.tolist()}


def _bernoulli_xor(p1: float, p2: float) -> float:
    return p1 * (1 - p2) + p2 * (1 - p1)


# A tiny detector error model with a single correlated pair of edges (0, 1) and (2, 3).
# `error(p_indep) Di Dj` give the independent (marginal) contributions, while the composite
# `error(p_corr) D0 D1 ^ D2 D3` introduces the joint co-occurrence between the two edges.
# The probabilities below are chosen so that the pair is *positively* correlated and the
# hard-evidence conditional P(mu | nu) stays below 0.5 (so the reweight is not clamped):
#   P(0,1) = P(2,3) = 0.2, P((0,1),(2,3)) = 0.08
#   => P(mu | nu) = 0.08 / 0.2 = 0.4
_P_INDEP = 0.142857142857142857  # chosen so bernoulli_xor(_P_INDEP, _P_CORR) == 0.2
_P_CORR = 0.08
_CORR_DEM = stim.DetectorErrorModel(f"""
    error({_P_INDEP}) D0 D1
    error({_P_INDEP}) D2 D3
    error({_P_CORR}) D0 D1 ^ D2 D3
""")


def _expected_reweighted_edge_weight(alpha: float) -> float:
    occ = _bernoulli_xor(_P_INDEP, _P_CORR)  # P(mu) = P(nu) = 0.2
    corr = _P_CORR  # P(mu, nu)
    p_pos = corr / occ  # P(mu | nu)
    # Regularized reweight: implied_p = alpha * P(mu | nu). alpha == 0 gives implied_p == 0
    # (no reweight), so guard against the log of zero.
    implied_p = min(0.5, alpha * p_pos)
    if implied_p <= 0.0:
        return np.log((1 - occ) / occ)
    w_reweight = np.log((1 - implied_p) / implied_p)
    w_original = np.log((1 - occ) / occ)
    # The reweight is only applied if it lowers the edge weight.
    return min(w_reweight, w_original)


def test_decode_reweight():
    # Simple graph: 0 -- 1 -- 2
    # Edge (0, 1) weight 2
    # Edge (1, 2) weight 2
    m = pymatching.Matching()
    m.add_edge(0, 1, fault_ids={0}, weight=2)
    m.add_edge(1, 2, fault_ids={1}, weight=2)

    # decode([1, 0, 1]) -> detection events 0, 2
    # Should match 0 to 2 via 1. Weight 4.
    res, weight = m.decode(np.array([1, 0, 1]), return_weight=True)
    assert weight == 4.0

    # Reweight (0, 1) to 5. New weight 5+2 = 7.
    reweights = np.array([[0, 1, 5.0]])
    res, weight = m.decode(
        np.array([1, 0, 1]), return_weight=True, edge_reweights=reweights
    )
    assert weight == 7.0

    # Check weights restored
    res, weight = m.decode(np.array([1, 0, 1]), return_weight=True)
    assert weight == 4.0


def test_decode_reweight_boundary():
    m = pymatching.Matching()
    m.add_boundary_edge(0, fault_ids={0}, weight=2)
    m.add_edge(0, 1, fault_ids={1}, weight=3)

    # decode([1, 0]) -> event at 0. Matches to boundary (weight 2).
    res, weight = m.decode(np.array([1, 0]), return_weight=True)
    assert weight == 2.0

    # Reweight boundary edge to 5.
    reweights = np.array([[0, -1, 5.0]])
    res, weight = m.decode(
        np.array([1, 0]), return_weight=True, edge_reweights=reweights
    )
    assert weight == 5.0

    # Restored
    res, weight = m.decode(np.array([1, 0]), return_weight=True)
    assert weight == 2.0


def test_decode_batch_reweight():
    m = pymatching.Matching()
    m.add_edge(0, 1, fault_ids={0}, weight=2)
    m.add_edge(1, 2, fault_ids={1}, weight=2)

    shots = np.array([[1, 0, 1], [1, 0, 1]], dtype=np.uint8)

    # Shot 0: reweight (0, 1) to 5. Expected weight 7.
    # Shot 1: no reweight. Expected weight 4.

    reweights = [np.array([[0, 1, 5.0]]), None]

    preds, weights = m.decode_batch(
        shots, return_weights=True, edge_reweights=reweights
    )
    assert weights[0] == 7.0
    assert weights[1] == 4.0

    # Check restored
    preds, weights = m.decode_batch(shots, return_weights=True)
    assert weights[0] == 4.0
    assert weights[1] == 4.0


def test_decode_batch_reweight_all_same():
    m = pymatching.Matching()
    m.add_edge(0, 1, fault_ids={0}, weight=2)

    shots = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    # Reweight to 5
    rw = np.array([[0, 1, 5.0]])
    reweights = [rw, rw]

    preds, weights = m.decode_batch(
        shots, return_weights=True, edge_reweights=reweights
    )
    assert weights[0] == 5.0
    assert weights[1] == 5.0

    preds, weights = m.decode_batch(shots, return_weights=True)
    assert weights[0] == 2.0


def test_decode_reweight_large_observables():
    # If num_observables > 64, the search graph should be present even if enable_correlations=False.
    m = pymatching.Matching()
    # Add enough edges with unique fault_ids to exceed 64 observables
    for i in range(70):
        m.add_edge(i, i + 1, fault_ids={i}, weight=1)

    assert m.num_fault_ids >= 70

    # Decode a simple case: error on edge (0, 1)
    # Expected weight 1.
    syndrome = np.zeros(m.num_nodes, dtype=np.uint8)
    syndrome[0] = 1
    syndrome[1] = 1

    res, weight = m.decode(syndrome, return_weight=True)
    assert weight == 1.0

    # Reweight edge (0, 1) to 10.
    reweights = np.array([[0, 1, 10.0]])
    res, weight = m.decode(syndrome, return_weight=True, edge_reweights=reweights)
    assert weight == 10.0

    # Verify the weight is restored
    res, weight = m.decode(syndrome, return_weight=True)
    assert weight == 1.0


def test_reweight_sign_flip_raises_error():
    m = pymatching.Matching()
    m.add_edge(0, 1, weight=2)
    m.add_edge(1, 2, weight=-3)

    # Positive to negative (flip)
    with pytest.raises(ValueError, match="sign flip not allowed"):
        m.decode(np.array([1, 0, 1]), edge_reweights=np.array([[0, 1, -5.0]]))

    # Negative to positive (flip)
    with pytest.raises(ValueError, match="sign flip not allowed"):
        m.decode(np.array([1, 0, 1]), edge_reweights=np.array([[1, 2, 3.0]]))


def test_reweight_negative_to_negative():
    # Graph: 0 -- 1 -- 2
    # (0, 1) weight 5
    # (1, 2) weight -3.
    # Solution for detection events at 0, 2.
    # Standard matching: 0 matches to 2 via 1. Path: (0,1), (1,2).
    # Cost: 5 + (-3) = 2.
    # Note: Negative weight -3 means edge (1,2) is pre-flipped.
    # Events at 0, 2 means syndrome is 1 at 0, 1 at 2.
    # If (1,2) is pre-flipped, it causes events at 1, 2.
    # Observed syndrome: 0:1, 1:0, 2:1.
    # Adjusted syndrome (xor with negative weight syndrome):
    # 0:1, 1:1, 2:0.
    # Now we match 0 and 1. Path (0, 1) cost 5.
    # Total cost = 5 + (-3) = 2.

    m = pymatching.Matching()
    m.add_edge(0, 1, weight=5)
    m.add_edge(1, 2, weight=-3)

    # Check baseline
    res, weight = m.decode(np.array([1, 0, 1]), return_weight=True)
    assert weight == 2.0

    # Reweight (1, 2) to -10.
    # New cost calculation:
    # Path (0, 1) cost 5.
    # Total cost = 5 + (-10) = -5.
    reweights = np.array([[1, 2, -10.0]])
    res, weight = m.decode(
        np.array([1, 0, 1]), return_weight=True, edge_reweights=reweights
    )
    assert weight == -5.0

    # Verify restoration
    res, weight = m.decode(np.array([1, 0, 1]), return_weight=True)
    assert weight == 2.0


def test_decode_to_edges_array_reweight_changes_solution_edges():
    m = pymatching.Matching()
    m.add_edge(0, 1, weight=1.0)
    m.add_edge(1, 2, weight=1.0)
    m.add_edge(0, 2, weight=3.0)

    syndrome = np.array([0, 1, 0], dtype=np.uint8)
    syndrome = np.array([1, 0, 1], dtype=np.uint8)

    edges = m.decode_to_edges_array(syndrome)
    assert _normalized_undirected_edges(edges) == {(0, 1), (1, 2)}

    reweights = np.array([[0, 2, 0.5]])
    edges_reweighted = m.decode_to_edges_array(syndrome, edge_reweights=reweights)
    assert _normalized_undirected_edges(edges_reweighted) == {(0, 2)}


def test_decode_to_edges_array_reweight_restores_weights():
    m = pymatching.Matching()
    m.add_edge(0, 1, weight=1.0)
    m.add_edge(1, 2, weight=1.0)
    m.add_edge(0, 2, weight=3.0)

    syndrome = np.array([1, 0, 1], dtype=np.uint8)
    reweights = np.array([[0, 2, 0.5]])

    _ = m.decode_to_edges_array(syndrome, edge_reweights=reweights)
    edges_after = m.decode_to_edges_array(syndrome)
    assert _normalized_undirected_edges(edges_after) == {(0, 1), (1, 2)}


def test_decode_to_edges_array_reweight_sign_flip_raises_error():
    m = pymatching.Matching()
    m.add_edge(0, 1, weight=2)
    m.add_edge(1, 2, weight=1)

    syndrome = np.array([1, 0, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="sign flip not allowed"):
        m.decode_to_edges_array(syndrome, edge_reweights=np.array([[0, 1, -2.0]]))


# --- Regularized reweight (alpha) tests --------------------------------------------------------


def test_alpha_one_matches_baseline_correlated_decode():
    # Regression: enable_correlations with the default alpha (hard conditioning) must be identical
    # to explicitly passing alpha=1.0, for decode / decode_batch / decode_to_edges_array.
    m = pymatching.Matching.from_detector_error_model(_CORR_DEM, enable_correlations=True)
    syndrome = np.array([1, 1, 1, 1], dtype=np.uint8)

    corr_default, w_default = m.decode(
        syndrome, return_weight=True, enable_correlations=True
    )
    corr_alpha1, w_alpha1 = m.decode(
        syndrome, return_weight=True, enable_correlations=True, alpha=1.0
    )
    assert np.array_equal(corr_default, corr_alpha1)
    assert w_default == w_alpha1

    edges_default = m.decode_to_edges_array(syndrome, enable_correlations=True)
    edges_alpha1 = m.decode_to_edges_array(syndrome, enable_correlations=True, alpha=1.0)
    assert _normalized_undirected_edges(edges_default) == _normalized_undirected_edges(
        edges_alpha1
    )

    # Only syndromes with even parity on each correlated pair {0,1} and {2,3} admit a perfect
    # matching (the DEM has no boundary edges), so sample from the four valid patterns.
    valid_patterns = np.array(
        [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.uint8
    )
    np.random.seed(0)
    shots = valid_patterns[np.random.randint(0, 4, size=50)]
    preds_default, weights_default = m.decode_batch(
        shots, return_weights=True, enable_correlations=True
    )
    preds_alpha1, weights_alpha1 = m.decode_batch(
        shots, return_weights=True, enable_correlations=True, alpha=1.0
    )
    assert np.array_equal(preds_default, preds_alpha1)
    assert np.array_equal(weights_default, weights_alpha1)


def test_alpha_ignored_when_correlations_disabled():
    # When enable_correlations=False, alpha is accepted but has no effect.
    m = pymatching.Matching.from_detector_error_model(_CORR_DEM, enable_correlations=True)
    syndrome = np.array([1, 1, 1, 1], dtype=np.uint8)

    corr_a, w_a = m.decode(syndrome, return_weight=True, alpha=0.3)
    corr_b, w_b = m.decode(syndrome, return_weight=True, alpha=1.7)
    corr_c, w_c = m.decode(syndrome, return_weight=True)
    assert np.array_equal(corr_a, corr_b)
    assert np.array_equal(corr_a, corr_c)
    assert w_a == w_b == w_c


def test_alpha_reweighted_weight_follows_regularized_formula():
    # Correctness: on the tiny correlated DEM, the solution for syndrome [1,1,1,1] is exactly the
    # two correlated edges (0,1) and (2,3), each reweighted symmetrically. The reported solution
    # weight is therefore 2 * (reweighted single-edge weight), which must follow the regularized
    # reweight formula for alpha in {0.0, 0.4, 1.0}.
    m = pymatching.Matching.from_detector_error_model(_CORR_DEM, enable_correlations=True)
    syndrome = np.array([1, 1, 1, 1], dtype=np.uint8)

    for alpha in (0.0, 0.4, 1.0):
        _, weight = m.decode(
            syndrome, return_weight=True, enable_correlations=True, alpha=alpha
        )
        expected = 2 * _expected_reweighted_edge_weight(alpha)
        assert weight == pytest.approx(expected, abs=1e-3)


def test_alpha_monotonic_in_solution_weight():
    # The reported solution weight is monotonically non-increasing in alpha: a larger alpha trusts
    # the correlation more, discounting the correlated edges more aggressively. Under the
    # regularized rule implied_p = alpha * P(mu | nu), the reweight only lowers the edge weight once
    # implied_p exceeds the prior P(mu) (here at alpha = P(mu) / P(mu | nu) = 0.5), so alpha values
    # in the active-boost regime are used to exercise the strict decrease.
    m = pymatching.Matching.from_detector_error_model(_CORR_DEM, enable_correlations=True)
    syndrome = np.array([1, 1, 1, 1], dtype=np.uint8)

    weights = []
    for alpha in (0.0, 0.6, 1.0):
        _, weight = m.decode(
            syndrome, return_weight=True, enable_correlations=True, alpha=alpha
        )
        weights.append(weight)
    # weights are for alpha = 0.0, 0.6, 1.0 respectively => strictly decreasing
    assert weights[0] > weights[1] > weights[2]


def test_alpha_above_one_not_clamped():
    # alpha > 1.0 is a legal, more-aggressive extrapolation and must not be clamped: it should
    # discount the correlated edges at least as much as alpha == 1.0.
    m = pymatching.Matching.from_detector_error_model(_CORR_DEM, enable_correlations=True)
    syndrome = np.array([1, 1, 1, 1], dtype=np.uint8)

    _, w_alpha_1 = m.decode(
        syndrome, return_weight=True, enable_correlations=True, alpha=1.0
    )
    _, w_alpha_15 = m.decode(
        syndrome, return_weight=True, enable_correlations=True, alpha=1.5
    )
    assert w_alpha_15 <= w_alpha_1
