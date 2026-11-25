# Copyright 2022 PyMatching Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import stim

import pymatching

# Edges (0, 1) and (2, 3) each have marginal probability 0.2, and P((2, 3) | (0, 1)) = 0.08 / 0.2 = 0.4.
DEM = stim.DetectorErrorModel("""
    error(0.14285714285714285) D0 D1
    error(0.14285714285714285) D2 D3
    error(0.08) D0 D1 ^ D2 D3
""")

# Asymmetric marginals: edge A = (0, 1) has marginal 0.2, edge B = (2, 3) has marginal 0.1, with joint 0.08.
# So P(B | A) = 0.4 and P(A | B) = 0.8.
ASYMMETRIC_DEM = stim.DetectorErrorModel("""
    error(0.14285714285714285) D0 D1
    error(0.023809523809523808) D2 D3
    error(0.08) D0 D1 ^ D2 D3
""")

SYNDROME = np.array([1, 1, 1, 1], dtype=np.uint8)
SHOTS = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.uint8)


def _w(p):
    return np.log((1 - p) / p)


def _edge_index(m, u, v):
    return next(i for i, (a, b, _) in enumerate(m.edges()) if {a, b} == {u, v})


def _weight(m, **kwargs):
    return m.decode(SYNDROME, return_weight=True, enable_correlations=True, **kwargs)[1]


def _asymmetric_expected_weight(s_a, s_b):
    # Both edges are in the first-pass matching. A is reweighted using P(A | B) and A's strength, B using P(B | A)
    # and B's strength. Reweighting only ever lowers an edge weight.
    return min(_w(0.2), _w(min(0.5, s_a * 0.8))) + min(_w(0.1), _w(min(0.5, s_b * 0.4)))


# --- global_reg_strength -----------------------------------------------------------------------------------------


def test_default_global_reg_strength_is_one():
    m = pymatching.Matching.from_detector_error_model(DEM, enable_correlations=True)
    assert m.global_reg_strength == 1.0
    assert m.reg_strengths is None
    default = m.decode_batch(SHOTS, return_weights=True, enable_correlations=True)
    explicit = m.decode_batch(SHOTS, return_weights=True, enable_correlations=True, global_reg_strength=1.0)
    np.testing.assert_array_equal(default[0], explicit[0])
    np.testing.assert_array_equal(default[1], explicit[1])
    np.testing.assert_array_equal(
        m.decode_to_edges_array(SYNDROME, enable_correlations=True),
        m.decode_to_edges_array(SYNDROME, enable_correlations=True, global_reg_strength=1.0),
    )


@pytest.mark.parametrize("strength", [0.6, 1.0, 1.5])
def test_global_reg_strength_scales_conditional_probability(strength):
    m = pymatching.Matching.from_detector_error_model(DEM, enable_correlations=True)
    p = min(0.5, strength * 0.4)
    assert _weight(m, global_reg_strength=strength) == pytest.approx(2 * _w(p), abs=1e-3)


def test_changing_global_reg_strength_between_calls():
    m = pymatching.Matching.from_detector_error_model(DEM, enable_correlations=True)
    weights = [_weight(m, global_reg_strength=s) for s in (1.0, 0.6, 1.0)]
    assert weights[0] == weights[2] != weights[1]


@pytest.mark.parametrize("strength", [0, -1.0, float("nan")])
def test_global_reg_strength_must_be_positive(strength):
    m = pymatching.Matching.from_detector_error_model(DEM, enable_correlations=True)
    with pytest.raises(ValueError):
        m.decode(SYNDROME, enable_correlations=True, global_reg_strength=strength)
    with pytest.raises(ValueError):
        m.set_global_reg_strength(strength)


def test_set_global_reg_strength():
    m = pymatching.Matching.from_detector_error_model(DEM, enable_correlations=True)
    m.set_global_reg_strength(0.6)
    assert m.global_reg_strength == 0.6
    assert _weight(m) == _weight(m, global_reg_strength=0.6)
    stored = m.decode_batch(SHOTS, return_weights=True, enable_correlations=True)
    explicit = m.decode_batch(SHOTS, return_weights=True, enable_correlations=True, global_reg_strength=0.6)
    np.testing.assert_array_equal(stored[1], explicit[1])
    m.set_global_reg_strength(None)
    assert m.global_reg_strength == 1.0
    assert _weight(m) == _weight(m, global_reg_strength=1.0)


# --- reg_strengths as an array -----------------------------------------------------------------------------------


def test_uniform_reg_strengths_match_global_reg_strength():
    m = pymatching.Matching.from_detector_error_model(DEM, enable_correlations=True)
    for s in (0.6, 1.0, 1.5):
        glob = m.decode_batch(SHOTS, return_weights=True, enable_correlations=True, global_reg_strength=s)
        per_edge = m.decode_batch(
            SHOTS, return_weights=True, enable_correlations=True, reg_strengths=[s] * m.num_edges
        )
        np.testing.assert_array_equal(glob[0], per_edge[0])
        np.testing.assert_array_equal(glob[1], per_edge[1])


@pytest.mark.parametrize("s_a,s_b", [(0.25, 1.0), (1.0, 0.25), (0.5, 2.0), (1.0, 1.0)])
def test_reg_strengths_use_strength_of_reweighted_edge(s_a, s_b):
    m = pymatching.Matching.from_detector_error_model(ASYMMETRIC_DEM, enable_correlations=True)
    strengths = np.ones(m.num_edges)
    strengths[_edge_index(m, 0, 1)] = s_a
    strengths[_edge_index(m, 2, 3)] = s_b
    assert _weight(m, reg_strengths=strengths) == pytest.approx(_asymmetric_expected_weight(s_a, s_b), abs=1e-3)


def test_reg_strengths_consistent_across_decode_methods():
    m = pymatching.Matching.from_detector_error_model(ASYMMETRIC_DEM, enable_correlations=True)
    strengths = np.linspace(0.3, 1.2, m.num_edges)
    preds, weights = m.decode_batch(SHOTS, return_weights=True, enable_correlations=True, reg_strengths=strengths)
    for shot, pred, weight in zip(SHOTS, preds, weights):
        p, w = m.decode(shot, return_weight=True, enable_correlations=True, reg_strengths=list(strengths))
        np.testing.assert_array_equal(p, pred)
        assert w == weight
    np.testing.assert_array_equal(
        m.decode_to_edges_array(SYNDROME, enable_correlations=True, reg_strengths=strengths),
        m.decode_to_edges_array(SYNDROME, enable_correlations=True),
    )


def test_switching_between_global_and_per_edge_strengths():
    m = pymatching.Matching.from_detector_error_model(ASYMMETRIC_DEM, enable_correlations=True)
    strengths = np.ones(m.num_edges)
    strengths[_edge_index(m, 2, 3)] = 0.25
    weights = [
        _weight(m),
        _weight(m, reg_strengths=strengths),
        _weight(m, global_reg_strength=0.25),
        _weight(m, reg_strengths=strengths),
        _weight(m),
    ]
    assert weights[0] == weights[4]
    assert weights[1] == weights[3]
    assert len({weights[0], weights[1], weights[2]}) == 3


# --- reg_strengths as a dict -------------------------------------------------------------------------------------


@pytest.mark.parametrize("s_a,s_b", [(0.25, 1.0), (1.0, 0.25), (0.5, 2.0)])
def test_reg_strengths_dict_matches_array(s_a, s_b):
    m = pymatching.Matching.from_detector_error_model(ASYMMETRIC_DEM, enable_correlations=True)
    strengths = np.ones(m.num_edges)
    strengths[_edge_index(m, 0, 1)] = s_a
    strengths[_edge_index(m, 2, 3)] = s_b
    expected = _asymmetric_expected_weight(s_a, s_b)
    # Node order within an edge does not matter.
    assert _weight(m, reg_strengths={(0, 1): s_a, (3, 2): s_b}) == pytest.approx(expected, abs=1e-3)
    assert _weight(m, reg_strengths={(0, 1): s_a, (3, 2): s_b}) == _weight(m, reg_strengths=strengths)


def test_reg_strengths_dict_missing_edges_default_to_one():
    m = pymatching.Matching.from_detector_error_model(ASYMMETRIC_DEM, enable_correlations=True)
    assert _weight(m, reg_strengths={(2, 3): 0.25}) == pytest.approx(_asymmetric_expected_weight(1.0, 0.25), abs=1e-3)
    assert _weight(m, reg_strengths={}) == _weight(m)


def test_reg_strengths_dict_boundary_edges():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1
        error(0.1) D1
        error(0.05) D0 D1 ^ D1
    """)
    m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    strengths = np.ones(m.num_edges)
    strengths[next(i for i, (_, v, _) in enumerate(m.edges()) if v is None)] = 0.3
    for key in [(1, None), (None, 1)]:
        m.set_reg_strengths({key: 0.3})
        assert m.reg_strengths == {(0, 1): 1.0, (1, None): 0.3}
        for shot in [[1, 0], [0, 1], [1, 1]]:
            assert _weight_of(m, shot) == _weight_of(m, shot, reg_strengths=strengths)


def _weight_of(m, shot, **kwargs):
    return m.decode(np.array(shot, dtype=np.uint8), return_weight=True, enable_correlations=True, **kwargs)[1]


@pytest.mark.parametrize(
    "reg_strengths",
    [
        {(0, 2): 0.5},  # not an edge
        {(0, None): 0.5},  # no boundary edge at node 0
        {(0, 1): 0.5, (1, 0): 0.6},  # the same edge twice
        {0: 0.5},  # not an edge tuple
        {(0, 1, 2): 0.5},  # not an edge tuple
        {(0, 1): 0.0},  # not positive
    ],
)
def test_invalid_reg_strengths_dict(reg_strengths):
    m = pymatching.Matching.from_detector_error_model(DEM, enable_correlations=True)
    with pytest.raises(ValueError):
        m.decode(SYNDROME, enable_correlations=True, reg_strengths=reg_strengths)
    with pytest.raises(ValueError):
        m.set_reg_strengths(reg_strengths)


# --- set_reg_strengths ---------------------------------------------------------------------------------------------


def test_set_reg_strengths():
    m = pymatching.Matching.from_detector_error_model(ASYMMETRIC_DEM, enable_correlations=True)
    m.set_reg_strengths({(1, 0): 0.25})
    assert m.reg_strengths == {(0, 1): 0.25, (2, 3): 1.0}
    expected = _asymmetric_expected_weight(0.25, 1.0)
    assert _weight(m) == pytest.approx(expected, abs=1e-3)
    _, weights = m.decode_batch(SHOTS, return_weights=True, enable_correlations=True)
    assert weights[-1] == pytest.approx(expected, abs=1e-3)

    # Arrays are accepted too.
    strengths = np.ones(m.num_edges)
    strengths[_edge_index(m, 2, 3)] = 0.25
    m.set_reg_strengths(strengths)
    assert m.reg_strengths == {(0, 1): 1.0, (2, 3): 0.25}
    assert _weight(m) == pytest.approx(_asymmetric_expected_weight(1.0, 0.25), abs=1e-3)

    # None resets to the default.
    m.set_reg_strengths(None)
    assert m.reg_strengths is None
    assert _weight(m) == _weight(m, global_reg_strength=1.0)


def test_setters_replace_each_other():
    m = pymatching.Matching.from_detector_error_model(ASYMMETRIC_DEM, enable_correlations=True)
    m.set_reg_strengths({(0, 1): 0.25})
    m.set_global_reg_strength(0.6)
    assert m.reg_strengths is None
    assert m.global_reg_strength == 0.6
    assert _weight(m) == _weight(m, global_reg_strength=0.6)
    m.set_reg_strengths({(0, 1): 0.25})
    assert m.global_reg_strength == 1.0
    assert _weight(m) == _weight(m, reg_strengths={(0, 1): 0.25})


def test_call_arguments_override_stored_strengths():
    m = pymatching.Matching.from_detector_error_model(ASYMMETRIC_DEM, enable_correlations=True)
    m.set_reg_strengths({(0, 1): 0.25})
    stored = _weight(m)
    fresh = pymatching.Matching.from_detector_error_model(ASYMMETRIC_DEM, enable_correlations=True)
    assert _weight(m, global_reg_strength=0.6) == _weight(fresh, global_reg_strength=0.6)
    assert _weight(m, reg_strengths={(2, 3): 0.25}) == _weight(fresh, reg_strengths={(2, 3): 0.25})
    # The stored strengths are used again once no strengths are passed.
    assert _weight(m) == stored


def test_stored_reg_strengths_must_be_reset_after_adding_edges():
    m = pymatching.Matching.from_detector_error_model(DEM, enable_correlations=True)
    m.set_reg_strengths({(0, 1): 0.5})
    m.add_edge(1, 2, weight=1.0)
    with pytest.raises(ValueError, match="set_reg_strengths"):
        _weight(m)
    m.set_reg_strengths({(0, 1): 0.5})
    _weight(m)


# --- invalid combinations ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(reg_strengths=[1.0]),
        dict(reg_strengths=[1.0, 1.0, 1.0, 1.0]),
        dict(reg_strengths=[1.0, 0.0]),
        dict(reg_strengths=[1.0, -0.5]),
        dict(reg_strengths=[1.0, float("nan")]),
        dict(reg_strengths=[[1.0, 1.0]]),
        dict(reg_strengths=[1.0, 1.0], global_reg_strength=0.5),
        dict(reg_strengths=[1.0, 1.0], global_reg_strength=1.0),
        dict(reg_strengths={(0, 1): 0.5}, global_reg_strength=1.0),
    ],
)
def test_invalid_reg_strengths(kwargs):
    m = pymatching.Matching.from_detector_error_model(DEM, enable_correlations=True)
    assert m.num_edges == 2
    with pytest.raises(ValueError):
        m.decode(SYNDROME, enable_correlations=True, **kwargs)
    with pytest.raises(ValueError):
        m.decode_batch(SHOTS, enable_correlations=True, **kwargs)
    with pytest.raises(ValueError):
        m.decode_to_edges_array(SYNDROME, enable_correlations=True, **kwargs)
