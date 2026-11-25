// Copyright 2022 PyMatching Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PYMATCHING2_IMPLIED_WEIGHT_UNCONVERTED_H
#define PYMATCHING2_IMPLIED_WEIGHT_UNCONVERTED_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "pymatching/sparse_blossom/ints.h"

namespace pm {

struct ImpliedWeightUnconverted {
    size_t node1;
    size_t node2;
    double implied_weight;
    double implied_probability = 0.0;
    /// Index (in `UserGraph::edges` order) of the edge (node1, node2) whose weight this rule updates.
    /// Used to look up a per-edge reweighting strength. SIZE_MAX if unknown.
    size_t affected_edge_index = SIZE_MAX;
    bool operator==(const ImpliedWeightUnconverted& other) const {
        return (this->node1 == other.node1) && (this->node2 == other.node2) &&
               (this->implied_weight == other.implied_weight);
    }
};

/// The strengths used to scale implied (conditional) probabilities in correlated matching.
/// If `per_edge` is non-empty, `per_edge[i]` is used for the rules that update edge `i`
/// (`ImpliedWeightUnconverted::affected_edge_index`); otherwise `global` is used for every rule.
struct ReweightingStrengths {
    double global = 1.0;
    std::vector<double> per_edge;

    bool operator==(const ReweightingStrengths& other) const {
        return global == other.global && per_edge == other.per_edge;
    }
    bool operator!=(const ReweightingStrengths& other) const {
        return !(*this == other);
    }

    /// Throws std::invalid_argument unless all strengths are positive (and not NaN).
    void validate() const {
        if (!(global > 0))
            throw std::invalid_argument("global_reg_strength must be positive");
        for (double s : per_edge) {
            if (!(s > 0))
                throw std::invalid_argument("all reg_strengths must be positive");
        }
    }

    double strength_for(const ImpliedWeightUnconverted& rule) const {
        if (per_edge.empty())
            return global;
        if (rule.affected_edge_index >= per_edge.size())
            throw std::invalid_argument(
                "Per-edge reg_strengths were given, but an implied weight rule refers to an edge index (" +
                std::to_string(rule.affected_edge_index) + ") outside the " + std::to_string(per_edge.size()) +
                " strengths provided.");
        return per_edge[rule.affected_edge_index];
    }

    /// The implied weight for `rule` after scaling its implied probability by the applicable strength.
    double implied_weight_for(const ImpliedWeightUnconverted& rule) const {
        double strength = strength_for(rule);
        if (strength == 1.0)
            return rule.implied_weight;
        double p = std::min(0.5, strength * rule.implied_probability);
        return std::log((1 - p) / p);
    }
};

struct ImpliedWeight {
    weight_int* edge0_ptr;
    weight_int* edge1_ptr;
    weight_int implied_weight;
};

}  // namespace pm

#endif  // PYMATCHING2_IMPLIED_WEIGHT_UNCONVERTED_H
