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

#include "pymatching/sparse_blossom/ints.h"

namespace pm {

struct ImpliedWeightUnconverted {
    size_t node1;
    size_t node2;
    // The implied weight of the affected edge under hard Bayesian conditioning (alpha == 1.0),
    // i.e. log((1 - p) / p) with p = min(0.5, P(mu | nu)). This is the value used by the
    // (unchanged) fast path when alpha == 1.0.
    double implied_weight;
    // The two endpoints of Pearl's soft-evidence (virtual evidence) mixture, used when alpha != 1.0
    // to recompute the implied weight at decode time as
    //   implied_p = alpha * implied_p_pos + (1 - alpha) * implied_p_neg.
    // implied_p_pos = P(mu | nu)      (unclamped; the same quantity the alpha == 1.0 path clamps)
    // implied_p_neg = P(mu | not nu)  (floored at 0).
    double implied_p_pos = 0.0;
    double implied_p_neg = 0.0;
    bool operator==(const ImpliedWeightUnconverted& other) const {
        return (this->node1 == other.node1) && (this->node2 == other.node2) &&
               (this->implied_weight == other.implied_weight);
    }
};

struct ImpliedWeight {
    weight_int* edge0_ptr;
    weight_int* edge1_ptr;
    weight_int implied_weight;
};

}  // namespace pm

#endif  // PYMATCHING2_IMPLIED_WEIGHT_UNCONVERTED_H
