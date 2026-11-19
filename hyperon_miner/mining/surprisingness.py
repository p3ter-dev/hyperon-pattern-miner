"""
Surprisingness computation for Hyperon Pattern Miner.
Computes empirical probability, partitions conjuncts for block independence estimation,
and calculates [e_min, e_max] bounds and divergence scores.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import math


@dataclass(frozen=True)
class SurprisingnessResult:
    empirical_prob: float
    expected_interval: Tuple[float, float]  # (e_min, e_max)
    score: float                           # Divergence metric
    label: str                             # "surpassingly_frequent", "surprisingly_rare", "expected"


def compute_surprisingness(
    pattern_support: int,
    sub_clause_supports: List[int],
    universe_size: int,
    epsilon: float = 1e-6
) -> SurprisingnessResult:
    """
    Computes surprisingness following OpenCog Hyperon's partition model:
    - empirical_prob = pattern_support / universe_size
    - Independent expectation = Product of sub_clause probabilities.
    - Bounds [e_min, e_max] capture uncertainty/dependence slack.
    """
    if universe_size <= 0:
        return SurprisingnessResult(0.0, (0.0, 0.0), 0.0, "expected")

    p_empirical = pattern_support / universe_size # observed probability of the pattern

    if not sub_clause_supports:
        return SurprisingnessResult(p_empirical, (p_empirical, p_empirical), 0.0, "expected")

    # under block independence, expected joint probability is product of marginals
    # marginal probabilities of individual pieces
    clause_probs = [s / universe_size for s in sub_clause_supports]
    p_independent = 1.0
    for cp in clause_probs:
        p_independent *= max(cp, epsilon) # joint probability of independent clauses

    # Upper and lower uncertainty bounds
    # expected interval
    e_min = max(p_independent * 0.75, epsilon) # lower bound of independence
    e_max = min(p_independent * 1.35, 1.0) # upper bound of independence

    if p_empirical > e_max:
        # surprisingly frequent: empirical probability exceeds upper bound of independence
        # score measures the relative divergence from the independence interval
        score = (p_empirical - e_max) / max(e_max, epsilon)
        label = "surprisingly_frequent"
    elif p_empirical < e_min:
        # Surprisingly rare
        score = (e_min - p_empirical) / max(e_min, epsilon)
        label = "surprisingly_rare"
    else:
        score = 0.0
        label = "expected"

    return SurprisingnessResult(
        empirical_prob=round(p_empirical, 4),
        expected_interval=(round(e_min, 4), round(e_max, 4)),
        score=round(score, 4),
        label=label
    )

