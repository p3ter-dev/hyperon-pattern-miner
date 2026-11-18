"""Unit tests for Hyperon Pattern Miner and Surprisingness calculations."""

import unittest
from hyperon_miner.atomspace.dataset import load_commonsense_atomspace
from hyperon_miner.mining.miner import HyperonPatternMiner
from hyperon_miner.mining.surprisingness import compute_surprisingness


class TestPatternMiner(unittest.TestCase):
    def setUp(self):
        self.kb = load_commonsense_atomspace()
        self.miner = HyperonPatternMiner(self.kb, min_support=2)

    def test_mining_discovery(self):
        patterns = self.miner.mine("X")
        self.assertGreater(len(patterns), 10)

        # Ensure all patterns satisfy min_support
        for p in patterns:
            self.assertGreaterEqual(p.support, 2)
            self.assertGreaterEqual(len(p.witnesses), 2)
            self.assertGreaterEqual(len(p.source_atoms), 2)

    def test_surprisingness_bounds(self):
        # High frequency pattern exceeding independent expectation
        surp = compute_surprisingness(
            pattern_support=5,
            sub_clause_supports=[5, 5],
            universe_size=16
        )
        self.assertIn(surp.label, ["surprisingly_frequent", "expected"])
        self.assertGreaterEqual(surp.score, 0.0)


if __name__ == "__main__":
    unittest.main()

