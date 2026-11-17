"""Unit tests for Hyperon AtomSpace representation and pattern matching."""

import unittest
from hyperon_miner.atomspace.types import (
    AtomSpace, ConceptNode, PredicateNode, VariableNode,
    InheritanceLink, EvaluationLink, ListLink
)


class TestAtomSpace(unittest.TestCase):
    def setUp(self):
        self.space = AtomSpace()
        self.c_dolphin = ConceptNode("dolphin")
        self.c_mammal = ConceptNode("mammal")
        self.p_lives_in = PredicateNode("lives_in")
        self.c_ocean = ConceptNode("ocean")

        self.space.add(InheritanceLink((self.c_dolphin, self.c_mammal)))
        self.space.add(EvaluationLink((self.p_lives_in, ListLink((self.c_dolphin, self.c_ocean)))))

    def test_size_and_types(self):
        self.assertGreater(self.space.size(), 0)
        self.assertEqual(len(self.space.get_by_type("InheritanceLink")), 1)
        self.assertEqual(len(self.space.get_by_type("EvaluationLink")), 1)

    def test_clause_query(self):
        var_x = VariableNode("X")
        query = InheritanceLink((var_x, self.c_mammal))
        matches = self.space.query_clause(query)

        self.assertEqual(len(matches), 1)
        val, target = matches[0]
        self.assertEqual(val[var_x], self.c_dolphin)

    def test_conjunction_query(self):
        var_x = VariableNode("X")
        c1 = InheritanceLink((var_x, self.c_mammal))
        c2 = EvaluationLink((self.p_lives_in, ListLink((var_x, self.c_ocean))))

        matches = self.space.query_conjunction([c1, c2])
        self.assertEqual(len(matches), 1)
        val, witnesses = matches[0]
        self.assertEqual(val[var_x], self.c_dolphin)
        self.assertEqual(len(witnesses), 2)


if __name__ == "__main__":
    unittest.main()
