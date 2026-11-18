"""Unit tests for Deterministic Explanation Validator."""

import unittest
from hyperon_miner.explanation.evidence import EvidenceBundle
from hyperon_miner.explanation.validator import DeterministicValidator


class TestDeterministicValidator(unittest.TestCase):
    def setUp(self):
        self.validator = DeterministicValidator()
        self.bundle = EvidenceBundle(
            pattern_id="PAT-TEST",
            s_expression='(AndLink (InheritanceLink $X (ConceptNode "mammal")) (EvaluationLink (PredicateNode "lives_in") (ListLink $X (ConceptNode "ocean"))))',
            clauses_readable=["$X is classified as a mammal", "$X lives_in ocean"],
            support=2,
            empirical_prob=0.125,
            expected_interval=(0.02, 0.05),
            surprisingness_score=1.5,
            surprisingness_label="surprisingly_frequent",
            target_variable="X",
            ground_instances=["dolphin", "whale"],
            source_atoms=['(InheritanceLink (ConceptNode "dolphin") (ConceptNode "mammal"))'],
            allowed_entities={"mammal", "dolphin", "whale", "ocean"},
            allowed_predicates={"lives_in"},
            allowed_attributes={"ocean"}
        )

    def test_valid_explanation_passes(self):
        valid_text = (
            "Pattern PAT-TEST describes entities classified as a mammal that lives_in ocean. "
            "A total support of 2 instances was found: dolphin and whale. "
            "This pattern is surprisingly frequent."
        )
        res = self.validator.validate(valid_text, self.bundle)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.unsupported_claim_rate, 0.0)
        self.assertGreaterEqual(res.evidence_coverage, 75.0)

    def test_hallucinated_entity_rejected(self):
        # Adding an ungrounded animal: 'cheetah'
        hallucinated_text = (
            "Pattern PAT-TEST describes entities classified as a mammal that lives_in ocean. "
            "Instances found: dolphin, whale, and cheetah. "
            "Support of 2 was observed."
        )
        res = self.validator.validate(hallucinated_text, self.bundle)
        self.assertFalse(res.is_valid)
        self.assertGreater(res.unsupported_claim_rate, 0.0)
        self.assertTrue(any("cheetah" in v for v in res.violations))

    def test_numerical_mismatch_rejected(self):
        # Claiming incorrect support count: 99
        tampered_text = (
            "Pattern PAT-TEST describes entities classified as a mammal that lives_in ocean. "
            "A total support of 99 instances was found: dolphin and whale."
        )
        res = self.validator.validate(tampered_text, self.bundle)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("NUMERICAL MISMATCH" in v for v in res.violations))


if __name__ == "__main__":
    unittest.main()

