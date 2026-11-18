"""
Evidence Data Extraction Layer.
Packages mined symbolic patterns, variable valuations, and source hyperedges
into a deterministic, strictly bounded EvidenceBundle.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Any
import json
import re

from hyperon_miner.mining.miner import MinedPattern


@dataclass
class EvidenceBundle:
    """Strictly bounded factual representation of a mined pattern."""
    pattern_id: str
    s_expression: str
    clauses_readable: List[str]
    support: int
    empirical_prob: float
    expected_interval: Tuple[float, float]
    surprisingness_score: float
    surprisingness_label: str
    target_variable: str
    ground_instances: List[str]
    source_atoms: List[str]
    allowed_entities: Set[str]
    allowed_predicates: Set[str]
    allowed_attributes: Set[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "s_expression": self.s_expression,
            "clauses_readable": self.clauses_readable,
            "support": self.support,
            "empirical_prob": self.empirical_prob,
            "expected_interval": list(self.expected_interval),
            "surprisingness_score": self.surprisingness_score,
            "surprisingness_label": self.surprisingness_label,
            "target_variable": self.target_variable,
            "ground_instances": self.ground_instances,
            "source_atoms": self.source_atoms,
            "allowed_entities": sorted(list(self.allowed_entities)),
            "allowed_predicates": sorted(list(self.allowed_predicates)),
            "allowed_attributes": sorted(list(self.allowed_attributes)),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class EvidenceExtractor:
    """Transforms raw MinedPattern structures into structured, verified EvidenceBundles."""

    @staticmethod
    def _clause_to_readable(clause_str: str) -> str:
        """Converts raw S-expression clause to human-readable relationship description."""
        # e.g., (InheritanceLink $X (ConceptNode "mammal")) -> "$X is a mammal"
        m_inh = re.search(r'\(InheritanceLink\s+\$([A-Za-z0-9_]+)\s+\(ConceptNode\s+"([^"]+)"\)\)', clause_str)
        if m_inh:
            var, category = m_inh.groups()
            return f"${var} is classified as a {category}"

        # e.g., (EvaluationLink (PredicateNode "lives_in") (ListLink $X (ConceptNode "ocean"))) -> "$X lives_in ocean"
        m_eval = re.search(r'\(EvaluationLink\s+\(PredicateNode\s+"([^"]+)"\)\s+\(ListLink\s+\$([A-Za-z0-9_]+)\s+\(ConceptNode\s+"([^"]+)"\)\)\)', clause_str)
        if m_eval:
            pred, var, attr = m_eval.groups()
            return f"${var} {pred} {attr}"

        return clause_str

    @classmethod
    def extract(cls, pattern: MinedPattern, target_var: str = "X") -> EvidenceBundle:
        """Extracts complete bounded evidence from a MinedPattern."""
        clauses_readable = [cls._clause_to_readable(c.to_s_expr()) for c in pattern.clauses]
        ground_instances = [w[target_var] for w in pattern.witnesses if target_var in w]

        allowed_entities = set(ground_instances)
        allowed_predicates = set()
        allowed_attributes = set()

        for c_str in [c.to_s_expr() for c in pattern.clauses]:
            # Extract concept nodes
            for concept in re.findall(r'\(ConceptNode\s+"([^"]+)"\)', c_str):
                if concept in ["mammal", "bird", "reptile", "fish", "animal"]:
                    allowed_entities.add(concept)
                else:
                    allowed_attributes.add(concept)

            # Extract predicates
            for pred in re.findall(r'\(PredicateNode\s+"([^"]+)"\)', c_str):
                allowed_predicates.add(pred)

        # Also pull in exact ground concepts from source atoms
        for sa in pattern.source_atoms:
            for concept in re.findall(r'\(ConceptNode\s+"([^"]+)"\)', sa):
                allowed_entities.add(concept)

        return EvidenceBundle(
            pattern_id=pattern.pattern_id,
            s_expression=pattern.to_s_expr(),
            clauses_readable=clauses_readable,
            support=pattern.support,
            empirical_prob=pattern.empirical_prob,
            expected_interval=pattern.expected_interval,
            surprisingness_score=pattern.surprisingness_score,
            surprisingness_label=pattern.surprisingness_label,
            target_variable=target_var,
            ground_instances=sorted(ground_instances),
            source_atoms=pattern.source_atoms,
            allowed_entities=allowed_entities,
            allowed_predicates=allowed_predicates,
            allowed_attributes=allowed_attributes
        )

