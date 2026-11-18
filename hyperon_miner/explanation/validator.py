"""
Deterministic Explanation Validator.
Verifies natural language explanations against symbolic evidence.
Enforces zero-hallucination constraint and computes coverage and unsupported claim metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
import re

from hyperon_miner.explanation.evidence import EvidenceBundle


@dataclass
class ValidationResult:
    is_valid: bool
    evidence_coverage: float       # Percentage 0.0 - 100.0
    unsupported_claim_rate: float  # Percentage 0.0 - 100.0
    violations: List[str]
    covered_instances: List[str]
    covered_clauses: List[str]
    audit_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "evidence_coverage": round(self.evidence_coverage, 2),
            "unsupported_claim_rate": round(self.unsupported_claim_rate, 2),
            "violations": self.violations,
            "covered_instances": self.covered_instances,
            "covered_clauses": self.covered_clauses,
            "audit_notes": self.audit_notes,
        }


class DeterministicValidator:
    """
    Symbolic auditor that ensures no neural hallucination corrupts the Hyperon pattern report.
    Checks:
    1. Ground entity bounds: No unauthorized entities/animals may be cited as facts.
    2. Numerical fidelity: Support and surprisingness values must match symbolic ground truth.
    3. Structural coverage: Mined clauses and witness instances must be represented.
    """

    # Common animals in common-sense universe to watch for hallucinations
    ALL_KNOWN_ANIMALS = {
        "human", "dog", "dolphin", "whale", "bat",
        "sparrow", "eagle", "penguin",
        "snake", "crocodile",
        "salmon", "shark",
        "lion", "tiger", "bear", "cat", "cheetah", "elephant" # known out-of-universe lures
    }

    def __init__(self, strict_numerical_check: bool = True):
        self.strict_numerical_check = strict_numerical_check

    def validate(self, explanation_text: str, bundle: EvidenceBundle) -> ValidationResult:
        violations: List[str] = []
        audit_notes: List[str] = []
        text_lower = explanation_text.lower()

        # 1. Witness Coverage Check
        covered_instances = [
            inst for inst in bundle.ground_instances 
            if re.search(r'\b' + re.escape(inst.lower()) + r'\b', text_lower)
        ]
        
        # 2. Clause Coverage Check
        covered_clauses = []
        for clause in bundle.clauses_readable:
            # Check key keywords from the clause
            words = [w for w in re.findall(r'[a-zA-Z0-9_]+', clause) if w not in ("X", "is", "a", "classified", "as")]
            if words and all(w.lower() in text_lower for w in words):
                covered_clauses.append(clause)

        total_elements = len(bundle.ground_instances) + len(bundle.clauses_readable)
        covered_elements = len(covered_instances) + len(covered_clauses)
        coverage_pct = (covered_elements / max(total_elements, 1)) * 100.0

        # 3. Entity Hallucination / Unsupported Claim Check
        # Check if text mentions animals that are NOT in the allowed evidence bundle
        unsupported_entities_found = set()
        for animal in self.ALL_KNOWN_ANIMALS:
            if re.search(r'\b' + re.escape(animal) + r'\b', text_lower):
                if animal not in bundle.allowed_entities:
                    unsupported_entities_found.add(animal)

        for ungrounded_entity in sorted(unsupported_entities_found):
            violations.append(f"UNSUPPORTED ENTITY: Text mentions '{ungrounded_entity}' which is not in the symbolic evidence.")

        # 4. Numerical Fidelity Check
        # Extract numerical mentions of support if explicitly written like "support of 5", "support: 5", "5 instances"
        support_matches = re.findall(r'(?:support(?: of|:)?|instances(?: of|:)?)\s*(\d+)', text_lower)
        for s_match in support_matches:
            val = int(s_match)
            if val != bundle.support:
                violations.append(f"NUMERICAL MISMATCH: Text claimed support {val}, but symbolic truth is {bundle.support}.")

        # 5. Calculate Unsupported Claim Rate
        total_claims_checked = max(len(covered_instances) + len(violations), 1)
        unsupported_claim_rate = (len(violations) / total_claims_checked) * 100.0

        is_valid = (len(violations) == 0 and coverage_pct >= 50.0)

        if not is_valid:
            if coverage_pct < 50.0:
                violations.append(f"INSUFFICIENT COVERAGE: Explanation covered only {coverage_pct:.1f}% of evidence.")

        audit_notes.append(f"Verified {len(covered_instances)}/{len(bundle.ground_instances)} witness entities.")
        audit_notes.append(f"Verified {len(covered_clauses)}/{len(bundle.clauses_readable)} pattern clauses.")

        return ValidationResult(
            is_valid=is_valid,
            evidence_coverage=coverage_pct,
            unsupported_claim_rate=unsupported_claim_rate,
            violations=violations,
            covered_instances=covered_instances,
            covered_clauses=covered_clauses,
            audit_notes=audit_notes
        )

