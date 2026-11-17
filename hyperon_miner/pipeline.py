"""
End-to-End Neuro-Symbolic Explanation Pipeline.
Integrates Hyperon AtomSpace, Pattern Miner, Evidence Extractor,
Explanation Engine, and Deterministic Validator into a unified interface.
"""

from typing import List, Dict, Any, Optional, Tuple
from hyperon_miner.atomspace.types import AtomSpace
from hyperon_miner.mining.miner import HyperonPatternMiner, MinedPattern
from hyperon_miner.explanation.evidence import EvidenceExtractor, EvidenceBundle
from hyperon_miner.explanation.generator import ExplanationEngine, ExplanationOutput
from hyperon_miner.explanation.validator import DeterministicValidator


class HyperonExplanationPipeline:
    """Complete Neuro-Symbolic Pipeline for Hyperon Pattern Explanation."""

    def __init__(
        self,
        atomspace: AtomSpace,
        min_support: int = 2,
        prefer_llm: bool = False
    ):
        self.atomspace = atomspace
        self.miner = HyperonPatternMiner(atomspace, min_support=min_support)
        self.validator = DeterministicValidator()
        self.explanation_engine = ExplanationEngine(prefer_llm=prefer_llm, validator=self.validator)

    def run_mining(self, target_variable: str = "X") -> List[MinedPattern]:
        """Step 1: Deterministic Pattern Mining."""
        return self.miner.mine(target_variable)

    def explain_pattern(self, pattern: MinedPattern, target_variable: str = "X") -> ExplanationOutput:
        """Step 2: Evidence Extraction & Validated Explanation Generation."""
        bundle = EvidenceExtractor.extract(pattern, target_var=target_variable)
        return self.explanation_engine.generate_and_validate(bundle)

    def process_all(self, target_variable: str = "X", limit: Optional[int] = None) -> List[Tuple[MinedPattern, EvidenceBundle, ExplanationOutput]]:
        """Executes full mining and explanation across patterns."""
        patterns = self.run_mining(target_variable)
        if limit:
            patterns = patterns[:limit]

        results = []
        for p in patterns:
            bundle = EvidenceExtractor.extract(p, target_var=target_variable)
            output = self.explanation_engine.generate_and_validate(bundle)
            results.append((p, bundle, output))

        return results
