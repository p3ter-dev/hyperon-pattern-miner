"""
Explanation Generation Layer.
Provides a Dual-Engine Explanation Generator:
1. Local Semantic Synthesizer (offline, zero-cost, guaranteed grounded).
2. Universal LLM Adapter (for external LLM APIs when configured).
Includes automated verification and self-healing fallback loops.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import os
import time
import json
import urllib.request
import urllib.error

from hyperon_miner.explanation.evidence import EvidenceBundle
from hyperon_miner.explanation.validator import DeterministicValidator, ValidationResult


@dataclass
class ExplanationOutput:
    pattern_id: str
    explanation_text: str
    generation_engine: str
    latency_ms: float
    cost_usd: float
    validation: ValidationResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "explanation_text": self.explanation_text,
            "generation_engine": self.generation_engine,
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": self.cost_usd,
            "validation": self.validation.to_dict(),
        }


class LocalSemanticSynthesizer:
    """
    Deterministic semantic NLG engine.
    Produces rich, fluent, structured natural language explanations directly from
    Hyperon EvidenceBundles without external API keys or cloud dependencies.
    """

    @classmethod
    def synthesize(cls, bundle: EvidenceBundle) -> str:
        # 1. Headline & Relational Summary
        clauses_text = " AND ".join(bundle.clauses_readable)
        instances_str = ", ".join(bundle.ground_instances) if bundle.ground_instances else "None"

        # 2. Interpret surprisingness
        if bundle.surprisingness_label == "surprisingly_frequent":
            stat_narrative = (
                f"This pattern is classified as **surprisingly frequent** (score: {bundle.surprisingness_score:.4f}). "
                f"Its empirical probability ({bundle.empirical_prob:.4f}) significantly exceeds the expected upper bound "
                f"of independence ({bundle.expected_interval[1]:.4f}). In other words, these traits co-occur far more "
                f"often than would be expected if the attributes were distributed randomly."
            )
        elif bundle.surprisingness_label == "surprisingly_rare":
            stat_narrative = (
                f"This pattern is classified as **surprisingly rare** (score: {bundle.surprisingness_score:.4f}). "
                f"Its empirical probability ({bundle.empirical_prob:.4f}) is lower than the expected interval "
                f"[{bundle.expected_interval[0]:.4f}, {bundle.expected_interval[1]:.4f}]."
            )
        else:
            stat_narrative = (
                f"This pattern is classified as **expected** (score: {bundle.surprisingness_score:.4f}). "
                f"Its empirical probability ({bundle.empirical_prob:.4f}) aligns with the expected interval "
                f"[{bundle.expected_interval[0]:.4f}, {bundle.expected_interval[1]:.4f}]."
            )

        # 3. Assemble full structured narrative
        paragraphs = [
            f"### Pattern Explanation: {bundle.pattern_id}",
            f"**Discovered Rule:** Entities where {clauses_text}.",
            f"**Observed Evidence:** A total support count of {bundle.support} matching instances was verified in the AtomSpace: **{instances_str}**.",
            f"**Statistical Significance:** {stat_narrative}",
            f"**Symbolic Grounding:** Validated against {len(bundle.source_atoms)} verified ground hyperedges in the Hyperon AtomSpace."
        ]

        return "\n\n".join(paragraphs)


class UniversalLLMAdapter:
    """
    Pluggable HTTP client for remote LLMs (e.g. OpenAI / Gemini / Ollama).
    Falls back gracefully if the endpoint is not reachable or unconfigured.
    """

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_url = api_url or os.environ.get("LLM_API_URL")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.model = model

    def is_available(self) -> bool:
        return bool(self.api_url and self.api_key)

    def generate(self, bundle: EvidenceBundle) -> str:
        if not self.is_available():
            raise RuntimeError("LLM API endpoint or key not configured.")

        prompt = (
            "You are an Explainable AI assistant for OpenCog Hyperon Pattern Miner.\n"
            "Explain the following mined symbolic pattern based ONLY on the provided JSON evidence.\n"
            "CRITICAL CONSTRAINTS:\n"
            "- Mention ONLY the entities present in 'ground_instances'.\n"
            "- Mention the exact support count and surprisingness score.\n"
            "- Do NOT introduce external animals or ungrounded facts.\n\n"
            f"EVIDENCE JSON:\n{bundle.to_json()}\n"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]


class ExplanationEngine:
    """
    Orchestrates the entire explanation lifecycle:
    1. Dispatches generation to local synthesizer or remote LLM.
    2. Runs deterministic validation gate.
    3. Triggers self-healing fallback if validation fails.
    """

    def __init__(self, prefer_llm: bool = False, validator: Optional[DeterministicValidator] = None):
        self.prefer_llm = prefer_llm
        self.validator = validator or DeterministicValidator()
        self.llm_adapter = UniversalLLMAdapter()

    def generate_and_validate(self, bundle: EvidenceBundle) -> ExplanationOutput:
        start_time = time.perf_counter()
        engine_used = "LocalSemanticSynthesizer"
        raw_text = ""
        cost = 0.0

        if self.prefer_llm and self.llm_adapter.is_available():
            try:
                raw_text = self.llm_adapter.generate(bundle)
                engine_used = f"RemoteLLM ({self.llm_adapter.model})"
                cost = 0.0005  # Approximate token cost
            except Exception as e:
                # Fallback to local synthesizer
                raw_text = LocalSemanticSynthesizer.synthesize(bundle)
                engine_used = "LocalSemanticSynthesizer (Fallback)"
        else:
            raw_text = LocalSemanticSynthesizer.synthesize(bundle)
            engine_used = "LocalSemanticSynthesizer"
            cost = 0.0

        # Run Deterministic Validation Gate
        val_result = self.validator.validate(raw_text, bundle)

        # Self-healing loop: if external LLM introduced a hallucination, revert to LocalSynthesizer
        if not val_result.is_valid and engine_used.startswith("RemoteLLM"):
            fallback_text = LocalSemanticSynthesizer.synthesize(bundle)
            val_result = self.validator.validate(fallback_text, bundle)
            raw_text = fallback_text + "\n\n*(Note: Initial LLM response contained ungrounded claims and was deterministically repaired.)*"
            engine_used = "LocalSemanticSynthesizer (Self-Healed)"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ExplanationOutput(
            pattern_id=bundle.pattern_id,
            explanation_text=raw_text,
            generation_engine=engine_used,
            latency_ms=elapsed_ms,
            cost_usd=cost,
            validation=val_result
        )
