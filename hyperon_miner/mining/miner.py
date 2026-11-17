"""
Hyperon Pattern Miner Engine.
Implements frequent pattern discovery, conjunction expansion, variable grounding,
and evidence extraction over an AtomSpace.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from hyperon_miner.atomspace.types import (
    Atom, Node, Link, ConceptNode, PredicateNode, VariableNode,
    ListLink, InheritanceLink, EvaluationLink, AndLink,
    AtomSpace, Valuation
)
from hyperon_miner.mining.surprisingness import compute_surprisingness, SurprisingnessResult


@dataclass
class MinedPattern:
    """Represents a discovered symbolic pattern with complete statistical and structural evidence."""
    pattern_id: str
    clauses: List[Atom]
    variables: List[VariableNode]
    support: int
    empirical_prob: float
    expected_interval: Tuple[float, float]
    surprisingness_score: float
    surprisingness_label: str
    # Ground variable substitutions satisfying the pattern: [{ $X: ConceptNode("dolphin") }, ...]
    witnesses: List[Dict[str, str]]
    # The concrete ground hypergraph atoms in AtomSpace that formed the matches
    source_atoms: List[str]

    def to_s_expr(self) -> str:
        clauses_str = " ".join(c.to_s_expr() for c in self.clauses)
        if len(self.clauses) > 1:
            return f"(AndLink {clauses_str})"
        return clauses_str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "s_expression": self.to_s_expr(),
            "support": self.support,
            "empirical_prob": self.empirical_prob,
            "expected_interval": list(self.expected_interval),
            "surprisingness_score": self.surprisingness_score,
            "surprisingness_label": self.surprisingness_label,
            "witnesses": self.witnesses,
            "source_atoms": self.source_atoms,
        }


class HyperonPatternMiner:
    """
    Symbolic pattern miner for Hyperon AtomSpace.
    Performs deterministic pattern extraction, variable binding, and surprisingness scoring.
    """
    def __init__(self, atomspace: AtomSpace, min_support: int = 2):
        self.atomspace = atomspace
        self.min_support = min_support

    def _get_universe_size(self, target_var: VariableNode) -> int:
        """Determines the number of distinct entities available for binding target_var."""
        entities = set()
        for atom in self.atomspace.get_by_type("InheritanceLink"):
            if isinstance(atom, InheritanceLink) and isinstance(atom.child, ConceptNode):
                entities.add(atom.child.name)
        return max(len(entities), 1)

    def extract_single_clause_candidates(self, target_var: VariableNode) -> List[Tuple[Atom, int, List[Valuation], List[Atom]]]:
        """
        Generates primitive candidate clauses containing target_var and measures their support.
        """
        candidates: List[Tuple[Atom, int, List[Valuation], List[Atom]]] = []

        # 1. Candidate InheritanceLinks: (InheritanceLink $X (ConceptNode ?Category))
        categories = {a.parent for a in self.atomspace.get_by_type("InheritanceLink") if isinstance(a, InheritanceLink)}
        for cat in categories:
            clause = InheritanceLink((target_var, cat))
            matches = self.atomspace.query_clause(clause)
            if len(matches) >= self.min_support:
                vals = [m[0] for m in matches]
                src = [m[1] for m in matches]
                candidates.append((clause, len(matches), vals, src))

        # 2. Candidate EvaluationLinks: (EvaluationLink ?Predicate (ListLink $X ?Attribute))
        eval_links = [a for a in self.atomspace.get_by_type("EvaluationLink") if isinstance(a, EvaluationLink)]
        predicate_arg_pairs = set()
        for el in eval_links:
            if isinstance(el.predicate, PredicateNode) and len(el.arguments) == 2:
                predicate_arg_pairs.add((el.predicate, el.arguments[1]))

        for pred, target_attr in predicate_arg_pairs:
            clause = EvaluationLink((pred, ListLink((target_var, target_attr))))
            matches = self.atomspace.query_clause(clause)
            if len(matches) >= self.min_support:
                vals = [m[0] for m in matches]
                src = [m[1] for m in matches]
                candidates.append((clause, len(matches), vals, src))

        return candidates

    def mine(self, target_var_name: str = "X") -> List[MinedPattern]:
        """
        Executes pattern mining:
        1. Extract single clause patterns satisfying min_support.
        2. Expand into conjunctions (AndLink c1 c2).
        3. Match conjunctions and collect variable bindings + ground source atoms.
        4. Calculate surprisingness bounds and scores.
        """
        target_var = VariableNode(target_var_name)
        universe_size = self._get_universe_size(target_var)

        single_candidates = self.extract_single_clause_candidates(target_var)
        mined_patterns: List[MinedPattern] = []
        pattern_counter = 1

        # Record single clause patterns
        single_support_map: Dict[str, int] = {}
        for clause, supp, vals, src in single_candidates:
            c_key = clause.to_s_expr()
            single_support_map[c_key] = supp

            surp = compute_surprisingness(supp, [supp], universe_size)
            witness_dicts = [{v.name: atom.name if isinstance(atom, Node) else atom.to_s_expr() 
                              for v, atom in val.items()} for val in vals]
            source_atom_strs = [a.to_s_expr() for a in src]

            pattern = MinedPattern(
                pattern_id=f"PAT-{pattern_counter:03d}",
                clauses=[clause],
                variables=[target_var],
                support=supp,
                empirical_prob=surp.empirical_prob,
                expected_interval=surp.expected_interval,
                surprisingness_score=surp.score,
                surprisingness_label=surp.label,
                witnesses=witness_dicts,
                source_atoms=source_atom_strs
            )
            mined_patterns.append(pattern)
            pattern_counter += 1

        # Conjunction expansion: pairwise combinations
        n = len(single_candidates)
        for i in range(n):
            clause1, supp1, _, _ = single_candidates[i]
            for j in range(i + 1, n):
                clause2, supp2, _, _ = single_candidates[j]

                # Do not conjoin two identical link types with contradicting targets
                if type(clause1) is type(clause2) and clause1.to_s_expr() == clause2.to_s_expr():
                    continue

                conjunction = [clause1, clause2]
                conjunction_matches = self.atomspace.query_conjunction(conjunction)

                # Filter distinct entities satisfying target_var
                distinct_witnesses: Dict[str, Tuple[Valuation, List[Atom]]] = {}
                for val, witnesses in conjunction_matches:
                    bound_val = val.get(target_var)
                    if bound_val and isinstance(bound_val, Node):
                        distinct_witnesses[bound_val.name] = (val, witnesses)

                conj_support = len(distinct_witnesses)
                if conj_support >= self.min_support:
                    surp = compute_surprisingness(conj_support, [supp1, supp2], universe_size)

                    formatted_witnesses = []
                    all_source_atoms: Set[str] = set()
                    for entity_name, (val, witnesses) in distinct_witnesses.items():
                        formatted_witnesses.append({
                            target_var_name: entity_name
                        })
                        for w in witnesses:
                            all_source_atoms.add(w.to_s_expr())

                    pattern = MinedPattern(
                        pattern_id=f"PAT-{pattern_counter:03d}",
                        clauses=conjunction,
                        variables=[target_var],
                        support=conj_support,
                        empirical_prob=surp.empirical_prob,
                        expected_interval=surp.expected_interval,
                        surprisingness_score=surp.score,
                        surprisingness_label=surp.label,
                        witnesses=formatted_witnesses,
                        source_atoms=sorted(list(all_source_atoms))
                    )
                    mined_patterns.append(pattern)
                    pattern_counter += 1

        return mined_patterns
