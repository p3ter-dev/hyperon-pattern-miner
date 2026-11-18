"""
Core Hyperon AtomSpace Representation.
Provides typed Nodes, Links, and an indexed AtomSpace with pattern matching.
Faithfully mirrors OpenCog Hyperon's hypergraph atom model.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Iterator


class Atom:
    """Base class for all AtomSpace entities (Nodes and Links)."""
    @property
    def type_name(self) -> str:
        return self.__class__.__name__

    def is_node(self) -> bool:
        return isinstance(self, Node)

    def is_link(self) -> bool:
        return isinstance(self, Link)

    def is_variable(self) -> bool:
        return isinstance(self, VariableNode)

    def to_s_expr(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Node(Atom):
    """Atomic entity with an immutable string name."""
    name: str

    def to_s_expr(self) -> str:
        return f'({self.type_name} "{self.name}")'

    def __repr__(self) -> str:
        return self.to_s_expr()


class ConceptNode(Node):
    """Represents an entity, category, or concept."""
    pass


class PredicateNode(Node):
    """Represents a relation or property predicate."""
    pass


class VariableNode(Node):
    """Represents a logical variable in pattern mining (e.g., $X, $Y)."""
    def to_s_expr(self) -> str:
        return f"${self.name}"

    def __repr__(self) -> str:
        return f"${self.name}"


@dataclass(frozen=True)
class Link(Atom):
    """Hyperedge connecting an ordered list of Atoms (Nodes or nested Links)."""
    outgoing: Tuple[Atom, ...]

    def to_s_expr(self) -> str:
        children_str = " ".join(child.to_s_expr() for child in self.outgoing)
        return f"({self.type_name} {children_str})"

    def __repr__(self) -> str:
        return self.to_s_expr()


class ListLink(Link):
    """Ordered sequence of atoms, often argument lists for predicates."""
    pass


class InheritanceLink(Link):
    """Represents taxonomic inheritance: (InheritanceLink child parent)."""
    @property
    def child(self) -> Atom:
        return self.outgoing[0]

    @property
    def parent(self) -> Atom:
        return self.outgoing[1]


class EvaluationLink(Link):
    """Represents predicate evaluation: (EvaluationLink predicate (ListLink args...))."""
    @property
    def predicate(self) -> Atom:
        return self.outgoing[0]

    @property
    def arguments(self) -> Tuple[Atom, ...]:
        if len(self.outgoing) > 1 and isinstance(self.outgoing[1], ListLink):
            return self.outgoing[1].outgoing
        return self.outgoing[1:]


class AndLink(Link):
    """Conjunction of clauses in a pattern."""
    pass


# Valuation maps VariableNode to a concrete ground Atom
Valuation = Dict[VariableNode, Atom]


def substitute(atom: Atom, valuation: Valuation) -> Atom:
    """Applies variable substitution theta to an atom."""
    if isinstance(atom, VariableNode):
        return valuation.get(atom, atom)
    if isinstance(atom, Link):
        new_outgoing = tuple(substitute(child, valuation) for child in atom.outgoing)
        return atom.__class__(new_outgoing)
    return atom


def match_atoms(pattern: Atom, target: Atom, current_valuation: Valuation) -> Optional[Valuation]:
    """
    Unifies / matches a pattern atom against a concrete target atom.
    Returns an updated valuation dictionary if compatible, else None.
    """
    if isinstance(pattern, VariableNode):
        if pattern in current_valuation:
            return current_valuation if current_valuation[pattern] == target else None
        new_val = dict(current_valuation)
        new_val[pattern] = target
        return new_val

    # If pattern is a Node, types and names must match exactly
    if isinstance(pattern, Node):
        if isinstance(target, Node) and type(pattern) is type(target) and pattern.name == target.name:
            return current_valuation
        return None

    # If pattern is a Link, type and arity must match
    if isinstance(pattern, Link):
        if not isinstance(target, Link) or type(pattern) is not type(target):
            return None
        if len(pattern.outgoing) != len(target.outgoing):
            return None

        val = current_valuation
        for p_child, t_child in zip(pattern.outgoing, target.outgoing):
            res = match_atoms(p_child, t_child, val)
            if res is None:
                return None
            val = res
        return val

    return None


class AtomSpace:
    """
    Indexed knowledge store for Hyperon Atoms.
    Supports addition, query by type, and subgraph pattern matching.
    """
    def __init__(self):
        self._atoms: Set[Atom] = set()
        self._by_type: Dict[str, Set[Atom]] = {}

    def add(self, atom: Atom) -> Atom:
        """Adds an atom and recursively registers its sub-atoms."""
        if atom in self._atoms:
            return atom
        self._atoms.add(atom)
        t = atom.type_name
        if t not in self._by_type:
            self._by_type[t] = set()
        self._by_type[t].add(atom)

        if isinstance(atom, Link):
            for child in atom.outgoing:
                self.add(child)
        return atom

    def add_all(self, *atoms: Atom) -> None:
        for a in atoms:
            self.add(a)

    def size(self) -> int:
        return len(self._atoms)

    def get_by_type(self, type_name: str) -> Set[Atom]:
        return self._by_type.get(type_name, set())

    def all_atoms(self) -> Set[Atom]:
        return set(self._atoms)

    def query_clause(self, clause_pattern: Atom) -> List[Tuple[Valuation, Atom]]:
        """
        Queries a single clause pattern against all atoms matching its type.
        Returns pairs of (valuation, matching_ground_atom).
        """
        results: List[Tuple[Valuation, Atom]] = []
        target_atoms = self.get_by_type(clause_pattern.type_name)
        for target in target_atoms:
            val = match_atoms(clause_pattern, target, {})
            if val is not None:
                results.append((val, target))
        return results

    def query_conjunction(self, clauses: List[Atom]) -> List[Tuple[Valuation, List[Atom]]]:
        """
        Finds all variable valuations that simultaneously satisfy all clauses.
        Returns list of (valuation, list_of_ground_witness_atoms).
        """
        if not clauses:
            return []

        def _extend_match(clause_idx: int, current_val: Valuation, witnesses: List[Atom]) -> Iterator[Tuple[Valuation, List[Atom]]]:
            if clause_idx >= len(clauses):
                yield (current_val, list(witnesses))
                return

            clause = clauses[clause_idx]
            target_candidates = self.get_by_type(clause.type_name)
            for target in target_candidates:
                new_val = match_atoms(clause, target, current_val)
                if new_val is not None:
                    witnesses.append(target)
                    yield from _extend_match(clause_idx + 1, new_val, witnesses)
                    witnesses.pop()

        return list(_extend_match(0, {}, []))

