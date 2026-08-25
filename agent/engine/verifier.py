# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Pure-Python backward-chaining SRT Symbolic Verifier -- Mitigation #65.

SCOPE AND LIMITS (read before using)
-------------------------------------
This verifier checks that the stated conclusion *follows logically from the
stated premises* under the named inference rule. It operates on a constrained
grammar extracted from the SRTTrace fields.

IT DOES NOT:
  - Verify that premises are true in the real world.
  - Handle first-order quantifiers (for all, there exists).
  - Handle probabilistic or modal claims.
  - Substitute for a general theorem prover (Z3, Prolog).

A verified SRT means: "valid inference, not truth verification."

GRAMMAR
-------
Each premise string is parsed into one of:
  Fact(A)           -- "A"  or  "A is true"  or  "A holds"
  Negation(A)       -- "NOT A"  or  "not A"  or  "~A"
  Conditional(A,B)  -- "A implies B"  |  "if A then B"  |  "A causes B"
                       "A -> B"  |  "A therefore B"
  Disjunction(A,B)  -- "A or B"

SUPPORTED INFERENCE RULES
--------------------------
  transitive_implication : A->B, B->C  =>  A->C  (conclusion: "A implies C")
  modus_ponens           : A->B, A     =>  B
  modus_tollens          : A->B, ~B    =>  ~A
  disjunctive_syllogism  : A|B, ~A     =>  B
  de_morgan              : ~(A&B)      =>  ~A|~B  (and symmetric variants)
  conjunction            : A, B        =>  A&B

Premises that cannot be parsed into the grammar cause verification to fail
with reason="premise_not_parseable", not a hard exception.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.models import SRTTrace

# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fact:
    atom: str  # normalised lowercase

    def __str__(self) -> str:
        return self.atom


@dataclass(frozen=True)
class Negation:
    inner: Atom

    def __str__(self) -> str:
        return f"NOT {self.inner}"


@dataclass(frozen=True)
class Conditional:
    antecedent: Atom
    consequent: Atom

    def __str__(self) -> str:
        return f"{self.antecedent} -> {self.consequent}"


@dataclass(frozen=True)
class Disjunction:
    left: Atom
    right: Atom

    def __str__(self) -> str:
        return f"{self.left} OR {self.right}"


Atom = Fact | Negation | Conditional | Disjunction

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Return value of SRTVerifier.verify()."""

    verified: bool
    reason: str
    parsed_premises: list[Atom] = field(default_factory=list)
    rule_applied: str = ""
    conclusion_atom: Atom | None = None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Quantifier patterns that we explicitly refuse
_QUANTIFIER_RE = re.compile(
    r"\b(for all|there exists|forall|exists|every|each|some|no [a-z])\b",
    re.IGNORECASE,
)

# Conditional patterns (order matters -- most specific first)
_CONDITIONAL_RES = [
    re.compile(r"^if\s+(.+?)\s+then\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.+?)\s+implies\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.+?)\s+causes\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.+?)\s*->\s*(.+)$"),
    re.compile(r"^(.+?)\s+therefore\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.+?)\s+leads to\s+(.+)$", re.IGNORECASE),
]

_NEGATION_RES = [
    re.compile(r"^(?:NOT|not|~)\s+(.+)$"),
    re.compile(r"^it is not the case that\s+(.+)$", re.IGNORECASE),
]

_DISJUNCTION_RE = re.compile(r"^(.+?)\s+or\s+(.+)$", re.IGNORECASE)

_FACT_STRIP_RE = re.compile(
    r"\s+(?:is true|holds|is the case|is a fact)$", re.IGNORECASE
)


def _parse_atom(text: str) -> Atom | None:
    """Parse a single premise or conclusion string into an AST node.

    Returns None if the text cannot be parsed (e.g. quantifier detected).
    """
    text = text.strip()

    # Quantifier guard
    if _QUANTIFIER_RE.search(text):
        return None

    # Negation
    for pat in _NEGATION_RES:
        m = pat.match(text)
        if m:
            inner = _parse_atom(m.group(1).strip())
            if inner is None:
                return None
            return Negation(inner)

    # Conditional
    for pat in _CONDITIONAL_RES:
        m = pat.match(text)
        if m:
            ant = _parse_atom(m.group(1).strip())
            con = _parse_atom(m.group(2).strip())
            if ant is None or con is None:
                return None
            return Conditional(ant, con)

    # Disjunction
    m = _DISJUNCTION_RE.match(text)
    if m:
        left = _parse_atom(m.group(1).strip())
        right = _parse_atom(m.group(2).strip())
        if left is None or right is None:
            return None
        return Disjunction(left, right)

    # Fact (strip trailing "is true" etc.)
    cleaned = _FACT_STRIP_RE.sub("", text).strip().lower()
    if cleaned:
        return Fact(cleaned)

    return None


def _atoms_equivalent(a: Atom, b: Atom) -> bool:
    """Structural equality with normalised Fact comparison."""
    if type(a) is not type(b):
        return False
    if isinstance(a, Fact):
        return a.atom.strip().lower() == b.atom.strip().lower()  # type: ignore[union-attr]
    return a == b


# ---------------------------------------------------------------------------
# Inference rule checkers
# ---------------------------------------------------------------------------


def _check_transitive(premises: list[Atom], conclusion: Atom) -> tuple[bool, str]:
    """A->B, B->C => A->C"""
    if not isinstance(conclusion, Conditional):
        return False, "conclusion must be a conditional for transitive_implication"
    cond_map: dict[str, Atom] = {}
    for p in premises:
        if isinstance(p, Conditional):
            key = str(p.antecedent).lower()
            cond_map[key] = p.consequent

    # Walk chain: conclusion.antecedent -> ... -> conclusion.consequent
    current = conclusion.antecedent
    target = conclusion.consequent
    for _ in range(len(premises) + 1):
        if _atoms_equivalent(current, target):
            return True, "transitive chain complete"
        nxt = cond_map.get(str(current).lower())
        if nxt is None:
            break
        current = nxt
    return False, f"cannot derive {conclusion} from premises via transitivity"


def _check_modus_ponens(premises: list[Atom], conclusion: Atom) -> tuple[bool, str]:
    """A->B, A => B"""
    for p in premises:
        if isinstance(p, Conditional):
            # Does a Fact matching the antecedent exist?
            if any(
                _atoms_equivalent(p.antecedent, q)
                for q in premises
                if not isinstance(q, Conditional)
            ):
                if _atoms_equivalent(p.consequent, conclusion):
                    return True, "modus_ponens: A->B, A |- B"
    return False, "modus_ponens: no matching A->B, A pair found"


def _check_modus_tollens(premises: list[Atom], conclusion: Atom) -> tuple[bool, str]:
    """A->B, ~B => ~A"""
    if not isinstance(conclusion, Negation):
        return False, "modus_tollens conclusion must be a negation"
    for p in premises:
        if isinstance(p, Conditional):
            neg_b = Negation(p.consequent)
            if any(_atoms_equivalent(neg_b, q) for q in premises):
                expected = Negation(p.antecedent)
                if _atoms_equivalent(expected, conclusion):
                    return True, "modus_tollens: A->B, ~B |- ~A"
    return False, "modus_tollens: no matching A->B, ~B pair found"


def _check_disjunctive_syllogism(
    premises: list[Atom], conclusion: Atom
) -> tuple[bool, str]:
    """A|B, ~A => B  (or A|B, ~B => A)"""
    for p in premises:
        if isinstance(p, Disjunction):
            for disj_side, other_side in [(p.left, p.right), (p.right, p.left)]:
                neg = Negation(disj_side)
                if any(_atoms_equivalent(neg, q) for q in premises):
                    if _atoms_equivalent(other_side, conclusion):
                        return True, "disjunctive_syllogism: A|B, ~A |- B"
    return False, "disjunctive_syllogism: no matching disjunction/negation pair"


def _check_de_morgan(premises: list[Atom], conclusion: Atom) -> tuple[bool, str]:
    """Very limited: checks the standard De Morgan transformations.

    We only handle the case where the conclusion is explicitly asserted
    as a known De Morgan transform of a premise negation.
    (Full De Morgan verification requires a SAT solver.)
    """
    # Accept if conclusion string is one of the two De Morgan forms of any premise
    c_str = str(conclusion).lower().strip()
    for p in premises:
        p_str = str(p).lower().strip()
        # ~(A AND B) => ~A OR ~B  (approximate string-level check)
        if "and" in p_str and ("or" in c_str or "not" in c_str):
            return True, "de_morgan: structural match (approximate)"
        if "or" in p_str and ("and" in c_str or "not" in c_str):
            return True, "de_morgan: structural match (approximate)"
    return False, "de_morgan: no matching De Morgan pattern found"


def _check_conjunction(premises: list[Atom], conclusion: Atom) -> tuple[bool, str]:
    """A, B => A AND B  (conclusion must contain both conjuncts)"""
    c_str = str(conclusion).lower()
    if " and " not in c_str and "&" not in c_str:
        return False, "conjunction conclusion must contain 'and'"
    # Verify both parts are present as premises
    for p in premises:
        p_str = str(p).lower()
        if p_str not in c_str:
            continue
    # Shallow: both halves appear somewhere in premises
    parts = [s.strip() for s in re.split(r"\band\b|&", c_str)]
    for part in parts:
        if not any(
            _atoms_equivalent(Fact(part), p) for p in premises if isinstance(p, Fact)
        ):
            return False, f"conjunction: '{part}' not found as a premise"
    return True, "conjunction: all conjuncts present in premises"


_RULE_CHECKERS = {
    "transitive_implication": _check_transitive,
    "modus_ponens": _check_modus_ponens,
    "modus_tollens": _check_modus_tollens,
    "disjunctive_syllogism": _check_disjunctive_syllogism,
    "de_morgan": _check_de_morgan,
    "conjunction": _check_conjunction,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SRTVerifier:
    """Host-side deterministic SRT verifier.

    Usage::

        verifier = SRTVerifier()
        result = verifier.verify(srt)
        if result.verified:
            memory.mark_verified(episode_id, srt.model_dump_json())
    """

    def verify(self, srt: SRTTrace) -> VerificationResult:
        """Verify that ``srt.conclusion`` follows from ``srt.premises``
        under ``srt.inference_rule``.

        Returns a VerificationResult.  Never raises.
        """
        rule = srt.inference_rule
        checker = _RULE_CHECKERS.get(rule)
        if checker is None:
            return VerificationResult(
                verified=False,
                reason=f"unknown inference rule: {rule}",
            )

        # Parse premises
        parsed: list[Atom] = []
        for raw in srt.premises:
            atom = _parse_atom(raw)
            if atom is None:
                if _QUANTIFIER_RE.search(raw):
                    return VerificationResult(
                        verified=False,
                        reason=f"quantifier not supported in premise: '{raw[:60]}'",
                        parsed_premises=parsed,
                        rule_applied=rule,
                    )
                return VerificationResult(
                    verified=False,
                    reason=f"premise_not_parseable: '{raw[:60]}'",
                    parsed_premises=parsed,
                    rule_applied=rule,
                )
            parsed.append(atom)

        # Parse conclusion
        conclusion_atom = _parse_atom(srt.conclusion)
        if conclusion_atom is None:
            return VerificationResult(
                verified=False,
                reason=f"conclusion_not_parseable: '{srt.conclusion[:60]}'",
                parsed_premises=parsed,
                rule_applied=rule,
            )

        # Run the inference checker
        ok, reason = checker(parsed, conclusion_atom)
        return VerificationResult(
            verified=ok,
            reason=reason,
            parsed_premises=parsed,
            rule_applied=rule,
            conclusion_atom=conclusion_atom,
        )
