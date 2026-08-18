"""
Uncertainty estimation for the agentic retrieval loop.

Two signals only, as scoped:

  1. n_matching_entity_ids : how many Entity IDs share the best-matching
     surface form. Two or more means the same name was observed at more than
     one location, so the query is genuinely under-specified.
  2. margin = top1_score - top2_score : how decisively the best candidate beats
     its nearest rival. Scores come from SpatialMemory.query, normalised within
     a single call, so the margin is comparable inside one decision.

confidence = margin_confidence * crowding_discount

Read plainly: how clearly the best Entity ID wins, discounted by how many
rivals it had to beat. One candidate means nothing to disambiguate, so
confidence is 1.0 and the agent answers normally.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Margin at which the winner is treated as fully decisive. Calibrated on A1_JAKE:
# an unqualified query ("I need the plate") lands near 0.03, while one naming the
# location ("the plate in the kitchen") lands near 0.22.
MARGIN_SATURATION = 0.20

# How much each extra rival beyond the second erodes confidence. Kept mild so a
# decisive margin still wins: an entity seen in five rooms is not ambiguous once
# the query names one of them.
CROWDING_WEIGHT = 0.05

# Fixed per the project scope; sweep only if time permits.
DEFAULT_CONFIDENCE_THRESHOLD = 0.75


@dataclass
class AmbiguityAssessment:
    """Outcome of one uncertainty check."""
    confidence: float
    n_matching_entity_ids: int
    margin: float
    surface_form: Optional[str] = None
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)

    @property
    def is_ambiguous(self) -> bool:
        return self.n_matching_entity_ids >= 2

    def to_dict(self) -> Dict[str, object]:
        return {
            "confidence": round(self.confidence, 4),
            "n_matching_entity_ids": self.n_matching_entity_ids,
            "margin": round(self.margin, 4),
            "surface_form": self.surface_form,
            "locations": list(self.locations),
            "candidate_entity_ids": [entity_id for entity_id, _ in self.candidates],
        }


def compute_confidence(n_matching_entity_ids: int, margin: float) -> float:
    """
    Combine the two signals into a confidence in [0, 1].

    Fewer than two matching Entity IDs means there is nothing to disambiguate,
    so confidence is maximal regardless of margin.
    """
    if n_matching_entity_ids < 2:
        return 1.0

    margin_confidence = min(max(margin, 0.0) / MARGIN_SATURATION, 1.0)
    crowding_discount = 1.0 / (1.0 + CROWDING_WEIGHT * (n_matching_entity_ids - 2))
    return margin_confidence * crowding_discount


def assess(candidates: Sequence[Tuple[str, float]],
           locations: Sequence[str],
           surface_form: Optional[str] = None) -> AmbiguityAssessment:
    """Build an assessment from ranked same-surface-form candidates."""
    n = len(candidates)
    margin = (candidates[0][1] - candidates[1][1]) if n >= 2 else 1.0
    return AmbiguityAssessment(
        confidence=compute_confidence(n, margin),
        n_matching_entity_ids=n,
        margin=margin,
        surface_form=surface_form,
        candidates=list(candidates),
        locations=list(locations),
    )


def should_abstain(assessment: AmbiguityAssessment,
                   threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> bool:
    """
    Abstain only when confidence is low AND more than one Entity ID matches.

    Both conditions are required. Low confidence with a single candidate means
    a weak lexical match, not an ambiguity a user could resolve, and asking
    about it would be an over-trigger.
    """
    return assessment.confidence < threshold and assessment.is_ambiguous


def _readable(location_label: str) -> str:
    return location_label.replace("_", " ")


def format_disambiguation_question(assessment: AmbiguityAssessment) -> str:
    """
    Phrase the question around the distinguishing attribute, which is location.

    Returns a question naming each candidate place, so the answer identifies one
    Entity ID rather than merely confirming the object.
    """
    name = assessment.surface_form or "that item"
    places = [_readable(label) for label in assessment.locations]

    if len(places) < 2:
        return f"Which {name} do you mean?"
    if len(places) == 2:
        options = f"the one in the {places[0]} or the one in the {places[1]}"
    else:
        head = ", ".join(f"the {place}" for place in places[:-1])
        options = f"the one in {head}, or the one in the {places[-1]}"
    return f"Which {name} do you mean, {options}?"
