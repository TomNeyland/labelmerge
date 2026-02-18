from __future__ import annotations

from pydantic import BaseModel


class Member(BaseModel):
    """One text item in a group."""

    text: str
    count: int = 1


class Group(BaseModel):
    """A set of near-duplicate text items."""

    group_id: int
    members: list[Member]
    canonical: str | None = None
    size: int
    total_occurrences: int


class ThresholdStats(BaseModel):
    """Stats for a single threshold in a sweep."""

    threshold: float
    n_groups: int
    n_singletons: int
    max_group_size: int
    median_group_size: float
    pct_grouped: float


class Result(BaseModel):
    """Output of a dedup run."""

    groups: list[Group]
    singletons: list[Member]
    threshold: float
    model: str
    n_input: int
    n_grouped: int
    n_singletons: int

    def to_mapping(self) -> dict[str, str]:
        """Flat original -> canonical mapping.

        Uses most frequent member as canonical if not LLM-named.
        """
        mapping: dict[str, str] = {}
        for group in self.groups:
            canonical = group.canonical or group.members[0].text
            for member in group.members:
                mapping[member.text] = canonical
        for singleton in self.singletons:
            mapping[singleton.text] = singleton.text
        return mapping


class SweepResult(BaseModel):
    """Output of a threshold sweep."""

    thresholds: list[float]
    results: list[ThresholdStats]
