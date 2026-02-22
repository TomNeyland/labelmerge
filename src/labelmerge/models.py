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

    def canonical_values(self) -> list[str]:
        """Sorted unique canonical values, suitable for enum output."""
        return sorted(set(self.to_mapping().values()))

    def to_jsonschema(self) -> dict[str, object]:
        """JSON Schema fragment for a string enum."""
        return {"type": "string", "enum": self.canonical_values()}


class SweepResult(BaseModel):
    """Output of a threshold sweep."""

    thresholds: list[float]
    results: list[ThresholdStats]


class DiffMatch(BaseModel):
    """A value that matched an existing canonical above threshold."""

    value: str
    canonical: str
    similarity: float


class DiffUnmatched(BaseModel):
    """A value that did not meet the threshold for auto-mapping."""

    value: str
    best_match: str | None
    similarity: float


class DiffExact(BaseModel):
    """A value already present in the canonical vocabulary."""

    value: str
    canonical: str


class DiffResult(BaseModel):
    """Structured output of vocabulary diff mode."""

    matched: list[DiffMatch]
    unmatched: list[DiffUnmatched]
    exact: list[DiffExact]

    def to_mapping(self) -> dict[str, str]:
        """Mapping of auto-resolved values (exact + matched) to canonicals."""
        mapping: dict[str, str] = {}
        for item in self.exact:
            mapping[item.value] = item.canonical
        for item in self.matched:
            mapping[item.value] = item.canonical
        return mapping


class ApplyChange(BaseModel):
    """Aggregated normalization change summary."""

    from_value: str
    to: str
    occurrences: int
    files: int


class ApplyDryRunResult(BaseModel):
    """Structured dry-run summary for apply mode."""

    files_scanned: int
    values_checked: int
    would_change: int
    already_canonical: int
    unmapped: int
    changes: list[ApplyChange]
