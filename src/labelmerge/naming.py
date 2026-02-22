from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import openai

from labelmerge.models import Group


@dataclass(frozen=True)
class ReviewRules:
    """Configurable heuristics for split-review suspicion detection."""

    antonym_pairs: tuple[tuple[str, str], ...] = (
        ("left", "right"),
        ("upper", "lower"),
        ("pre", "post"),
        ("anterior", "posterior"),
        ("proximal", "distal"),
        ("systolic", "diastolic"),
    )
    regex_conflicts: tuple[tuple[str, str], ...] = ((r"\bpt\b", r"\bptt\b"),)
    enable_subtype_suffix_check: bool = True
    enable_parenthetical_qualifier_check: bool = True
    canonical_style_hint: str = "concise, stable, and lowercase when possible"


@runtime_checkable
class ReviewPolicy(Protocol):
    """Extension point for LLM group review behavior."""

    def is_suspicious_group(self, group: Group) -> bool:
        """Return True when the group should be reviewed for possible splits."""
        ...

    def build_batch_prompt(self, batch: list[tuple[int, Group]], *, allow_split: bool) -> str:
        """Build the LLM prompt for a review/naming batch."""
        ...


@dataclass(frozen=True)
class _ReviewedSubgroup:
    canonical: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class _ReviewDecision:
    action: str  # "name" | "split"
    canonical: str | None = None
    subgroups: tuple[_ReviewedSubgroup, ...] = ()


class GenericReviewPolicy:
    """Default review policy with configurable generic heuristics."""

    def __init__(self, rules: ReviewRules | None = None) -> None:
        self.rules = rules or ReviewRules()

    def is_suspicious_group(self, group: Group) -> bool:
        return _is_suspicious_group(group, rules=self.rules)

    def build_batch_prompt(self, batch: list[tuple[int, Group]], *, allow_split: bool) -> str:
        return _build_batch_prompt(
            batch,
            allow_split=allow_split,
            canonical_style_hint=self.rules.canonical_style_hint,
        )


async def name_groups(
    groups: list[Group],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    api_key: str | None = None,
    batch_size: int = 8,
    review_profile: str = "generic",
    review_rules_path: str | Path | None = None,
    review_policy: str | None = None,
) -> list[Group]:
    """Name groups with LLM review that can also split bad merges.

    The model may return either:
      - action=name: one canonical for the full group
      - action=split: multiple reviewed subgroups with their own canonicals
    """
    if not groups:
        return []

    client = openai.AsyncOpenAI(api_key=api_key)
    policy = load_review_policy(
        review_profile=review_profile,
        review_rules_path=review_rules_path,
        review_policy=review_policy,
    )
    decisions: dict[int, _ReviewDecision] = {}

    suspicious_batches: list[list[tuple[int, Group]]] = []
    safe_batches: list[list[tuple[int, Group]]] = []
    current_suspicious: list[tuple[int, Group]] = []
    current_safe: list[tuple[int, Group]] = []

    for idx, group in enumerate(groups):
        bucket = current_suspicious if policy.is_suspicious_group(group) else current_safe
        bucket.append((idx, group))
        if len(bucket) >= batch_size:
            if bucket is current_suspicious:
                suspicious_batches.append(bucket.copy())
                current_suspicious.clear()
            else:
                safe_batches.append(bucket.copy())
                current_safe.clear()

    if current_suspicious:
        suspicious_batches.append(current_suspicious.copy())
    if current_safe:
        safe_batches.append(current_safe.copy())

    for batch in safe_batches:
        decisions.update(
            await _request_batch_decisions(
                client=client,
                batch=batch,
                model=model,
                temperature=temperature,
                allow_split=False,
                policy=policy,
            )
        )

    for batch in suspicious_batches:
        decisions.update(
            await _request_batch_decisions(
                client=client,
                batch=batch,
                model=model,
                temperature=temperature,
                allow_split=True,
                policy=policy,
            )
        )

    reviewed: list[Group] = []
    for idx, group in enumerate(groups):
        decision = decisions.get(idx)
        if decision is None:
            reviewed.append(group.model_copy(update={"canonical": _default_canonical(group)}))
            continue
        reviewed.extend(_apply_review_decision(group, decision))

    return reviewed


def _default_canonical(group: Group) -> str:
    return group.canonical or group.members[0].text


def load_review_rules(path: str | Path | None) -> ReviewRules:
    """Load optional heuristic overrides from a JSON file."""
    if path is None:
        return ReviewRules()

    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Review rules file must be a JSON object")

    defaults = ReviewRules()

    antonym_pairs = defaults.antonym_pairs
    if "antonym_pairs" in raw:
        value = raw["antonym_pairs"]
        if not isinstance(value, list):
            raise ValueError("review rules: antonym_pairs must be a list")
        parsed_pairs: list[tuple[str, str]] = []
        for item in value:
            if (
                not isinstance(item, list | tuple)
                or len(item) != 2
                or not all(isinstance(x, str) and x.strip() for x in item)
            ):
                raise ValueError("review rules: each antonym_pairs item must be [str, str]")
            parsed_pairs.append((item[0].strip().lower(), item[1].strip().lower()))
        antonym_pairs = tuple(parsed_pairs)

    regex_conflicts = defaults.regex_conflicts
    if "regex_conflicts" in raw:
        value = raw["regex_conflicts"]
        if not isinstance(value, list):
            raise ValueError("review rules: regex_conflicts must be a list")
        parsed_conflicts: list[tuple[str, str]] = []
        for item in value:
            if (
                not isinstance(item, list | tuple)
                or len(item) != 2
                or not all(isinstance(x, str) and x for x in item)
            ):
                raise ValueError("review rules: each regex_conflicts item must be [pattern_a, pattern_b]")
            parsed_conflicts.append((item[0], item[1]))
        regex_conflicts = tuple(parsed_conflicts)

    subtype = raw.get("enable_subtype_suffix_check", defaults.enable_subtype_suffix_check)
    qualifiers = raw.get(
        "enable_parenthetical_qualifier_check",
        defaults.enable_parenthetical_qualifier_check,
    )
    canonical_style_hint = raw.get("canonical_style_hint", defaults.canonical_style_hint)

    if not isinstance(subtype, bool):
        raise ValueError("review rules: enable_subtype_suffix_check must be boolean")
    if not isinstance(qualifiers, bool):
        raise ValueError("review rules: enable_parenthetical_qualifier_check must be boolean")
    if not isinstance(canonical_style_hint, str) or not canonical_style_hint.strip():
        raise ValueError("review rules: canonical_style_hint must be a non-empty string")

    return ReviewRules(
        antonym_pairs=antonym_pairs,
        regex_conflicts=regex_conflicts,
        enable_subtype_suffix_check=subtype,
        enable_parenthetical_qualifier_check=qualifiers,
        canonical_style_hint=canonical_style_hint.strip(),
    )


def load_review_policy(
    *,
    review_profile: str = "generic",
    review_rules_path: str | Path | None = None,
    review_policy: str | None = None,
) -> ReviewPolicy:
    """Load a review policy (built-in generic by default).

    `review_policy` supports `module.submodule:attr`, where `attr` is either:
    - a policy instance implementing `ReviewPolicy`, or
    - a zero-arg callable returning one.
    """
    if review_policy:
        module_name, sep, attr_name = review_policy.partition(":")
        if not sep or not module_name or not attr_name:
            raise ValueError("review policy must be in 'module.path:attr' form")
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        policy_obj = attr() if callable(attr) else attr
        if not isinstance(policy_obj, ReviewPolicy):
            raise ValueError("loaded review policy does not implement ReviewPolicy")
        return policy_obj

    if review_profile != "generic":
        raise ValueError(
            f"Unknown review profile: {review_profile!r}. "
            "Use 'generic' or provide --review-policy module:attr."
        )

    return GenericReviewPolicy(load_review_rules(review_rules_path))


def _normalize_text_for_heuristics(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _is_suspicious_group(group: Group, *, rules: ReviewRules | None = None) -> bool:
    config = rules or ReviewRules()
    if group.size <= 1:
        return False
    texts = [m.text for m in group.members]
    normalized = [_normalize_text_for_heuristics(t) for t in texts]
    token_sets = [set(t.split()) for t in normalized if t]

    for left, right in config.antonym_pairs:
        seen_left = any(left in toks for toks in token_sets)
        seen_right = any(right in toks for toks in token_sets)
        if seen_left and seen_right:
            return True

    joined = " ".join(normalized)
    for pat_a, pat_b in config.regex_conflicts:
        if re.search(pat_a, joined) and re.search(pat_b, joined):
            return True

    # Flag chains like NR2A/NR2B or receptor subtype suffixes sharing the same base.
    if config.enable_subtype_suffix_check:
        suffix_variants: dict[str, set[str]] = {}
        for text in normalized:
            for match in re.finditer(r"\b([a-z]+\d+)([a-z])\b", text):
                base, suffix = match.groups()
                suffix_variants.setdefault(base, set()).add(suffix)
        if any(len(suffixes) > 1 for suffixes in suffix_variants.values()):
            return True

    # Distinct parenthetical qualifiers inside a small group are often false merges.
    if config.enable_parenthetical_qualifier_check:
        qualifiers = {
            q.strip().lower()
            for raw in texts
            for q in re.findall(r"\(([^)]+)\)", raw)
            if q.strip()
        }
        if len(qualifiers) >= 2:
            return True

    return False


async def _request_batch_decisions(
    *,
    client: openai.AsyncOpenAI,
    batch: list[tuple[int, Group]],
    model: str,
    temperature: float,
    allow_split: bool,
    policy: ReviewPolicy,
) -> dict[int, _ReviewDecision]:
    prompt = policy.build_batch_prompt(batch, allow_split=allow_split)
    response = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    if content is None:
        return {idx: _fallback_name_decision(group) for idx, group in batch}
    return _parse_batch_response(content, batch, allow_split=allow_split)


def _build_batch_prompt(
    batch: list[tuple[int, Group]],
    *,
    allow_split: bool,
    canonical_style_hint: str = "concise, stable, and lowercase when possible",
) -> str:
    action_instructions = (
        "For each group, decide whether to keep as one concept (action=name) "
        "or split into distinct concepts (action=split)."
        if allow_split
        else "For each group, return action=name and a concise canonical label."
    )

    schema_text = (
        '[{"group_index": 0, "action": "name", "canonical": "..."}, ...]'
        if not allow_split
        else (
            '[{"group_index": 0, "action": "name", "canonical": "..."}, '
            '{"group_index": 1, "action": "split", "subgroups": '
            '[{"canonical": "...", "members": ["exact member text"]}]}]'
        )
    )

    lines: list[str] = [
        "You are reviewing semantic label groups for a controlled vocabulary.",
        action_instructions,
        "Return ONLY valid JSON (no markdown, no explanations).",
        "Use exact member text strings in split subgroups.",
        f"Canonical labels should be {canonical_style_hint}.",
        f"JSON schema shape: {schema_text}",
        "",
        "Groups:",
    ]

    for local_index, (_original_index, group) in enumerate(batch):
        lines.append(f"- group_index: {local_index}")
        for member in group.members:
            lines.append(f'  - {member.count}x "{member.text}"')

    return "\n".join(lines)


def _parse_batch_response(
    content: str,
    batch: list[tuple[int, Group]],
    *,
    allow_split: bool,
) -> dict[int, _ReviewDecision]:
    local_groups = {local_idx: group for local_idx, (_idx, group) in enumerate(batch)}
    original_indices = {local_idx: original_idx for local_idx, (original_idx, _g) in enumerate(batch)}

    parsed = _loads_json_relaxed(content)
    if isinstance(parsed, dict) and "results" in parsed:
        parsed = parsed["results"]

    if isinstance(parsed, str) and len(batch) == 1:
        parsed = [{"group_index": 0, "action": "name", "canonical": parsed}]

    if not isinstance(parsed, list):
        return {idx: _fallback_name_decision(group) for idx, group in batch}

    decisions: dict[int, _ReviewDecision] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        group_index = item.get("group_index")
        if not isinstance(group_index, int) or group_index not in local_groups:
            continue
        group = local_groups[group_index]
        decision = _parse_decision(item, group, allow_split=allow_split)
        decisions[original_indices[group_index]] = decision

    for idx, group in batch:
        decisions.setdefault(idx, _fallback_name_decision(group))

    return decisions


def _parse_decision(item: dict[str, object], group: Group, *, allow_split: bool) -> _ReviewDecision:
    action_raw = item.get("action")
    action = action_raw.strip().lower() if isinstance(action_raw, str) else ""

    if not action:
        if "subgroups" in item:
            action = "split"
        elif "canonical" in item:
            action = "name"

    if action == "split" and allow_split:
        subgroups = _parse_split_subgroups(item.get("subgroups"), group)
        if subgroups:
            return _ReviewDecision(action="split", subgroups=subgroups)

    canonical_raw = item.get("canonical")
    if isinstance(canonical_raw, str) and canonical_raw.strip():
        return _ReviewDecision(action="name", canonical=canonical_raw.strip())

    return _fallback_name_decision(group)


def _parse_split_subgroups(value: object, group: Group) -> tuple[_ReviewedSubgroup, ...] | None:
    if not isinstance(value, list):
        return None

    member_lookup = {m.text: m for m in group.members}
    seen: set[str] = set()
    parsed: list[_ReviewedSubgroup] = []

    for item in value:
        if not isinstance(item, dict):
            return None
        canonical = item.get("canonical")
        members = item.get("members")
        if not isinstance(canonical, str) or not canonical.strip():
            return None
        if not isinstance(members, list) or not members:
            return None

        subgroup_members: list[str] = []
        for member in members:
            if not isinstance(member, str):
                return None
            if member not in member_lookup or member in seen:
                return None
            seen.add(member)
            subgroup_members.append(member)

        parsed.append(
            _ReviewedSubgroup(canonical=canonical.strip(), members=tuple(subgroup_members))
        )

    if seen != set(member_lookup):
        return None
    if len(parsed) < 2:
        return None

    return tuple(parsed)


def _apply_review_decision(group: Group, decision: _ReviewDecision) -> list[Group]:
    if decision.action != "split" or not decision.subgroups:
        canonical = decision.canonical or _default_canonical(group)
        return [group.model_copy(update={"canonical": canonical})]

    member_lookup = {m.text: m for m in group.members}
    split_groups: list[Group] = []
    for subgroup in decision.subgroups:
        members = [member_lookup[text] for text in subgroup.members]
        members.sort(key=lambda m: m.count, reverse=True)
        split_groups.append(
            Group(
                group_id=group.group_id,
                members=members,
                canonical=subgroup.canonical,
                size=len(members),
                total_occurrences=sum(m.count for m in members),
            )
        )

    split_groups.sort(key=lambda g: (g.total_occurrences, g.size), reverse=True)
    return split_groups


def _fallback_name_decision(group: Group) -> _ReviewDecision:
    return _ReviewDecision(action="name", canonical=_default_canonical(group))


def _loads_json_relaxed(content: str) -> object:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
    if match is None:
        return text
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return text
