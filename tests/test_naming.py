from __future__ import annotations

import pytest

from labelmerge.core import LabelMerge
from labelmerge.models import Group, Member, Result
from labelmerge.naming import (
    GenericReviewPolicy,
    _ReviewDecision,
    _ReviewedSubgroup,
    _apply_review_decision,
    _is_suspicious_group,
    _parse_split_subgroups,
    load_review_policy,
    load_review_rules,
)


def _make_group(*texts: tuple[str, int], group_id: int = 0) -> Group:
    members = [Member(text=text, count=count) for text, count in texts]
    return Group(
        group_id=group_id,
        members=members,
        size=len(members),
        total_occurrences=sum(m.count for m in members),
    )


class _AlwaysSafePolicy:
    def is_suspicious_group(self, group: Group) -> bool:
        return False

    def build_batch_prompt(self, batch, *, allow_split: bool) -> str:
        return "[]"


def _make_test_policy() -> _AlwaysSafePolicy:
    return _AlwaysSafePolicy()


def test_is_suspicious_group_detects_left_right() -> None:
    group = _make_group(("grip strength (left)", 3), ("grip strength (right)", 2))
    assert _is_suspicious_group(group) is True


def test_load_review_rules_customizes_antonym_pairs(tmp_path) -> None:
    rules_file = tmp_path / "review_rules.json"
    rules_file.write_text(
        """
        {
          "antonym_pairs": [["inbound", "outbound"]],
          "regex_conflicts": [],
          "enable_subtype_suffix_check": false,
          "enable_parenthetical_qualifier_check": false
        }
        """
    )
    rules = load_review_rules(rules_file)
    policy = GenericReviewPolicy(rules)
    group = _make_group(("shipment route inbound", 2), ("shipment route outbound", 2))

    assert policy.is_suspicious_group(group) is True


def test_load_review_policy_supports_plugin_factory() -> None:
    policy = load_review_policy(review_policy="tests.test_naming:_make_test_policy")

    assert isinstance(policy, _AlwaysSafePolicy)


def test_parse_split_subgroups_requires_exact_partition() -> None:
    group = _make_group(("PT", 4), ("PTT", 3))
    valid = _parse_split_subgroups(
        [
            {"canonical": "prothrombin time", "members": ["PT"]},
            {"canonical": "partial thromboplastin time", "members": ["PTT"]},
        ],
        group,
    )
    invalid = _parse_split_subgroups(
        [{"canonical": "prothrombin time", "members": ["PT"]}],
        group,
    )

    assert valid is not None
    assert len(valid) == 2
    assert invalid is None


def test_apply_review_decision_splits_group_and_preserves_counts() -> None:
    group = _make_group(("Total Cholesterol", 5), ("LDL-C", 2), ("LDL cholesterol", 4))
    decision = _ReviewDecision(
        action="split",
        subgroups=(
            _ReviewedSubgroup(canonical="total cholesterol", members=("Total Cholesterol",)),
            _ReviewedSubgroup(canonical="ldl cholesterol", members=("LDL cholesterol", "LDL-C")),
        ),
    )

    split = _apply_review_decision(group, decision)

    assert len(split) == 2
    assert split[0].canonical in {"total cholesterol", "ldl cholesterol"}
    ldl_group = next(g for g in split if g.canonical == "ldl cholesterol")
    assert ldl_group.size == 2
    assert ldl_group.total_occurrences == 6


@pytest.mark.asyncio
async def test_core_name_groups_rebuilds_stats_after_split(monkeypatch) -> None:
    result = Result(
        groups=[
            _make_group(("grip strength (left)", 3), ("grip strength (right)", 2), group_id=0),
        ],
        singletons=[Member(text="sleep quality", count=1)],
        threshold=0.9,
        model="test",
        n_input=3,
        n_grouped=2,
        n_singletons=1,
    )

    async def fake_name_groups(*args, **kwargs):
        return [
            Group(
                group_id=0,
                members=[Member(text="grip strength (left)", count=3)],
                canonical="grip strength (left hand)",
                size=1,
                total_occurrences=3,
            ),
            Group(
                group_id=0,
                members=[Member(text="grip strength (right)", count=2)],
                canonical="grip strength (right hand)",
                size=1,
                total_occurrences=2,
            ),
        ]

    monkeypatch.setattr("labelmerge.core.name_groups", fake_name_groups)

    lm = LabelMerge.__new__(LabelMerge)  # avoid requiring embedder init for this unit test
    reviewed = await LabelMerge.name_groups(lm, result, model="fake")

    assert len(reviewed.groups) == 2
    assert reviewed.groups[0].group_id == 0
    assert reviewed.groups[1].group_id == 1
    assert reviewed.n_grouped == 0
    assert reviewed.n_singletons == 3
    assert reviewed.to_mapping()["grip strength (left)"] == "grip strength (left hand)"
