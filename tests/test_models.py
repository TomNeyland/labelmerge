from __future__ import annotations

from semdedup.models import Group, Member, Result


def test_result_to_mapping() -> None:
    result = Result(
        groups=[
            Group(
                group_id=0,
                members=[
                    Member(text="foo", count=3),
                    Member(text="Foo", count=2),
                    Member(text="FOO", count=1),
                ],
                size=3,
                total_occurrences=6,
            )
        ],
        singletons=[Member(text="bar", count=1)],
        threshold=0.85,
        model="test",
        n_input=4,
        n_grouped=3,
        n_singletons=1,
    )
    mapping = result.to_mapping()
    assert mapping["foo"] == "foo"
    assert mapping["Foo"] == "foo"
    assert mapping["FOO"] == "foo"
    assert mapping["bar"] == "bar"
    assert len(mapping) == 4


def test_result_to_mapping_with_canonical() -> None:
    result = Result(
        groups=[
            Group(
                group_id=0,
                members=[
                    Member(text="foo", count=3),
                    Member(text="Foo", count=2),
                ],
                canonical="canonical_foo",
                size=2,
                total_occurrences=5,
            )
        ],
        singletons=[],
        threshold=0.85,
        model="test",
        n_input=2,
        n_grouped=2,
        n_singletons=0,
    )
    mapping = result.to_mapping()
    assert mapping["foo"] == "canonical_foo"
    assert mapping["Foo"] == "canonical_foo"
