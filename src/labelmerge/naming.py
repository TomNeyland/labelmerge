from __future__ import annotations

import openai

from labelmerge.models import Group


async def name_groups(
    groups: list[Group],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    api_key: str | None = None,
) -> list[Group]:
    """Name each group by asking an LLM to pick a canonical label.

    Sends each group's members (with counts) to the model and asks for
    a single canonical name. Returns new Group objects with canonical set.
    """
    client = openai.AsyncOpenAI(api_key=api_key)
    named: list[Group] = []

    for group in groups:
        member_lines = "\n".join(f"  {m.count}x  {m.text}" for m in group.members)
        prompt = (
            "Given these near-duplicate text labels with their frequencies, "
            "pick the single best canonical name. Respond with ONLY the canonical name, "
            "nothing else.\n\n"
            f"Labels:\n{member_lines}"
        )

        response = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        canonical = response.choices[0].message.content
        assert canonical is not None
        named.append(group.model_copy(update={"canonical": canonical.strip()}))

    return named
