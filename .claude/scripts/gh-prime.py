#!/usr/bin/env python3
"""GitHub Issues session context for Claude Code hooks.

Fetches open issues via GraphQL and renders structured context
with nested epics, dependencies, and priorities.
"""

import json
import subprocess
import sys


def gh_graphql(query: str) -> dict:
    result = subprocess.run(
        ["gh", "api", "graphql", "-f", f"query={query}"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"_GraphQL error: {result.stderr.strip()}_", file=sys.stderr)
        return {}
    return json.loads(result.stdout)


def get_repo() -> str:
    result = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def fetch_issues(owner: str, name: str) -> list[dict]:
    data = gh_graphql(f"""
    {{
      repository(owner: "{owner}", name: "{name}") {{
        issues(states: OPEN, first: 100, orderBy: {{field: CREATED_AT, direction: ASC}}) {{
          nodes {{
            number
            title
            state
            labels(first: 20) {{
              nodes {{ name }}
            }}
            parent {{
              number
              title
              state
            }}
            subIssues(first: 100) {{
              nodes {{
                number
                title
                state
                labels(first: 20) {{
                  nodes {{ name }}
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """)
    if not data:
        return []
    return data["data"]["repository"]["issues"]["nodes"]


def get_priority(issue: dict) -> int:
    for label in issue["labels"]["nodes"]:
        name = label["name"]
        if name.startswith("P") and len(name) == 2 and name[1].isdigit():
            return int(name[1])
    return 5


def get_labels(issue: dict) -> list[str]:
    return [label["name"] for label in issue["labels"]["nodes"]]


def priority_badge(p: int) -> str:
    return {0: "P0!", 1: "P1", 2: "P2", 3: "P3", 4: "P4"}.get(p, "")


def render_issue_line(issue: dict, indent: int = 0) -> str:
    prefix = "  " * indent + ("|- " if indent > 0 else "")
    labels = get_labels(issue)
    p = get_priority(issue)
    p_str = f"[{priority_badge(p)}]" if p < 5 else ""
    type_labels = [lb for lb in labels if lb not in {f"P{i}" for i in range(5)} and lb != "epic"]
    type_str = f" ({', '.join(type_labels)})" if type_labels else ""
    closed = " ~CLOSED~" if issue.get("state") == "CLOSED" else ""
    return f"{prefix}#{issue['number']} {p_str}{type_str} {issue['title']}{closed}"


def render_tree(issues: list[dict]) -> str:
    lines = []
    child_numbers = set()
    for issue in issues:
        for sub in issue.get("subIssues", {}).get("nodes", []):
            child_numbers.add(sub["number"])
        if issue.get("parent"):
            child_numbers.add(issue["number"])

    def has_subs(iss: dict) -> bool:
        return bool(iss.get("subIssues", {}).get("nodes", []))

    epics = [i for i in issues if has_subs(i) and i["number"] not in child_numbers]
    epics.sort(key=get_priority)
    standalone = [i for i in issues if i["number"] not in child_numbers and not has_subs(i)]
    standalone.sort(key=get_priority)

    if epics:
        lines.append("### Epics")
        lines.append("")
        for epic in epics:
            lines.append(render_issue_line(epic))
            subs = epic["subIssues"]["nodes"]
            open_subs = sorted([s for s in subs if s.get("state") != "CLOSED"], key=get_priority)
            closed_subs = [s for s in subs if s.get("state") == "CLOSED"]
            for sub in open_subs:
                lines.append(render_issue_line(sub, indent=1))
            if closed_subs:
                lines.append(f"  |- ... {len(closed_subs)} closed")
            lines.append("")

    if standalone:
        lines.append("### Standalone Issues")
        lines.append("")
        for issue in standalone:
            lines.append(render_issue_line(issue))
        lines.append("")

    return "\n".join(lines)


def render_by_priority(issues: list[dict]) -> str:
    child_numbers = set()
    for issue in issues:
        for sub in issue.get("subIssues", {}).get("nodes", []):
            child_numbers.add(sub["number"])
        if issue.get("parent"):
            child_numbers.add(issue["number"])

    top_level = [i for i in issues if i["number"] not in child_numbers]
    counts: dict[str, int] = {}
    for issue in top_level:
        p = get_priority(issue)
        key = priority_badge(p) if p < 5 else "untagged"
        counts[key] = counts.get(key, 0) + 1

    total = len(issues)
    parts = [f"{v} {k}" for k, v in sorted(counts.items())]
    return f"**{total} open** ({', '.join(parts)})"


def main() -> None:
    repo = get_repo()
    if not repo:
        print("_Could not determine repo (offline or not a git repo)_")
        return
    owner, name = repo.split("/")
    link_script = "~/.claude/skills/gh-issues/scripts/gh-link.sh"

    issues = fetch_issues(owner, name)

    print(f"""# GitHub Issues Workflow Context

> Auto-loaded on session start. {render_by_priority(issues)}

## Session Close Protocol

Before saying "done" — work is NOT done until pushed:
```
[ ] git status && git add <files> && git commit && git push
[ ] gh issue close <numbers> --repo {repo}
```

## Commands

| Action | Command |
|--------|---------|
| List open | `gh issue list --repo {repo} --state open` |
| By priority | `gh issue list --repo {repo} --label P1` |
| View detail | `gh issue view <#> --repo {repo}` |
| Create | `gh issue create --repo {repo} --title "..." --label "task,P2" --body "..."` |
| Close | `gh issue close <#> --repo {repo}` |
| Add sub-issue | `{link_script} add-sub {repo} <parent#> <child#>` |
| Blocked by | `{link_script} add-blocked-by {repo} <blocked#> <blocker#>` |
| Batch sub | `{link_script} batch-sub {repo} <parent#> <c1> <c2> ...` |
| Batch blocked | `{link_script} batch-blocked-by {repo} <blocked#> <b1> <b2> ...` |
| Show tree | `{link_script} show-relations {repo} <#>` |

**Labels** — type: `epic task feature bug refactor` / priority: `P0 P1 P2 P3 P4`

## Open Issues
""")

    if issues:
        print(render_tree(issues))
    else:
        print("_No open issues (or offline)_")


if __name__ == "__main__":
    main()
