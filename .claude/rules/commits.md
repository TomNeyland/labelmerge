# Commit Convention

Use conventional commits for auto-changelog via git-cliff:

- `feat:` new feature
- `fix:` bug fix
- `refactor:` code restructuring
- `test:` adding/updating tests
- `docs:` documentation
- `ci:` CI/CD changes
- `chore:` maintenance

Format: `type(scope): description`

Examples:
- `feat(core): implement connected components grouping`
- `fix(cache): handle concurrent cache writes`
- `test(similarity): add hypothesis property tests`

Always include `Co-Authored-By: Claude <noreply@anthropic.com>` in commit messages.
