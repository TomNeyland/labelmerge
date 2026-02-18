# PyPI Package Name Decision

## Issue

The desired package name `labelmerge` is **already taken on PyPI** by a different project:
- **Existing Package**: [labelmerge 0.4.7](https://pypi.org/project/labelmerge/)
- **Description**: A BIDS app for merging multiple label maps in neuroimaging
- **Owner**: khanlab organization
- **Last Updated**: January 2024

## Options for Publishing

Since we cannot use `labelmerge` directly, here are the recommended alternatives:

### Option 1: `labelmerge-text` (Recommended)
- **Pros**: Clear differentiation from the neuroimaging package, indicates text processing
- **Cons**: Slightly longer name
- **Install**: `pip install labelmerge-text`
- **Import**: `from labelmerge import LabelMerge` (internal package name stays `labelmerge`)

### Option 2: `semantic-labelmerge`
- **Pros**: Emphasizes the semantic aspect, SEO-friendly
- **Cons**: Longer name
- **Install**: `pip install semantic-labelmerge`
- **Import**: `from labelmerge import LabelMerge`

### Option 3: `label-merge` (with hyphen)
- **Pros**: Short, readable
- **Cons**: May cause confusion with existing package, hyphens in package names can be problematic
- **Install**: `pip install label-merge`
- **Import**: `from labelmerge import LabelMerge`

### Option 4: Contact existing owner
- Try to reach out to khanlab to see if they would be willing to transfer or allow co-use
- **Unlikely to succeed**: The package is actively maintained

## Recommendation

**Use `labelmerge-text`** as the PyPI distribution name.

This provides clear differentiation while keeping the internal package name as `labelmerge`, which is what users will import in their code.

## Implementation

When ready to publish, update `pyproject.toml`:

```toml
[project]
name = "labelmerge-text"  # PyPI distribution name
# ... rest stays the same
```

The import will still be:
```python
from labelmerge import LabelMerge
```

## Repository Name

The GitHub repository will be renamed to `LabelMerge` (from `semdedup`), which is independent of the PyPI package name.
