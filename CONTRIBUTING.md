# Contributing

Thank you for improving Frame Quorum. Small, focused changes with evidence are easiest to review.

## Set up

1. Use Python 3.11 or newer.
2. Create and activate a virtual environment.
3. Run `python -m pip install -e ".[dev]"`.
4. Run `python -m pytest --cov`, `python -m ruff check .`, and `python -m ruff format --check .`.

Tests must be offline and deterministic. Prefer small synthetic Pillow images over committing photographs or large
fixtures. A behavior change to selection needs a regression test and a documentation update describing its effect
on existing manifests.

## Pull requests

- Explain the user-visible problem, not only the implementation.
- Keep generated artifacts out of commits except the documented example in `examples/output`.
- Do not change selection weights silently. Explain policy changes and include before/after manifests.
- Preserve backward compatibility within the `1.x` manifest schema or propose a schema-version change.
- Confirm that you have rights to all code, images, and other material in the contribution.

Use of development tools, including AI assistance, does not transfer responsibility: review, understand, test, and
disclose materially generated contributions according to the hosting platform's current policies.

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
