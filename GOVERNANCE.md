# Governance

Frame Quorum is maintainer-led. The current release maintainer is
[@appleweiping](https://github.com/appleweiping). Design discussion, compatibility changes, and
roadmap work happen in public issues and pull requests; vulnerabilities follow the private process
in [SECURITY.md](SECURITY.md).

Changes to scanning, metrics, selection constraints, benchmark definitions, or output schemas
require regression tests, a compatibility note, and an updated changelog. Evaluation claims must
retain all trials and must follow the evidence boundaries in
[experiments.md](docs/experiments.md).

A release requires green CI, regenerated deterministic examples, wheel and source-distribution
checks, an isolated installation smoke test, checksums, and build provenance. Maintainer roles and
decision rules may evolve through an explicit pull request as the contributor base grows.

