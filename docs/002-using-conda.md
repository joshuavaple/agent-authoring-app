# ADR-002: Using conda as dependency management tool

**Status:** Accepted (Revisit Candidate)  
**Date:** 2026-06-04
**Tags:** tooling, dependency-management
**Decider:** [Joshua Le]

## Context
Early-stage development. A dependency management tool is needed for the Python environment. conda was chosen for its ability to manage both Python packages, and its familiarity.

## Decision
Use conda with an `environment.yml` at the project root for environment setup.

## Consequences
**Positive:**
- Handles non-PyPI dependencies (system libs, CUDA, etc.) natively
- Familiar tooling; straightforward onboarding with `conda env create`

**Negative:**
- conda environments can be slow to resolve and recreate
- Less aligned with modern Python packaging standards (PEP 517/518)
- `environment.yml` is less portable than a `pyproject.toml`-based setup
- `uv` is increasingly popular, seen in various Databricks examples in validating the model service by recreating its inferencing environment.

## Alternatives Not Yet Explored
- A `pyproject.toml`-based setup with a modern tool like `uv` or `hatch` could unify dependency management in a PEP-standard way and may offer faster installs and better portability. Not evaluated — worth revisiting if non-PyPI dependency needs remain minimal.
