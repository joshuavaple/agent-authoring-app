# ADR-001: Single Shared Environment for App and UI

**Status:** Accepted (Technical Debt)
**Date:** 2026-06-04
**Tags:** tooling
**Decider:** [Joshua Le]

## Context
Early-stage development. Backend App (FastAPI) and frontend UI (Streamlit) are co-located in a single Python environment for development velocity. 

Proper separation requires a shared package strategy for various modules,like Pydantic schemas, and adds environment management overhead. These are not yet justified.

## Decision
Use a single `requirements.txt` / `environment.yml` at the project root covering both sets of services.

## Consequences
**Positive:**
- Faster onboarding, single `pip install -r requirements.txt`

**Negative (the debt):**
- Dependency conflicts between UI and backend will be silent until they break
- Blocks / complicates independent deployment of app vs UI - need to point to the common requirement files, increasing the size of both unnecessarily.
- Upgrade of any Streamlit dependencies risks breaking inference service and vice versa.