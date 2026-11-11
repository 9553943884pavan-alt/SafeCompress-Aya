# SafeCompress-Aya — PS11 Research Starter

**Team:** The_King · **Hackathon:** Lexsi 2026 Track 2b PS11 · **Model:** CohereLabs/tiny-aya-global

## Research question
Does circuit-guided mixed-precision quantization preserve Tiny Aya Global's refusal behavior better than matched-budget random protection, while keeping useful capabilities?

## Four conditions
M0: BF16 original; M1: uniform low-bit; M2: circuit-guided mixed precision; M3: random-protected matched precision budget.

## Important
This is a **research project scaffold**, not a finished training/quantization implementation. The experimental adapters and backend must be validated on Tiny Aya before GPU-scale runs. Record failures and negative results. Do not claim 4-bit storage from a config argument alone. Verify checkpoint save/reload and file sizes.

## Documents
- `docs/ROADMAP.md` day-by-day schedule and gates
- `docs/PROTOCOL.md` hypotheses, controls, evaluation, and split rules
- `docs/CHECKLIST.md` presubmit and prescale checks
- `docs/PAPERS.md` reading list
- `docs/RISK_REGISTER.md` failure modes and fallbacks
- `AGENTS.md` coding agent ground rules

## Suggested setup
Use Python 3.11+; consult current CircuitKIT TransformerLens/Transformers compatibility notes before pinning. Log in to Hugging Face securely to access gated Tiny Aya Global. Don't commit tokens or model weights.

## How to start
1. Complete Day 1 checkpoints in `docs/ROADMAP.md`.
2. Validate all three Lexsi tool imports.
3. Run a 10-20 item end-to-end smoke test.
4. Replace each `TODO` script with validated code and add tests.
5. Record exact package versions, model and data revision hashes.

## Stage A environment preflight

Install locally with `python -m pip install -e .`, then run `python scripts/preflight.py`.
The command does not download weights or test quantization. It writes an immutable
run directory under `results/raw/preflight/` with `report.json`, a package lock
snapshot, and SHA-256 checksums. A successful preflight is **not** a successful
model compatibility test. Stage B requires authorized model access and a real
BF16 generation plus uniform 4-bit save/reload/generation smoke test before
circuit discovery or evaluation. Keep model artifacts out of Git.

See official resources: https://hackathon.lexsi.ai/ and https://circuitkit.lexsi.ai/user-guide/applications/
