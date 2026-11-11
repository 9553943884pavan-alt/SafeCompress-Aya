# Experiment log template

| run_id | UTC time | git commit | model revision | data hash | algorithm | backend | protected layers | actual bytes | safety | overrefusal | ARC | peak VRAM | tok/s | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m0_baseline | | | | | BF16 | | | | | | | | | |
| m1_uniform | | | | | W4 uniform | | | | | | | | | |
| m2_guided | | | | | guided mixed | | | | | | | | | |
| m3_random | | | | | random mixed | | | | | | | | | |

## Stage A preflight, 2026-11-11

- Run ID: `20261111T073638Z_514d89556c46`; immutable report and SHA-256
  manifest: `results/raw/preflight/20261111T073638Z_514d89556c46/`.
- Python 3.12.10; Torch 2.6.0+cu124; Transformers 5.17.0. The local
  `nvidia-smi` reports RTX 3050 Laptop GPU, 6144 MiB, driver 610.88.
  A subprocess reports CUDA and BF16 support. These are environment checks,
  not model compatibility evidence.
- `accelerate`, `bitsandbytes`, CuratorKIT, CircuitKIT and AuditKIT are absent.
  Tiny Aya config is not cached; model access was not verified. No BF16 model
  generation or quantized save/reload/generation was attempted.
- Test runner failure: `python -m pytest -q` exits before collection because
  `pluggy` is missing from this Python environment. Run the standard-library
  smoke test with `PYTHONPATH=src python -m unittest discover -s tests`.
- Stage B remains gated until dependencies and authorized model weights are
  available. No safety, compression, or performance result exists yet.
