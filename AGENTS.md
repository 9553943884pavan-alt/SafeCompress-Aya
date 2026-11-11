# AGENTS.md — Rules for Codex and Antigravity

**Project:** SafeCompress-Aya · Lexsi AI Hackathon 2026 · Track 2b PS11  
**Research lead:** Team The_King (solo)  
**Model under study:** `CohereLabs/tiny-aya-global`  
**Aim:** Test whether circuit-guided mixed-precision compression preserves refusal behavior better than a **matched-budget random-protection** control, without unacceptable capability loss.

These instructions apply to all coding/review agents, scripts and notebooks in this repository. The research lead decides the experimental protocol, approves paid compute, and signs off on scientific claims. This document adapts useful engineering principles from the researcher's uploaded `RULES.md`, originally written for another ML challenge; its challenge-specific restrictions do **not** carry over.

## A. Mandatory reading and order of authority

1. Before modifying code, read `README.md`, `docs/PROTOCOL.md`, `docs/ROADMAP.md`, `docs/RESOURCE_PLAN.md`, `docs/RISK_REGISTER.md`, `docs/EXPERIMENT_LOG.md`, and relevant configs. **If a file is missing, report it; do not invent its contents.**
2. Follow the researcher's explicit instructions and the current experiment protocol. If this AGENTS.md conflicts with a verified contest rule or the agreed protocol, stop and request a decision; document the resolution.
3. Preserve the scientific setup unless explicitly approved: **M0** original BF16, **M1** uniform low-bit, **M2** circuit-guided mixed precision, **M3** random-protected mixed precision with a matched bit/parameter budget.
4. Prioritize a working, reproducible smallest end-to-end experiment over speculative optimization or new features.
5. Read implementation/API documentation for the exact installed revisions; never hallucinate a CircuitKIT/CuratorKIT/AuditKIT call, model support claim, bit-width capability or successful result.

## B. Research and evaluation integrity — non-negotiable

1. **No fabricated measurements.** Label values `MEASURED`, `USER-REPORTED`, `THEORETICAL`, or `PLANNED`. No measured Colab time, VRAM, costs, latency, checkpoint sizes or safety scores are known until logs are collected.
2. All scientific claims must trace to an immutable `run_id`, Git commit, config, model revision, data hash, hardware observed, raw predictions and logs. Store findings in `docs/EXPERIMENT_LOG.md`, `results/raw/`, and the measurement ledger in `docs/RESOURCE_PLAN.md`.
3. Separate `discovery`, `quantization_calibration`, `validation`, and `test` data. Never tune protected layers, prompt formatting, thresholds or backend choice against final test outcomes. Deduplicate and check near-paraphrase leakage. Never modify held-out labels or answer keys.
4. Same test prompts, model chat template, decoding, max new tokens and scoring rubric across M0–M3. For latency/memory: same GPU hardware, batch size, sequence-length distribution, warm-up and environment; run models sequentially in fresh processes when needed.
5. The primary hypothesis is **M2 versus M3** at matched effective precision and protected parameter/byte budget. Report both safety and helpfulness, plus size, peak memory and inference performance. Include M0 and M1 context.
6. Do not equate a refusal keyword, empty output, or gibberish with safety. Define unsafe compliance, over-refusal and invalid-output rate; review borderline cases manually. Save per-example annotations and uncertainty estimates when possible.
7. A circuit importance heatmap does not prove a causal refusal mechanism. Verify clean/corrupted contrasts, the scalar metric, and a suitable patching/ablation/random-control sanity check before strong causal claims.
8. **Quantization is not verified until `quantize → save → unload → reload → generate` succeeds** and actual on-disk bytes/effective bit formats are measured. Masks of zero-valued weights are not packed compression. Do not claim latency speedups without timed benchmarks.
9. A failed experiment or a null result is a valid reportable finding. Keep failures, limitations and negative evidence rather than optimizing toward a desired story.

## C. Efficient engineering rules adapted from RULES.md

1. **Small first:** Unit-test/CPU-test, then 4–8-item smoke test, then 16–32-item discovery, then the justified larger run. Do not prototype on an entire dataset or a full expensive attribution workload.
2. **Precompute and cache:** Tokenization, prompt templating, normalized prompt hashes, split manifests, calibration samples, circuit importance scores and benchmark predictions. Cache keys must include relevant model/tokenizer revision, preprocessing version, seed, data hash, objective and quantization config. Invalidate on changes; do not reuse stale caches silently.
3. **Use efficient formats:** Parquet/Arrow/Feather for processed tabular data, JSONL for append-only predictions/logs, and appropriate numeric dtypes. Avoid repeated CSV rereads. Never silently downcast IDs/hashes or metrics when precision loss would change results.
4. **Stream large files:** Use PyArrow/Polars or chunked loading and only needed columns when full loading is wasteful. Large intermediates should be written in resumable batches.
5. **Parallelize selectively:** Use bounded threads for independent I/O and process pools for suitable CPU-heavy tasks. Do not overparallelize Colab CPU or download multiple full gated checkpoints unnecessarily.
6. **Batch only when safe:** For **standard inference**, use `torch.inference_mode()`, reuse tokenizer batches, length-bucket where valid, and increase batch size after measuring headroom. **For EAP-IG/activation patching**, gradients and hooks must remain enabled and `batch_size=1` may be mandatory. The generic "never process a single sample" rule from the prior challenge does NOT apply to memory-sensitive causal experiments.
7. **Respect precision semantics:** BF16/FP16 are candidates only where supported and validated. Never cast every tensor to FP16 blindly or assume a configured `bits=4` guarantees actual four-bit packing.
8. **Vectorize when it improves efficiency:** Prefer NumPy/Pandas/Polars/tensor operations over DataFrame row loops, without replacing correct, inspectable circuit hooks with unsafe batching.
9. **Curation over quantity:** Inspect corrupt, duplicated, near-duplicated, poorly labeled and misclassified examples. Validate semantic contrast pairs; do not simply maximize dataset size.
10. **Dimensionality reduction is NOT a default requirement:** PCA/SVD from the old tabular-embedding task is not relevant to faithful full-model circuit discovery unless explicitly justified, benchmarked and approved. Do not compress activations in ways that invalidate causal conclusions.

## D. Compute-budget and hardware rules

1. Read `docs/RESOURCE_PLAN.md` before every Colab session. **Budget: 230 units total; planning ceiling 160.78; protected reserve 69.22.** These are planning numbers, not measured bills.
2. Known **USER-REPORTED** rates (units/h): CPU High RAM 0.26 (51GB RAM); L4 1.54 (24GB VRAM); A100 40GB 5.50; A100 80GB 6.77 (167GB RAM); G4 8.90 (95.4GB VRAM, 171GB RAM). Verify observed device/VRAM/RAM at runtime. Actual Colab balance is authoritative.
3. **Default:** local CPU for development/analysis; L4 for inference and tiny GPU tests; A100 40GB for difficult discovery; A100 80GB only for memory or measured runtime justification; G4 only with explicit research-lead approval. Do not assume a GPU is fastest or cheapest from its name.
4. **No autonomous expensive runs:** Agents may prepare commands, configs and small CPU tests, but must not start extended Colab GPU jobs, run large EAP-IG sweeps, allocate G4 or dip into reserve without an explicit user go-ahead. GPU time bills while the runtime is connected, including idle/debugging.
5. For each run log: `run_id`, UTC timestamps, Git SHA, dataset/model revisions, actual device name, GPU/CPU memory capacity, batch/sequence length, BF16/quantization settings, connected runtime hours, workload seconds, peak `torch.cuda` allocated and reserved GiB, disk bytes, tokens/sec, and result status. Record starting/ending Colab unit balance if visible. Log estimated units separately from actual Colab-reported consumption.
6. Integrate with the researcher's existing `resource_monitor.py`; its local code has **not** been provided for inspection. Preserve its public behavior while extending it to persist `results/raw/compute_usage.jsonl`, if instructed. Do not overwrite the script blindly.
7. Before heavy jobs: log environment, warm-up, reset memory stats, ensure output paths are persistent and confirm a maximum runtime/unit budget. After each completed stage: save artifacts and **disconnect unused GPUs**.
8. On OOM or backend failure: log one minimal reproduction, inspect hooks/activations/precision/batch settings, try a deliberate fix, then pause after two similar failures instead of repeatedly spending units. Escalate hardware only with evidence and approval.
9. If M2/M3 cannot be realized with comparable actual stored precision, report the limitation instead of disguising a mismatched comparison as a fair result.

## E. Repo structure, security and reproducibility

1. Put reusable library code in `src/safecompress_aya/`, entrypoints in `scripts/`, Colab exploration in `notebooks/`, configs in `configs/`, documentation in `docs/`, generated artifacts in `results/`, and final figures in `figures/`. Never make the only working method live inside a notebook.
2. Treat `data/raw/` as immutable. Keep tokenized/cached intermediates in `data/processed/` or `cache/`, with manifests and hashes. Checkpoint weights belong in ignored `models/` or approved artifact storage—not Git history.
3. Secure credentials: never commit HF tokens, Colab credentials, `.env`, API keys, or restricted data. Follow Tiny Aya and dataset licenses, gated-model access terms, and hackathon public-artifact requirements; don't assume redistribution permission.
4. Use version-pinned dependencies/lock files and `python -m pytest` or equivalent automated tests before merging. GPU integration tests should be explicitly marked and skipped on CPU-only machines.
5. Use isolated branches or worktrees for simultaneous Codex/Antigravity edits. Never let two agents rewrite the same implementation or experimental results concurrently.
6. Changes to a frozen protocol, split, metric, protected fraction, backend, or baseline require a recorded rationale and research-lead approval before final evaluation.
7. If a documented file or tool is absent, inspect the repo and state the blocker; never fake successful integration.

## F. Stage gates and ownership

- **Stage A:** clean environment + dataset provenance/splits + model/chat-template smoke test.
- **Stage B:** M0 BF16 + M1 low-bit verified by quantize/save/reload/generate. **No serious discovery spend before this gate.**
- **Stage C:** valid safety contrasts + EAP-IG scores + one meaningful causal/control check.
- **Stage D:** M2 guided + M3 random, truly packed and matched by effective size/precision.
- **Stage E:** fixed held-out evaluation, measured efficiency, statistical uncertainty and public reproducible report.

**Codex:** implement config-driven pipeline, profiling, integration tests and small reviewable commits.  
**Antigravity:** review scientific methodology, evaluation leakage, logging, fairness, precision/size verification, and reproducibility independently.  
**Research lead:** approve GPU spends, frozen experiment design and public claims.

### Required format for every agent handoff

Report: (1) goal and stage, (2) files touched and Git commit, (3) tests run with exact commands and pass/fail status, (4) hardware actually used, (5) measured peak memory/wall time and estimated/observed Colab cost **or `NOT MEASURED`**, (6) artifacts and hashes, (7) blockers/uncertainties, (8) next smallest validated step. **Do not claim GPU tests were performed if they were not.**

## G. Explicitly excluded rules from the uploaded Amazon challenge

The old `RULES.md` included Amazon entity-resolution-specific bans on external data, MIT/Apache-only model licensing, TSV submission schemas, `matched_entity_ids`, `country` open-set assumptions, product-image processing and price-cleaning. **These do not govern Lexsi PS11 and must not be copied as restrictions.** Instead, use the verified current Lexsi rules and each model/dataset license. The old global instruction to batch every GPU operation and its blanket PCA requirement are also intentionally narrowed as explained above.
