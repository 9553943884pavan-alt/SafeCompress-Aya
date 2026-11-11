# SafeCompress-Aya — Compute Resource Plan and Measurement Ledger

**Project:** Lexsi AI Hackathon 2026 · Track 2b / PS11 · Team The_King  
**Model:** `CohereLabs/tiny-aya-global` (~3.35B parameters)  
**Research objective:** At a matched precision/storage budget, test whether circuit-guided mixed precision better preserves refusal behavior than random layer protection.  
**Planning period:** November 11–14, 2026; reserve November 15 for submission validation.  
**Total Colab balance for planning:** **230 compute units** (user-provided).  
**Authority:** The Colab account's live balance and actual usage take precedence over this estimate.

> **Evidence labels:** `USER-REPORTED` = configuration or rate supplied by the researcher; `THEORETICAL` = derived estimate; `PLANNED` = proposed allocation; `MEASURED` = recorded from an actual run. **No GPU profiling logs, experiment durations, or actual billed units have been supplied yet.** Therefore no benchmark results in this document are labeled `MEASURED`. Do not infer measured performance from advertised VRAM or hourly rates.

## 1. Hardware and rates (USER-REPORTED)

| Colab runtime | GPU VRAM | System RAM | Units per hour | 1-hour cost | Spending 230 units solely here gives (THEORETICAL) |
|---|---:|---:|---:|---:|---:|
| CPU High RAM | N/A | 51 GB | 0.26 | 0.26 | 884.62 h |
| L4 | 24 GB | Not reported | 1.54 | 1.54 | 149.35 h |
| A100 40GB | 40 GB | Not reported | 5.50 | 5.50 | 41.82 h |
| A100 80GB | 80 GB | 167 GB | 6.77 | 6.77 | 33.97 h |
| G4 (Colab label; actual GPU SKU unverified) | 95.4 GB | 171 GB | 8.90 | 8.90 | 25.84 h |

The name `G4` does **not** identify a verified GPU architecture. Before using it, log `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv` and `torch.cuda.get_device_name(0)`.

**Accounting formula:** `estimated_units = elapsed_connected_hours × user_reported_units_per_hour`. Count **the entire paid runtime connection**, including waiting, debugging, data transfers, setup, and idle time—not merely CUDA kernel time. Colab billing/remaining balance is the authoritative record. `wall_time_seconds` and GPU execution time are separate quantities.

### Weight-storage reference (THEORETICAL, *not measured VRAM*)

Assuming 3.35 billion weights and weight-only storage: BF16 = ~6.70 GB decimal, 8-bit = ~3.35 GB, 4-bit = ~1.675 GB. Quantization metadata, protected BF16 layers, unquantized components, KV cache, activations, temporary tensors, and runtime overhead make real checkpoint bytes and GPU memory larger. Theoretically smaller weights do not prove a packed checkpoint or faster inference.

## 2. Budget policy and four-day caps (PLANNED)

**Spending ceiling for the main work:** 160.78 units. **Protected reserve:** 69.22 units (30.1%). The entries below are *caps/allowances*, not targets or a direction to leave runtimes running. Some A100 40GB and 80GB allowances cover alternative escalation paths and may remain entirely unused.

| Day / stage | CPU High RAM | L4 | A100 40GB | A100 80GB | Maximum planned units |
|---|---:|---:|---:|---:|---:|
| Day 1: setup, data, M0/M1 baseline + export/reload | 3 h | 7 h | 1 h | 0 h | **17.06** |
| Day 2: circuit discovery, patching sanity check | 2 h | 1 h | 7 h | 3 h | **60.87** |
| Day 3: guided & random mixed-precision M2/M3 | 2 h | 6 h | 4 h | 3 h | **52.07** |
| Day 4: frozen evaluations, figures, publication | 5 h | 12 h | 2 h | 0 h | **30.78** |
| **Total maximum** | **12 h** | **26 h** | **14 h** | **6 h** | **160.78** |

Budget check: `12×0.26 + 26×1.54 + 14×5.50 + 6×6.77 = 160.78`; `230 − 160.78 = 69.22`.

**Reserve allocation (PLANNED):** 25 units for failed/repeat experiments; 20 for compatibility/OOM; 15 for additional validation; 9.22 untouched. The reserve is **not** preapproved for agent-initiated experiments. Every spending decision must respect remaining real Colab units and the nearest research deadline.

### When to upgrade: break-even throughput ratios

For equal work, a faster GPU is *cheaper in units* only if its observed speedup exceeds its hourly rate ratio:

| Comparison | Required measured throughput ratio |
|---|---:|
| A100 40GB vs L4 | >3.57× |
| A100 80GB vs L4 | >4.40× |
| A100 80GB vs A100 40GB | >1.23× |
| G4 vs A100 80GB | >1.31× |

These are **mathematical cost thresholds, not measured throughput claims**. A more expensive GPU can still be the right choice if only it fits in memory or if the deadline makes wall-clock savings more valuable.

## 3. Runtime selection by research stage

| Stage | Lowest-risk starting runtime | Escalation trigger | Completion evidence |
|---|---|---|---|
| Data download, normalization, split hashes, manifests | Local CPU; CPU High RAM only when needed | Local RAM/IO insufficient | Disjoint split manifest and deterministic hashes |
| M0 original BF16 model: generate and benchmark | L4 | OOM or unacceptably slow measured throughput | Predictions, prompt config, peak VRAM, elapsed time |
| M1 uniform quantization | L4 for smoke test | Model/backend quantization memory actually exceeds L4 | Quantize → save → unload → reload → coherent generation; actual disk bytes |
| EAP-IG API smoke test (4–8 pairs) | L4 | OOM or validated timeout | Finite nonempty circuit scores; exact objective; run log |
| Discovery 16–32 then 64–128 validated pairs | L4 trial; A100 40GB main | Evidence of OOM or throughput deadline; A100 80GB if needed | Scored circuit, candidate layer ranking, control/patch check |
| M2 guided mixed precision and M3 random matched protection | L4 or A100 40GB | Backend export OOM; A100 80GB for staging | Effective bit allocation, matched protected parameter bytes, reload tests |
| Safety + ARC evaluation, M0–M3 sequentially | L4 | Per-model peak VRAM/throughput insufficient | Saved per-item outputs, no test leakage, matched decoding |
| Result aggregation, bootstrap, paper and plots | Local CPU | Larger than local RAM/needs CPU high RAM | Tables/figures reproducible from saved outputs |
| G4 | **Do not schedule by default** | A100 80GB OOM **and** verified G4 architecture supports workload | Explicit researcher approval and measured justification |

**Memory guardrails (PLANNED, not observed):** use an early-warning threshold of about 80–85% of VRAM (L4 ~19–20 GB, A100 40 ~32–34 GB, A100 80 ~64–68 GB, G4 ~76–81 GB). This is a planning cushion, not proof the next allocation will fit. Before upgrading, inspect peak allocated/reserved memory, GPU free memory, CPU RAM pressure, and whether the implementation accidentally retains previous models/graphs.

## 4. Four-day execution with go/no-go conditions

### Day 1 — Establish the foundation (November 11)

1. On local CPU, pin environment and dataset revisions; create split IDs/hashes; validate no discovery/calibration/validation/test leakage. Use Parquet or Arrow for processed data.
2. Start L4 only when model assets and baseline script are ready. Confirm BF16 Tiny Aya uses the right tokenizer/chat formatting; cache M0 predictions.
3. Test quantization with **the selected real backend**, not an unverified `bits=4` configuration. Save and reload the checkpoint in a fresh process; inspect effective dtype/packing and serialized file bytes.
4. Benchmark 5–20 representative items first. Expand baseline only if export/reload is successful.
5. **Go/no-go:** Do not spend heavily on circuit discovery if no feasible quantize/export/reload path exists. Save errors, dependencies and logs as a reproducible compatibility result.

### Day 2 — Safety-circuit discovery (November 12)

1. Fix and log the clean/corrupted contrast pairs, objective function, and final discovery-set hash.
2. L4: 4–8 pair API test, then 16–32 pair attribution if feasible. For EAP-IG start `batch_size=1`, `ig_steps=3`, BF16, modest sequence lengths.
3. A100 40GB only if needed: expand to 64–128 pairs. A100 80GB is an OOM fallback, not an automatic second run.
4. Cache importance scores and component/layer rankings using model revision + dataset hash + objective + code commit + method settings as the cache key. Do not cache all activations indefinitely: they can be huge.
5. Run at least one control (random components or matched causal patch) to test whether the selected features matter beyond correlation.
6. **Go/no-go:** No validated refusal contrast or causal signal → do not present a heatmap as an established safety circuit.

### Day 3 — Mixed-precision models (November 13)

1. Produce a protected layer/module map from frozen discovery scores (candidate protected fraction 20%; tune only on validation).
2. Build M2 circuit-guided and M3 matched random protected with the **same backend and near-equal protected parameter/bit budget**. Build and unload sequentially.
3. For **every** model, verify bit widths and bytes from actual exported weights and inspect save/reload/generation. No claims based on layer-config flags alone.
4. Evaluate on a validation subset for coherent output, size, memory and safety. Freeze configuration before the held-out test.
5. **Go/no-go:** If M2 and M3 are not actually comparable in storage/precision, repair controls before publishing any advantage claim.

### Day 4 — Evaluate and release (November 14)

1. L4: evaluate M0, M1, M2, M3 one at a time using the same hardware, tokenizer/chat format, batch size, prompt sets, decoding and max-new-token budget. Restart process between models for memory comparison.
2. Record unsafe compliance and over-refusal separately, plus output validity, ARC accuracy, actual file bytes, peak VRAM and generated tokens/sec. Manually audit 50–100 safety cases if feasible.
3. Use paired comparisons by prompt ID and uncertainty intervals; save all per-example outputs. Never reward nonsensical outputs as safe.
4. Compute plots and write report on CPU. Verify public code, licensing, permitted HF artifacts and experiment commands.
5. **Go/no-go:** Submission claims must be traceable to immutable runs, not predicted or demonstration-only values.

## 5. Profiling procedure and what counts as MEASURED

Your existing `resource_monitor.py` has an appropriate starting role, but its current content was not uploaded in this conversation and is **not inspected here**. Adapt it without overwriting local changes.

### Before every paid session

- Record UTC timestamp, Colab runtime label, user-reported hourly rate, current real unit balance, host CPU RAM, `nvidia-smi` name/VRAM, installed Torch/Transformers/CircuitKIT versions, Git commit, run ID and model revision.
- Confirm sufficient persistent storage, loaded model path and cache compatibility. Download/tokenize data before activating GPU where possible.
- Declare `max_runtime_minutes`, `max_estimated_units`, input example count and memory guardrail in the run config. Obtain researcher approval before jobs that may cross a significant cost limit.

### During each experiment

- Use monotonic wall time (`time.perf_counter`); synchronize CUDA before and after a timed workload. Measure warm-up separately.
- Track `torch.cuda.max_memory_allocated()` and `max_memory_reserved()` in GiB, plus `nvidia-smi` device usage when possible. Record CPU RAM peak too if practical.
- For **gradient-based** circuit attribution, do **not** wrap the run in `torch.inference_mode()` or globally disable gradients. `batch_size=1` may be required; do not force batching because of generic GPU rules.
- For **standard inference**, use `torch.inference_mode()` and safe adaptive batching after confirming memory headroom. Measure actual generated token counts.
- Save partial results/resumable checkpoints after a successful stage; avoid keeping multiple full models in GPU memory.

### At runtime shutdown

- Record actual total connected time including install/wait/debug; note whether the session remained idle. Calculate **estimated** unit charge from rate × connected time.
- Record the **Colab-reported actual unit balance before and after** when visible. `actual_units_spent = balance_before - balance_after` (may include other active runtimes; flag that uncertainty).
- Persist JSONL/CSV and text logs, including failures and OOMs. Explicitly disconnect when idle.

### Experiment measurement ledger (fill ONLY after runs)

| run_id | UTC | status | GPU observed | connected h | measured workload s | peak allocated / reserved GiB | disk bytes | tokens/s | estimated units | Colab-observed units | artifact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `PENDING_m0_bf16` | NOT MEASURED | pending | — | — | — | — | — | — | — | — | — |
| `PENDING_m1_uniform` | NOT MEASURED | pending | — | — | — | — | — | — | — | — | — |
| `PENDING_eap_ig` | NOT MEASURED | pending | — | — | — | — | — | — | — | — | — |
| `PENDING_m2_guided` | NOT MEASURED | pending | — | — | — | — | — | — | — | — | — |
| `PENDING_m3_random` | NOT MEASURED | pending | — | — | — | — | — | — | — | — | — |
| `PENDING_eval_final` | NOT MEASURED | pending | — | — | — | — | — | — | — | — | — |

**Machine-readable output:** append run records to `results/raw/compute_usage.jsonl` (or equivalent documented location). Optional aggregated CSV: `results/compute_usage.csv`. Use schema keys: `run_id`, `timestamp_utc`, `git_commit`, `model_revision`, `data_hash`, `runtime_label`, `gpu_name_observed`, `gpu_vram_gib_observed`, `cpu_ram_gib_observed`, `rate_units_per_hour`, `colab_units_before`, `colab_units_after`, `connected_hours`, `workload_seconds`, `estimated_units`, `actual_units_delta`, `peak_allocated_gib`, `peak_reserved_gib`, `checkpoint_bytes`, `generated_tokens`, `generated_tokens_per_second`, `status`, `artifact_path`, `notes`.

When an actual measurement is available, replace `PENDING` with a unique immutable run ID, enter only observed values, and annotate missing fields `NOT_RECORDED` instead of guessing. Link measured rows to `docs/EXPERIMENT_LOG.md` and the evaluation dataset manifest. Update this document each day with measured rows and remaining **real** Colab balance; never overwrite the original user-reported rate table with experiment-derived guesses.

## 6. Stop rules and decision policy

1. **Agent authority:** Codex/Antigravity may implement tests and recommend hardware, but must not start long Colab runs, select G4, or consume reserve units without the researcher's explicit go-ahead.
2. **One failure then diagnose:** After an OOM or backend failure, capture a minimal reproduction, inspect logs, and change only one variable. After at most two similar failures, stop paid retries and debug locally/CPU before escalating.
3. **No idle billed GPU:** Use local CPU or Colab CPU for coding, plots, docs and manual reviews; disconnect GPU sessions when implementation work begins.
4. **Performance evidence:** Warm up and repeat a representative benchmark. Compare per-job unit cost on **observed** throughput rather than assuming A100 or G4 is cheaper or faster.
5. **Scientific priority:** Protect the final evaluation and publishing budget before optional larger discovery sets, additional protection fractions, or repeated seeds.
6. **Submission integrity:** Negative results and unsuccessful backends are documented. Do not fabricate compression, speedups, safety improvements or model compatibility.

## 7. Cross-references

- `docs/PROTOCOL.md`: scientific hypotheses, fixed M0–M3 conditions, and evaluation methodology.
- `docs/ROADMAP.md`: daily task order and scientific go/no-go gates.
- `docs/EXPERIMENT_LOG.md`: experiment registry, versions, hashes and outcomes.
- `docs/RISK_REGISTER.md`: compatibility and methodological risks.
- `AGENTS.md`: binding coding-agent safety, cost and reproducibility rules.

**Change log:** Initial planning uses the researcher's stated rates and capacities. No actual observed experiment metrics were supplied with this update. Revise the measurement ledger after collecting real logs.
