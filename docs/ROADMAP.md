# Four-day execution roadmap (November 11–14, 2026)

## Day 1: baseline and quantization compatibility
- Verify registration, model gate, licenses, GPU, RAM, disk, HF identity.
- Pin and log Python/Torch/Transformers/TransformerLens/CircuitKIT/CuratorKIT/AuditKIT versions.
- Load original BF16 model and run deterministic generation using official chat template.
- Curate separate calibration / circuit-discovery / validation / held-out test inputs, export manifests and hashes.
- Benchmark BF16 quality, safety, memory, speed.
- Implement a uniform 4-bit baseline: quantize -> save -> reload -> generate; verify actual checkpoint bytes.
**Go/no-go:** do not proceed to full circuit discovery until the quantization/export/reload path works.

## Day 2: circuit discovery
- Construct 16-32 tightly matched clean/corrupted contrast pairs and a clear scalar refusal-related objective.
- Run small EAP-IG discovery with batch size 1; check parity and actual behavioral contrast.
- If valid, extend to 64-128 separate discovery examples; export circuit scores and layer ranking.
- Run a small patching or ablation check plus random controls.
- Save scores, run configs, GPU peak memory, warnings.
**Go/no-go:** no safety-circuit claim without behavioral validation; fallback to reproducible negative result.

## Day 3: safety-guided quantization
- Map circuit scores to supported quantization module/layer names and document protection choices.
- Create guided model: candidate high precision fraction 0.2 (or smallest feasible fraction).
- Create random-protected model with identical number/size of protected modules; control seed 42.
- Match effective precision budgets; compare actual bytes after packing.
- Save / reload / generate from each artifact. Check no NaN, language collapse, template breakage.
- Run preliminary validation. Select hyperparameters without opening held-out test results.
**Go/no-go:** if CircuitKIT backend is unsupported for Tiny Aya, report failure and do not relabel another method as circuit-guided.

## Day 4: evaluation and delivery
- Run frozen final test set for all 4 conditions at fixed decoding and sequence lengths.
- Score unsafe compliance (lower), over-refusal (lower), ARC accuracy (higher), size and peak VRAM (lower), throughput (higher).
- Manually adjudicate at least 50-100 mixed safety outputs with a documented rubric; bootstrap paired confidence intervals.
- Plot size/safety and size/utility tradeoffs; write failure analysis and limitations.
- Reproduce a smoke test from fresh runtime. Publish allowed artifacts to HF + public GitHub.
- Validate collection visibility, README, licensing, final abstract, link access and demonstration.

**Deadline:** first-stage Nov 15 2026 11:59 PM AoE. Aim to submit earlier.
