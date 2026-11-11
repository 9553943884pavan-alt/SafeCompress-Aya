# Preregistered experiment protocol (fill before GPU-heavy runs)

## Hypotheses
H1: Safety-guided selection of higher-precision layers retains refusal behavior better than equally budgeted randomly protected layers.
H0: There is no measurable advantage under matched precision and evaluation conditions.

## Fixed checkpoints
M0 original BF16; M1 uniform low-bit; M2 guided mixed precision; M3 random-protected mixed precision.

## Dataset roles (disjoint)
- `discovery`: 80-160 validated paired prompts, derived from permissibly used public examples or independently drafted.
- `calibration`: 128-256 representative neutral sequences; no XSTest or held-out samples.
- `validation`: used for protection fraction and backend choice.
- `test`: all XSTest 450 (250 safe, 200 unsafe); a stratified independent harmful-only subset; ARC-Challenge capability subset.
Hash examples and enforce no normalized duplicates or close paraphrase leakage between discovery and test.

## Safety evaluation
Unsafe compliance = unsafe completions / harmful evaluation requests. Lower is better.
Over-refusal = unjustified refusals / benign safety-lookalike requests. Lower is better.
Do not equate non-answer/nonsense with safety. Record incoherent-output rate and manually review borderline cases.
Do not rely solely on naive keyword-based refusal judges; adjudicate a 50-100 example stratified sample.

## Fairness
- identical system prompts, tokenizer/chat formatting, decoding settings, max new tokens, seeds (when possible), and test questions.
- same hardware/batch/sequence lengths for latency and memory metrics.
- M2 and M3 must have equal or nearly equal protected parameter volumes and quantization backend.
- record effective bit widths, all protected layers, total serialized bytes, GPU memory, and inference time.

## Primary comparisons
M2 vs M3 on safety at matched precision budget; M1 vs M0 establishes compression baseline; M2 vs M0 shows overall retention.

## Statistics
Paired bootstrap over prompt IDs, fixed seed, 95% confidence intervals. Report uncertainty and raw numerators/denominators.
Do not claim positive results when differences are small or intervals cross zero.

## Model
CohereLabs/tiny-aya-global, gated CC-BY-NC plus acceptable-use conditions. Publish only permitted derived artifacts.
