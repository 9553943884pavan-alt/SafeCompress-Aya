# Script implementation order

01 verify_env.py — log versions, CUDA, BF16 model load, generation and the official tokenizer template.
02 prepare_data.py — real CuratorKIT data stage with manifests and no test leakage.
03 baseline_eval.py — BF16 evaluation via AuditKIT; immutable prediction files.
04 quantize_uniform.py — check actual supported backend, save, reload, measure files.
05 discover_circuit.py — small EAP-IG run on disjoint paired examples; save scores.
06 quantize_guided.py — CircuitKIT quantize using validated circuit scores.
07 quantize_random.py — random protection at exactly matching parameter budget.
08 run_evaluations.py — same prompts/decoding, human safety rubric, output IDs.
09 analyze.py — plot risk-utility-size tradeoff, bootstrapped confidence intervals.
10 publish.py — allowed artifacts and metadata to public HF collections.

These scripts have intentionally NOT been fabricated as executable implementations because backend behavior on Tiny Aya must first be verified. Assign Codex/Antigravity to implement and test them sequentially.
