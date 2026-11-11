# Risks and fallback decisions

1. **Tiny Aya backend cannot export quantized checkpoint.** Immediate Day 1 blocker. Capture minimal reproducer. Try documented alternate supported backend. Do not spend a day discovering a circuit before verifying this.
2. **Circuit score does not correlate with actual refusal.** Record negative result and use random controls; do not label an arbitrary heatmap as safety-critical.
3. **Uniform W4 is safer than original / safety drop absent.** It may still be possible to show safety preservation, but do not manufacture regression. Report honest no-regression result and strengthen utility/over-refusal analysis.
4. **Guided and random models use unequal storage.** Match selected layers by parameter volume, then compare actual disk sizes. If imperfect, disclose.
5. **Quantization improves memory but not speed.** Report actual findings and backend/kernel overhead; don't claim latency gains without measurements.
6. **Public release conflicts with gated model terms.** Publish code, scores, configs, safe predictions and license-compatible artifacts. Link user to gated original and clarify derived checkpoint handling with organizer.
7. **GPU quota / OOM.** Small smoke tests first, batch 1, shorter sequence lengths without invalidating tasks, offload and resume after saving artifacts.
8. **Safety auto-judge unreliable.** Independent rubric, human adjudication, inter-annotator consistency if possible, label uncertainty.
