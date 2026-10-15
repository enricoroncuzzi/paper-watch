# Retrieval evals as a CI gate

Tests check that code does what it says; evals check that a system that ranks or generates does it well enough, on a labelled set. For retrieval the classic metrics are recall@k (did the right item appear in the top k), MRR (how high the first right item ranks, as 1/rank averaged over questions), and nDCG@k (graded credit that rewards putting right items near the top). A committed baseline turns those numbers into a regression gate: CI fails if a change drops a metric by more than a small tolerance. The offline eval uses the deterministic fake embedder so it is fast and reproducible; a nightly job later runs the same set against the real local embedding model, which costs nothing but a 180 MB download.

## What the per-kind numbers show

The 20 questions are tagged `exact_name` or `paraphrase`, and `evaluate` reports recall@k and MRR for each kind next to the overall metrics. On the committed set with the lexical `FakeEmbedder`, the baseline is:

| kind | recall@5 | MRR |
|---|---|---|
| exact_name | 1.000 | 1.000 |
| paraphrase | 0.500 | 0.450 |
| overall | 0.750 | 0.725 |

The gap is the point of the split, and it is evidence for the full-text risks the Task 10 review named. Exact-name questions share a distinctive token with the right item, so the lexical vector branch finds it and recall is perfect. Paraphrase questions are written with no content word in common with the relevant title, and their many terms rarely all appear in the item, so `websearch_to_tsquery`'s all-terms AND leaves the text branch empty and only the vector branch can help: about half the paraphrase questions still find the right item, the rest do not. That is the first risk the Task 10 review raised (an OR-relaxed query might retrieve more natural questions), and it is now measured by this gate rather than assumed. The second risk, the `english` tokeniser splitting space-separated model names, needs a name with a space in it and is left to the real-model eval in Task 11b, along with normalisation, `ef_search`, candidates and `rrf_k`, which a fake embedder and 30 items cannot measure. The gate itself compares only the three overall metrics; the per-kind numbers are the reading, not the tripwire.

The implementation is `evals/retrieval_eval.py` with the labelled set in `evals/data/`, the committed baseline in `evals/baseline.json`, the gate `paperwatch eval` (the `eval` CI job) and the harness tests in `tests/integration/test_eval_gate.py`.
