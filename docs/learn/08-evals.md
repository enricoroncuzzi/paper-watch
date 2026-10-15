# Retrieval evals as a CI gate

Tests check that code does what it says; evals check that a system that ranks or generates does it well enough, on a labelled set. For retrieval the classic metrics are recall@k (did the right item appear in the top k), MRR (how high the first right item ranks, as 1/rank averaged over questions), and nDCG@k (graded credit that rewards putting right items near the top). A committed baseline turns those numbers into a regression gate: CI fails if a change drops a metric by more than a small tolerance. The offline eval uses the deterministic fake embedder so it is fast and reproducible; a nightly job later runs the same set against the real local embedding model, which costs nothing but a 180 MB download.

## What the numbers show

`evaluate` reports the hybrid metrics overall and per kind, and, next to them, each branch on its own: text-only recall@k and MRR (through the same `text_query` the fusion uses), vector-only recall@k and MRR, and how often the text branch is non-empty. With the lexical `FakeEmbedder` on the committed set (39 items, 29 questions, 16 `exact_name` and 13 `paraphrase`) the baseline is:

| kind | hybrid recall@5 | hybrid MRR | text-only recall@5 | text-only MRR | text branch non-empty |
|---|---|---|---|---|---|
| exact_name | 1.000 | 1.000 | 0.750 | 0.750 | 0.750 |
| paraphrase | 0.577 | 0.538 | 0.077 | 0.077 | 0.077 |
| overall | 0.810 | 0.793 | 0.448 | 0.448 | 0.448 |

The gate commits the three hybrid metrics plus text-only recall and MRR on `exact_name` (0.750 each); the rest is reported, not gated.

Three findings come out of this, and only the first two are things the fake embedder can measure.

**Paraphrase questions measure lexical overlap with the body, not paraphrase ability.** The 13 paraphrase questions are title-disjoint: no content word is shared with the relevant title. But the fake embedder is a hashed bag of words, so the vector branch scores on any word shared with the item's body anyway. A paraphrase question that happens to share body words is retrieved, and one that does not is missed; that is the whole 0.577. The number is a fact about this embedder's overlap with these bodies, not about how well natural-language questions are handled.

**The text branch is empty for almost every natural question.** `websearch_to_tsquery` ANDs a question's content words, so a long natural question rarely matches any item: the text branch is non-empty for only 1 of the 13 paraphrase questions (0.077), and empty for 9 of the 10 pre-existing ones. That is the measured evidence for the first risk the Task 10 review raised (an OR-relaxed query might retrieve more natural questions), and the gate now fails if that text-branch recall drops. The one exception shows a related trap: `websearch_to_tsquery` turns a natural "or" into an OR operator, so "NimbusBench or long tool chains" matches both the target and a decoy that shares the common words, and the gate's text-only MRR is what catches a wrong ordering of that list.

**Risk (2), the `english` tokeniser and space-separated names, is now measured, and it costs the text branch.** Three committed questions use model names in the spelling a person would type: `Llama-3.1-8B` asked as "llama 3.1 8b", `Qwen2.5-VL` asked as "qwen 2.5", and `RAG/agentic` asked as "RAG agentic". The item's `tsvector` splits the hyphen and keeps the slash compound as one token, while the question's terms do not, so the text branch returns nothing for all three: text-only exact-name recall is 0.750, not 1.000, and that number is committed as the baseline. The hybrid metrics still find them because the lexical vector branch shares the raw tokens; the point is that the text branch misses exactly these spellings, and the eval reports it instead of assuming it.

The gate catches a broken text branch because of the branch-only metrics and the decoys, not because the hybrid ranking happens to move. Three decoy questions ("What is it about NimbusBench for me?") put the target below a decoy in the vector branch, so only a working text branch lifts it back to rank 1, and text-only exact-name recall falls to 0 if the branch returns nothing. Patching `text_query` to return nothing, to never match, to sort worst-first, or dropping the text list from the fusion each makes `paperwatch eval` exit 1; that is the acceptance test the Task 11 review asked for.

The risks numbers cannot measure here (ts_rank normalisation, `ef_search` recall, candidates and `rrf_k`, a similarity floor) depend on a real embedder and real volume, so they are measured later on real data with the local model at the corpus-size checkpoints, not with 39 synthetic items.

The implementation is `evals/retrieval_eval.py` with the labelled set in `evals/data/`, the committed baseline in `evals/baseline.json`, the gate `paperwatch eval` (the `eval` CI job) and the tests in `tests/integration/test_eval_gate.py` and `tests/unit/test_retrieval_eval.py`.
