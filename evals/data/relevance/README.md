# Relevance labels

`abstracts.jsonl` holds 328 recent arXiv abstracts, each labelled with one topic profile from `config/profiles.yaml` or `none`. `evals/relevance_eval.py` uses them to choose each profile's `relevance_threshold` and to measure the embedding classifier on abstracts it was not calibrated on.

**Review status: labelled by the Task 11b planning agent on 2026-10-13, awaiting the captain's review.** A changed label changes the data fingerprint, so after editing labels, rerun `uv run python -m evals.relevance_eval --write` and commit the new thresholds and baseline with the label change.

## Format

One JSON object per line, sorted by id:

- `arxiv_id`: the base arXiv id, without the version suffix.
- `title` and `abstract`: as arXiv publishes them, with whitespace collapsed (as the arXiv source does).
- `label`: `agentic-engineering`, `ai-forensics`, `ml-architectures` or `none`.
- `note` (optional, 101 rows): why a borderline abstract got its label, or which rule decided it.

The split is not stored. An abstract is in the validation half when the first byte of the SHA-256 of its id is even, and in the test half otherwise. It therefore keeps its split when abstracts are added. The current split has 167 validation and 161 test abstracts:

| label | validation | test |
|---|---|---|
| agentic-engineering | 24 | 29 |
| ai-forensics | 19 | 17 |
| ml-architectures | 21 | 21 |
| none | 103 | 94 |

## Reading the scores

The headline figures describe this set, not a live stream:

- The profile descriptions were chosen after two looks at the test half, so the test figures are slightly optimistic as held-out numbers. The thresholds themselves are chosen on the validation half only.
- The set is about 40 percent positive by design (131 of 328 abstracts; 41.6 percent of the test half), because positives were deliberately oversampled. Admitted precision is therefore a relative measure for comparing configurations, not a live precision estimate: on a stream with 5-10 percent relevant papers it would be far lower.
- The 95 percent bootstrap interval for test macro F1 is about 0.65 to 0.82 (`python -m evals.relevance_eval` prints the current interval), so the interval's half-width is about 0.09; read differences under that with care and compare two configurations on the same rows with a paired bootstrap.

The committed baseline was computed on arm64 (Apple M2); the `model-evals` workflow reproduces it on ubuntu-latest (x86-64) within the 0.03 tolerance.

## Labelling criteria

Each abstract gets exactly one label, decided by the paper's main contribution as the title and abstract state it.

- **agentic-engineering**: the paper builds, evaluates, analyses or secures systems in which a language model acts with some autonomy. That covers agent harnesses, scaffolds and skills, tool use and function calling, MCP, multi-agent LLM systems, coding, terminal, GUI and research agents, agent memory, context and planning, agent benchmarks and failure analysis, agent cost and routing, agent training (for example RL for search agents), agent safety, monitoring and security (prompt injection, malicious skills, sandboxes), and agent systems built for a domain such as science or health. Not included:
  - an LLM used as a single component without acting (classification, generation quality, LLM-as-judge);
  - LLM reasoning, alignment, values or interpretability studied in text-only scenarios;
  - classic non-LLM multi-agent systems and game theory;
  - legal and policy analysis of agentic AI;
  - robot policy learning.
- **ai-forensics**: the paper detects, localises, attributes or watermarks synthetic or manipulated content, or attacks such a detector or watermark. That covers deepfake and spoofed-speech detection, image manipulation localisation, AI-generated image, video, audio, music and text detection, and watermarking of generated text, code, images, audio, DNA and agent actions. Not included:
  - safety and moderation of what generators produce;
  - compression or editing of generated media;
  - GPS or network spoofing;
  - software supply-chain integrity;
  - "provenance" in the database or KV-cache sense.
- **ml-architectures**: the paper proposes or analyses a general-purpose neural network architecture or core building block. That covers attention and sparse or linear attention variants, recurrent memories and state space models, mixture-of-experts layers and routing, looped and hybrid transformers, positional encodings, tokenisation and embedding tables, and analyses of how these components behave. Not included:
  - a network designed for one application domain (medical imaging, PDEs, crash simulation, face detection);
  - quantisation, KV-cache compression, kernels and serving or training systems for an existing architecture;
  - training methods, optimisers and training-dynamics theory;
  - architecture search algorithms;
  - model releases.
- **none**: everything else.

When a paper fits two profiles, the label follows its main contribution. On a true tie the primary profile wins, in the order agentic-engineering, ai-forensics, ml-architectures.

## Provenance and licence

- Harvested on 2026-10-13 from arXiv's OAI-PMH interface (`https://oaipmh.arxiv.org/oai`, `ListRecords`, `metadataPrefix=arXiv`). The first result page was taken for each of cs.SE, cs.AI, cs.CL, cs.CV, cs.CR, cs.LG, cs.DB, eess.AS, eess.SP, eess.IV, math.OC and q-bio.QM with `from=2026-10-05`, and for cs.MA with `from=2026-10-01`. Requests went one at a time, 3 s apart. Only new submissions (ids `2609.*` and `2610.*`) were kept.
- Which abstracts were read was chosen by keyword buckets: agent, forensics, architecture and hard-negative terms, then a seeded hash order. Every label was then assigned by hand, against the criteria above, from the title and abstract. The keyword bucket never decided a label.
- arXiv releases its metadata, abstracts included, under CC0 1.0 (https://info.arxiv.org/help/api/tou.html). No PDF was downloaded.
