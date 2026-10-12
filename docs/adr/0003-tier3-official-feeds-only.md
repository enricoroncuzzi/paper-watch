# 3. Tier-3 lab content comes only from official RSS/Atom feeds

Date: 2026-10-12

## Status

Accepted

## Context

Tier 3 is the design's third reliability tier: the first-party writing of AI labs and
AI-engineering teams, read from the publisher's own feed. Its value is that the writing is
first-hand, not re-published or scraped third-hand. Scraped HTML is brittle (markup changes break
the parser), carries unclear reuse terms, and would make the corpus depend on page structure nobody
promised to keep stable. The design's Decision 3 names candidate labs, including several Chinese
labs, and requires each to publish an official RSS/Atom feed or documented API listing. A feed
research step on 2026-10-12 checked more than 60 labs and teams and verified 27 official feeds
live.

## Decision

Paper Watch ingests a publisher only through its own official RSS or Atom feed. Tier 3 is therefore
scoped as the official engineering and research blogs of AI labs and AI-engineering teams: the
highest-value agentic-engineering sources are teams such as LangChain, Sourcegraph and Arize AI,
not only frontier labs, and they are explicitly in scope. A feed is official when it is linked from
the publisher's own site (a `<link rel="alternate">` tag or a documented `/rss` or `/feed` path) or
listed in its documented API. Each feed is verified once at implementation time (200 plus an XML
content type) and the verified list lives in `config/feeds.yaml` as data, not code. The client never
scrapes HTML for content. A publisher without an official feed is not ingested through tier 3; its
papers still arrive through tier 1/2 author signals when they exist. Tier-3 text is stored for
private search only, with the licence recorded as "copyright (private search only)".

## Lab list

### Wave 1: active (14 feeds)

These publish first-party agentic-engineering material regularly and have a working, dated feed, so
they ship in `config/feeds.yaml` now.

| Lab / team | Region | Why |
|---|---|---|
| Microsoft Agent Framework | us-eu | First-party agent framework: harness, multi-agent, memory, HITL, MCP skills. |
| Microsoft Foundry | us-eu | Azure AI Foundry agent platform: hosted agents, agent ops. |
| LangChain | us-eu | LangGraph/LangSmith team; highest-frequency agent-engineering feed. |
| Sourcegraph | us-eu | Coding agents, agentic batch changes, code-migration agents. |
| Arize AI | us-eu | Agent evals and observability. |
| Braintrust | us-eu | Agent evaluation and experiments. |
| Zep | us-eu | Agent memory and context engineering. |
| Temporal | us-eu | Durable execution for long-running agents and workflows. |
| Modal | us-eu | Sandboxes and runtime infrastructure for coding agents. |
| Sierra | us-eu | Production customer-service agent engineering. |
| OpenAI | us-eu | Frontier lab; Codex/agent and model releases. |
| Google DeepMind | us-eu | Gemini/agent research and SynthID provenance. |
| Microsoft Research | us-eu | Agent research (Agent Lightning, agent harnesses), full text. |
| Mistral AI | us-eu | Open-model lab; agentic coding and legacy-modernisation posts. |

### Wave 2: present but disabled (13 feeds)

Broader coverage for the ai-forensics and ml-architectures profiles and for frontier-lab context.
They are listed but commented out so enabling them is a deliberate change.

| Lab / team | Region | Why | Note |
|---|---|---|---|
| Engineering at Meta | us-eu | Infra, agents, open-source tooling. | |
| Hugging Face | us-eu | Model/agent/tooling ecosystem. | Title-only feed: no body text to search. |
| Ai2 | us-eu | Open research lab; open models and research agents. | Summary only. |
| Google Research | us-eu | Agentic security and architectures. | Teaser only: near-empty bodies. |
| AWS Machine Learning Blog | us-eu | Bedrock AgentCore and agent engineering. | Full text. |
| NVIDIA Technical Blog | us-eu | Inference/agent serving and architecture engineering. | High volume. |
| Databricks | us-eu | Data+AI platform; agentic SDLC and agent deployment. | Summary only. |
| IBM Research | us-eu | Broad research including agentic AI. | Summary only. |
| PyTorch | us-eu | Framework and architecture engineering. | Full text. |
| Together AI | us-eu | Open-model inference and harness content. | Summary only. |
| Langflow | us-eu | Visual agent/RAG builder. | Low cadence. |
| Hive | us-eu | Synthetic-media detection and content moderation. | Full text. |
| Qwen (Alibaba) | cn | Only official Chinese feed found. | Stale since 2025-09-22; rely on arXiv team names. |

### Labs with no official feed

These have no first-party feed, so they are out of tier 3 and must be covered by tier 1/2 or not at
all. For the ones with a verified arXiv team name, the route is `lab_authors` in the admission
config (Task 6).

| Lab / team | Region | How covered |
|---|---|---|
| Anthropic | us-eu | arXiv author signal `Anthropic`, but reports usually list individual authors, so this is an accepted gap. |
| Meta AI | us-eu | No feed; Meta's engineering blog (Engineering at Meta, wave 2) is Meta's feed route. |
| DeepSeek | cn | arXiv `DeepSeek-AI`; publishes on GitHub too. |
| Moonshot AI (Kimi) | cn | arXiv `Kimi Team`; publishes on GitHub too. |
| Zhipu AI (GLM) | cn | arXiv `Team GLM`, variant `GLM-4.5 Team`. |
| MiniMax | cn | arXiv `MiniMax`. |
| ByteDance Seed | cn | arXiv `Bytedance Seed` (casing as arXiv carries it); GitHub too. |
| Tencent Hunyuan | cn | arXiv `Tencent Hunyuan Team`; GitHub too. |
| Baidu (ERNIE) | cn | No official feed and no matching arXiv team string found; uncovered. |
| Cohere | us-eu | No feed. |
| LlamaIndex | us-eu | No feed. |
| CrewAI | us-eu | No feed. |
| Letta (MemGPT) | us-eu | No feed. |
| Cursor | us-eu | No feed. |
| xAI | us-eu | No feed. |
| Perplexity | us-eu | No feed. |
| Cognition (Devin) | us-eu | No feed. |
| All Hands AI (OpenHands) | us-eu | No feed. |
| Haystack / deepset | us-eu | No feed. |
| Pinecone | us-eu | No feed. |
| Replit | us-eu | No feed. |
| Langfuse | us-eu | No feed. |
| Galileo | us-eu | No feed. |
| Patronus AI | us-eu | No feed. |
| Pydantic AI | us-eu | No blog or feed; documentation only. |
| Content Authenticity Initiative | us-eu | No feed. |
| Reality Defender | us-eu | No feed. |
| TrueMedia | us-eu | No feed. |
| Alibaba Cloud Blog | cn | No feed; Qwen is Alibaba's only feed. |
| Shanghai AI Lab (InternLM/InternVL) | cn | No feed; no arXiv team strings found, lists individuals. |
| OpenBMB / ModelBest | cn | arXiv `MiniCPM Team`. |
| 01.AI | cn | No feed and no arXiv team string found; uncovered. |
| StepFun | cn | arXiv `StepFun`. |
| Ant Group | cn | No first-party feed; arXiv `Ant Group`, `Ling Team`. |
| Xiaomi | cn | No first-party feed; arXiv `Xiaomi LLM-Core Team`. |

Not checked: SenseTime, iFlytek, Huawei (Pangu/Noah's Ark), Baichuan and Shanghai AI Lab's
foundation-model arm were outside the Decision 3 set. The evidence so far predicts the same
"no feed, publish on arXiv/GitHub" result.

### Official feeds excluded from the focused list

| Lab / team | Why excluded |
|---|---|
| Google Developers Blog | Live and on-topic, but items carry no `pubDate`/`updated`, so the client would silently drop every entry. |
| Microsoft AutoGen Blog | Official but stale (newest item 2025-01-17); superseded by Microsoft Agent Framework. |
| TensorFlow Blog | Official but stale (newest item 2025-08-19). |
| GitHub Blog | General product/engineering blog; agent content is occasional and interleaved. |
| Vercel News | Mostly changelog/product noise at high volume. |
| Cloudflare Blog | Infrastructure/security focus; agents appear occasionally. |
| n8n Blog | Automation/agent workflows, product-tutorial heavy. |
| Weaviate Blog | RAG/vector-search tooling, product-marketing heavy. |
| Qdrant Blog | RAG/vector search, summary only. |
| Apple Machine Learning Research | Summary only; ml-architectures fallback, not agentic. |
| Amazon Science | Broad research with occasional AI. |
| Google AI Blog | Mostly product announcements; Google Research and DeepMind already cover Google. |

### Labs without a feed: the arXiv team-name route

The Chinese labs and Anthropic are covered through verified arXiv author strings in the admission
config, not through feeds. Verified names: `DeepSeek-AI`, `Qwen`, `Qwen Team`, `Kimi Team`,
`Team GLM`, `GLM-4.5 Team`, `MiniMax`, `Bytedance Seed`, `Tencent Hunyuan Team`, `StepFun`,
`MiniCPM Team`, `Ant Group`, `Ling Team`, `Xiaomi LLM-Core Team`. The existing Western names
(`Google DeepMind`, `OpenAI`, `Anthropic`, `Meta FAIR`) stay. Anthropic reports normally list
individual authors, so its author string may not fire: that is an accepted gap, not an exception to
the no-feed rule. Baidu and 01.AI have no matching team string and stay uncovered.

### Content quality

Title-only and summary-only feeds give private search little to work with: the plan's extractor
yields a short excerpt or an empty body, so those items can appear in the weekly shortlist and be
linked but cannot ground an answer. In wave 1, OpenAI, Google DeepMind, LangChain, Sourcegraph,
Braintrust, Zep, Temporal, Modal, Sierra and Mistral AI carry summary or title only, while the
Microsoft feeds, Microsoft Research and Arize AI carry full text. The same split is why Hugging
Face, Mistral AI, Google Research and several wave-2 feeds sit outside the focused core.

## Consequences

- A feed outage is a reported failure, not a crash: the client records the feed URL and error and
  keeps the other feeds flowing.
- Adding, removing or enabling a lab is a line in `config/feeds.yaml`, reviewed like any other
  configuration change; wave 2 is deliberately commented until someone chooses the extra
  classification and embedding cost.
- Tier 3 cannot reach a publisher that has no official feed; those labs are covered through arXiv
  team names, and the few with neither stay uncovered by design rather than by scraping.
- Summary-only and title-only feeds are accepted as pointer sources, not as text the answer can
  quote.
- Most feeds expose only their last 10-20 posts, so an outage longer than that window loses the
  posts published before it: once they scroll off the feed, the next run cannot fetch them.
- A post the lab later edits is not refreshed: feed items carry version 1 only, so the stored row
  keeps the first version even after the source changes.
