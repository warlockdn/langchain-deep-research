# 🔬 Open Deep Research



Deep research has broken out as one of the most popular agent applications. This is a simple, configurable, fully open source deep research agent built around OpenAI models, Exa web search, and MCP servers. It's performance is on par with many popular deep research agents ([see Deep Research Bench leaderboard](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)).



### 🚀 Quickstart

1. Clone the repository and activate a virtual environment:

```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

1. Install dependencies:

```bash
uv sync
# or
uv pip install -r pyproject.toml
```

1. Set up your `.env` file to customize the environment variables:

```bash
cp .env.example .env
```

1. Launch agent with the LangGraph server locally:

```bash
# Install dependencies and start the LangGraph server
uv run langgraph dev --config langgraph.json
```

This will open the LangGraph Studio UI in your browser.

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Ask a question in the `messages` input field and click `Submit`. Select different configuration in the "Manage Assistants" tab.

### ⚙️ Configurations

#### LLM :brain:

Open Deep Research uses OpenAI chat models for a few different tasks. See the below model fields in the [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) file for more details. This can be accessed via the LangGraph Studio UI. 

- **Summarization** (default: `openai:gpt-4.1-mini`): Summarizes search API results
- **Research** (default: `openai:gpt-4.1`): Power the search agent
- **Compression** (default: `openai:gpt-4.1`): Compresses research findings
- **Final Report Model** (default: `openai:gpt-4.1`): Write the final report

> Note: the selected model will need to support [structured outputs](https://python.langchain.com/docs/integrations/chat/) and [tool calling](https://python.langchain.com/docs/how_to/tool_calling/).

#### Search API :mag:

Open Deep Research uses Exa as its built-in search backend. MCP tools can also be layered in via `mcp_config`. See the `search_api` and `mcp_config` fields in the [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) file for more details. This can be accessed via the LangGraph Studio UI. Make sure `EXA_API_KEY` is set in your environment before running searches.

#### Other

See the fields in the [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) for various other settings to customize the behavior of Open Deep Research. 

### 📄 PDF Chunk Counting

This repo includes a LangChain-driven PDF chunk counter. It loads local PDFs from `docs/` via LangChain's `DoclingLoader`, optionally splits the extracted documents into smaller chunks, and prints per-file and total chunk counts.

#### Local install

```bash
uv sync
```

The LangChain Docling integration is provided by `langchain-docling`. Optional post-load splitting uses `langchain-text-splitters`.

#### Local commands

Run the chunk counter:

```bash
uv run python scripts/pdf_chunk_count.py --input-dir docs
```

Write chunks to a specific file:

```bash
uv run python scripts/pdf_chunk_count.py \
  --input-dir docs \
  --output-file docs/pdf_chunks.jsonl
```

Split the extracted page documents into smaller chunks:

```bash
uv run python scripts/pdf_chunk_count.py \
  --input-dir docs \
  --split \
  --chunk-size 1000 \
  --chunk-overlap 200
```

Split and store the final chunks in PGVector:

```bash
uv run python scripts/pdf_chunk_count.py \
  --input-dir docs \
  --split \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --pgvector
```

Same PGVector target, but embed locally with [FastEmbed](https://github.com/qdrant/fastembed) instead of OpenAI (use `--pgvector-fastembed` or `--pgvector`, not both):

```bash
uv run python scripts/pdf_chunk_count.py \
  --input-dir docs \
  --split \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --pgvector-fastembed
  --fastembed-model nomic-ai/nomic-embed-text-v1.5
  
# optional: --fastembed-model nomic-ai/nomic-embed-text-v1.5
```

#### Docker (standalone Docling image)

The repo `Dockerfile` builds an AMD64 image that:
- installs Docling using CPU-only PyTorch wheels,
- pre-downloads Docling models during build (standalone runtime),
- runs only `scripts/pdf_chunk_count.py` and accepts CLI args at `docker run`.

Build image:

```bash
docker build -t pdf-chunker -f Dockerfile .
```

Show script help:

```bash
docker run --rm pdf-chunker
```

Run against mounted local docs:

```bash
docker run --rm \
  -v "$(pwd)/docs:/app/docs" \
  pdf-chunker \
  --input-dir docs \
  --output-file docs/pdf_chunks.jsonl
```

Split mode:

```bash
docker run --rm \
  -v "$(pwd)/docs:/app/docs" \
  pdf-chunker \
  --input-dir docs \
  --split \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --output-file docs/pdf_chunks.jsonl
```

#### Notes

- The script uses `DoclingLoader` with `export_type="markdown"` and writes one JSONL record per returned LangChain `Document`.
- When `--split` is set, the script applies LangChain's `RecursiveCharacterTextSplitter` to those page documents before writing JSONL.
- When `--pgvector` is set, the script also stores the final emitted documents in the configured PGVector collection using `OPENAI_EMBEDDING_MODEL` (or `PGVECTOR_EMBEDDING_MODEL`), `CONNECTION_STRING`, and `COLLECTION_NAME`.
- When `--pgvector-fastembed` is set, ingestion uses local FastEmbed embeddings instead (`fastembed` dependency). Model order: `--fastembed-model`, then `PGVECTOR_FASTEMBED_MODEL`, then default `BAAI/bge-small-en-v1.5`. The agent’s `document_search` tool still queries with OpenAI embeddings unless you change that separately; keep ingest and query embedders aligned for meaningful similarity search.
- By default, chunk records are written to `<input-dir>/pdf_chunks.jsonl` as JSONL with `source_file`, `chunk_index`, `page_content`, and `metadata`.
- The script exits non-zero if a discovered PDF produces no chunks, and it prints the missing filenames explicitly instead of silently undercounting.

### ⚡ FastAPI Server

A lightweight HTTP API wraps the same `deep_researcher` graph so you can call it from any HTTP client without LangGraph Studio.

#### Start

```bash
uv run uvicorn open_deep_research.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs are served at `http://localhost:8000/docs`.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/research` | Blocking – waits for the full report |
| `POST` | `/research/stream` | SSE stream of graph update events |
| `GET`  | `/health` | Liveness probe |

#### Request format

Both endpoints accept the same JSON body:

```json
{
  "messages": [
    {"role": "user", "content": "What are the latest advances in quantum error correction?"}
  ],
  "thread_id": "optional-existing-thread-id"
}
```

`thread_id` is optional – omit it to start a fresh conversation. Use `"role": "ai"` for assistant turns when resuming a thread.

#### Sample requests

**Blocking (curl)**

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What are the latest advances in quantum error correction?"}
    ]
  }'
```

Response:

```json
{
  "thread_id": "3f8a12bc-...",
  "final_report": "## Quantum Error Correction in 2025\n\n..."
}
```

**Streaming (curl)**

```bash
curl -N -X POST http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Give me a market overview of AI chip manufacturers."}
    ]
  }'
```

Each line is a `data: <json>` SSE event. The stream ends with `data: [DONE]`.

**Python (httpx)**

```python
import httpx, json

# blocking
resp = httpx.post(
    "http://localhost:8000/research",
    json={
        "messages": [
            {"role": "user", "content": "Summarise CRISPR therapeutics progress in 2025."}
        ]
    },
    timeout=300,
)
print(resp.json()["final_report"])

# streaming
with httpx.stream(
    "POST",
    "http://localhost:8000/research/stream",
    json={
        "messages": [
            {"role": "user", "content": "Overview of fusion energy breakthroughs."}
        ]
    },
    timeout=None,
) as r:
    for line in r.iter_lines():
        if line.startswith("data:") and line != "data: [DONE]":
            chunk = json.loads(line[5:])
            print(chunk)
```

---

### 📊 Evaluation

Open Deep Research is configured for evaluation with [Deep Research Bench](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard). This benchmark has 100 PhD-level research tasks (50 English, 50 Chinese), crafted by domain experts across 22 fields (e.g., Science & Tech, Business & Finance) to mirror real-world deep-research needs. It has 2 evaluation metrics, but the leaderboard is based on the RACE score.

#### Usage

> Warning: Running across the 100 examples can cost ~$20-$100 depending on the model selection.

The dataset is available on [LangSmith via this link](https://smith.langchain.com/public/c5e7a6ad-fdba-478c-88e6-3a388459ce8b/d). To kick off evaluation, run the following command:

```bash
# Run comprehensive evaluation on LangSmith datasets
python tests/run_evaluate.py
```

This will provide a link to a LangSmith experiment, which will have a name `YOUR_EXPERIMENT_NAME`. Once this is done, extract the results to a JSONL file that can be submitted to the Deep Research Bench.

```bash
python tests/extract_langsmith_data.py --project-name "YOUR_EXPERIMENT_NAME" --model-name "you-model-name" --dataset-name "deep_research_bench"
```

This creates `tests/expt_results/deep_research_bench_model-name.jsonl` with the required format. Move the generated JSONL file to a local clone of the Deep Research Bench repository and follow their [Quick Start guide](https://github.com/Ayanami0730/deep_research_bench?tab=readme-ov-file#quick-start) for evaluation submission.

#### Results


| Name                           | Commit                                                                                                        | Summarization       | Research       | Compression    | Total Cost | Total Tokens | RACE Score | Experiment                                                                                                                                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------- | ------------------- | -------------- | -------------- | ---------- | ------------ | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPT-5                          | [ca3951d](https://github.com/langchain-ai/open_deep_research/pull/168/commits)                                | openai:gpt-4.1-mini | openai:gpt-5   | openai:gpt-4.1 |            | 204,640,896  | 0.4943     | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-613c-4bda-8bde-f64f0422bbf3/compare?selectedSessions=4d5941c8-69ce-4f3d-8b3e-e3c99dfbd4cc&baseline=undefined) |
| Defaults                       | [6532a41](https://github.com/langchain-ai/open_deep_research/commit/6532a4176a93cc9bb2102b3d825dcefa560c85d9) | openai:gpt-4.1-mini | openai:gpt-4.1 | openai:gpt-4.1 | $45.98     | 58,015,332   | 0.4309     | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=cf4355d7-6347-47e2-a774-484f290e79bc&baseline=undefined)                                              |
| Deep Research Bench Submission | [c0a160b](https://github.com/langchain-ai/open_deep_research/commit/c0a160b57a9b5ecd4b8217c3811a14d8eff97f72) | openai:gpt-4.1-nano | openai:gpt-4.1 | openai:gpt-4.1 | $87.83     | 207,005,549  | 0.4344     | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=e6647f74-ad2f-4cb9-887e-acb38b5f73c0&baseline=undefined)                                              |


### 🚀 Deployments and Usage

#### LangGraph Studio

Follow the [quickstart](#-quickstart) to start LangGraph server locally and test the agent out on LangGraph Studio.

#### Hosted deployment

You can easily deploy to [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options). 

#### Open Agent Platform

Open Agent Platform (OAP) is a UI from which non-technical users can build and configure their own agents. OAP is great for allowing users to configure the Deep Researcher with different MCP tools and search APIs that are best suited to their needs and the problems that they want to solve.

We've deployed Open Deep Research to our public demo instance of OAP. All you need to do is add your API Keys, and you can test out the Deep Researcher for yourself! Try it out [here](https://oap.langchain.com)

You can also deploy your own instance of OAP, and make your own custom agents (like Deep Researcher) available on it to your users.

1. [Deploy Open Agent Platform](https://docs.oap.langchain.com/quickstart)
2. [Add Deep Researcher to OAP](https://docs.oap.langchain.com/setup/agents)

### Questions

Good question. This handbook is **not something you ask “what is Medicaid?”**—it’s best used for **policy, configuration, compliance, and operating decisions**.

Below is a **curated set of *high‑quality, detailed questions*** you can ask this document, grouped by intent. These are the kinds of questions **architects, product owners, ops, compliance, and bid teams** typically use it for.

***

## 1. Compliance & Regulatory Questions (Very strong fit)

Ask these when you want **definitive, audit-safe answers**:

*   *How does Express Scripts ensure compliance with federal MDRP rebate requirements at point of sale?*
*   *What edits are applied to prevent payment for non‑rebate‑eligible drugs?*
*   *How are HCFA NDC termination and obsolete dates enforced, and can run‑out periods be customized?*
*   *What is the process when state Medicaid guidance changes after implementation?*
*   *Who is responsible for monitoring new state regulations—plan or Express Scripts?*
*   *How does Express Scripts support state readiness reviews and Medicaid audits?*
*   *What prescriber validation checks are mandatory vs configurable by state?*

✅ Best for: **Compliance reviews, regulator discussions, risk assessments**

***

## 2. Plan Setup & Configuration Questions

Ask these when designing or onboarding a **Managed Medicaid plan**:

*   *What BIN/PCN strategy is recommended for Managed Medicaid, and when should it differ?*
*   *How are Medicaid population type codes assigned and maintained over time?*
*   *What happens if multiple Medicaid populations are mixed in one eligibility group?*
*   *How does real‑time eligibility updating work via eSD vs batch files?*
*   *What data elements are required to support household-level accumulators?*

✅ Best for: **Implementation planning, eligibility design, architecture decisions**

***

## 3. Copay, Exemption & Cost‑Sharing Questions

Very detailed and practical section—great questions include:

*   *How are Medicaid copays configured across contract, group, and member levels?*
*   *Which populations must always have $0 copays under federal rules?*
*   *How does pregnancy-based copay exemption work if eligibility data is stale?*
*   *Can copays be driven by ingredient cost (sliding scale), and how?*
*   *What NCPDP fields control copay exemption at POS?*

✅ Best for: **Benefit design, state-mandated copay compliance**

***

## 4. Coordination of Benefits (COB) & Payment Integrity

Ask when Medicaid is **payer of last resort**:

*   *How does Express Scripts identify members with Other Health Insurance (OHI)?*
*   *What is the difference between OPAP and OPPRA COB methodologies?*
*   *How are reject codes and messaging handled when Medicaid is incorrectly billed as primary?*
*   *What happens if COB info is missing at point of sale?*
*   *How does automated OHI/TPL file processing work?*

✅ Best for: **Cost avoidance, capitation accuracy, payment integrity**

***

## 5. Clinical & Utilization Management Questions

Ask when balancing **cost, access, and safety**:

*   *How are state-mandated formularies loaded and maintained?*
*   *What utilization management rules can be enforced at NDC vs class level?*
*   *How does Smart PA use ICD‑10 and claim history together?*
*   *What is “gold carding” and what governance is required to maintain it?*
*   *How are emergency fills handled for PA, step therapy, or non‑formulary rejects?*

✅ Best for: **Clinical governance, UM strategy, HEDIS improvement**

***

## 6. Controlled Substances & Opioid Management

Strong content for high‑risk categories:

*   *How are DEA class II–V drugs restricted by refill, days‑to‑fill, and quantity rules?*
*   *How does Express Scripts enforce state‑specific controlled substance laws?*
*   *What happens when state rules are stricter than federal law?*
*   *How does Advanced Opioid Management intervene at prescriber, pharmacy, and patient levels?*

✅ Best for: **Risk management, state audits, opioid mitigation programs**

***

## 7. Pharmacy Network & 340B Questions

Ideal for **network strategy and pricing**:

*   *How are pharmacies validated for Medicaid FFS enrollment?*
*   *What exceptions exist for non‑enrolled or out‑of‑state pharmacies?*
*   *How are 340B claims identified using NCPDP fields?*
*   *How does BOCD 08 vs SCC 20 impact claim pricing?*
*   *Can 340B claims bypass utilization edits like step therapy?*

✅ Best for: **Network design, pricing strategy, 340B compliance**

***

## 8. Encounters & State Reporting Questions

Critical if you’re responsible for **state acceptance rates**:

*   *What’s the difference between Encounter File Service and Encounter Advisor Service?*
*   *How are rejected encounters analyzed and remediated?*
*   *Which claim dates (fill, adjudication, check) drive encounter extraction?*
*   *How are encounter rules updated when state formats change?*

✅ Best for: **State reporting teams, managed care contracts**

***

## 9. Operational Ownership Questions (Very Useful)

These questions clarify **who does what**:

*   *Which responsibilities belong to Express Scripts vs the plan sponsor?*
*   *What inputs must the plan provide to stay compliant?*
*   *Which items require explicit plan approval before configuration?*
*   *What data must be provided during implementation vs ongoing operations?*
