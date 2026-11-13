# Curriculum Tasks & Implementation Log (Week 3 to 12)

| Metadata | Description |
|---|---|
| **Project** | Legal Contract Intelligence (RAG Application) |
| **Domain / Track** | Track F: Legal Contracts Intelligence (contract Q&A, clause lookup, amendment-aware responses) |
| **Tech Stack (actual)** | Python, Streamlit (`app.py`), Groq LLM (`core/llm_client.py`), `sentence-transformers` embeddings (`BAAI/bge-base-en-v1.5`), ChromaDB, BM25 (`rank-bm25`), Cross-Encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`), PyMuPDF/PDF ingestion, MCP HTTP client (`mcp/client.py`), pytest, optional Langfuse |
| **Observability** | Structured JSON logging, request/classification logs, trace IDs (`core/logging_setup.py`, `core/request_logger.py`, `core/observability.py`, `data/logs/*`) |
| **Primary Branch Reviewed** | `documents` (HEAD `fb57c88`, workspace `D:\vins\rag_project_app`) |

> **Week 12 status:** **Completed and available in the current branch (`documents`)**. Week 12 report artifacts and packaging are present in this repository (`reports/01_application_architecture.md`, `reports/02_eval_dataset.md`, `reports/03_eval_results.md`, `reports/04_known_weaknesses.md`, commit `f8662ba week 12 reports`, with follow-up docs/tests/observability updates on `documents`). Other branches were also reviewed to trace task implementation history.

---

## Week 3 to 12 Implementation Matrix

| week | task | implemented | if choice oriented which branch implemented the research |
| :---: | :--- | :---: | :--- |
| **Week 3** | **Retrieval & RAG (Build Week):** Build a mini “ask my documents” app for contract PDFs with chunking, embeddings, retrieval, grounded generation, source-backed answers, and safe handling when information is not found. | **Yes** | **Research/experimentation:** `feature-hybrid-approch-rpf-failed-one` (early retrieval/hybrid experiments: `bae8174 hybrid`, `a33cc0f failed-one-wont woek properly retrivel problem`). **Implemented and evolved in:** `week6`, `production-grade-application`, `restructuring-branch`, `whole-doc-implementation`, `documents`. |
| **Week 4** | **Debugging Retrieval — Hybrid, Reranking & Failure Separation:** Improve retrieval quality with keyword + semantic search, RRF/hybrid retrieval, cross-encoder reranking, MMR diversification, and retrieval-focused debugging/improvements. | **Yes** | **Research/choice evidence:** `feature-hybrid-approch-rpf-failed-one` (hybrid approach exploration), `week6` (`2522c6c cross - encoder added`, `4680c55 added mmr and implemented the diversity`, `0aac805 feature / restructure the project`). **Current implementation:** `retrieval/hybrid_search.py`, `retrieval/bm25_search.py`, `retrieval/mmr.py`, `reranking/reranker.py`. |
| **Week 5** | **Error Analysis — Traces & Taxonomy:** Read real traces/outputs, identify failure patterns, classify error modes, and prioritize fixes based on observed issues. | **Yes** | **Implemented through logging, evaluation iteration, and reporting across branches:** `restructuring-branch` (`b4473e1 implemented industry level logging`), `whole-doc-implementation` (`3d4f3ff completed the week 11 by logging the things`), `week-11` (eval/judge iteration commits), `documents` (`6941576 logging setuo`, `ee42d16 observability implemented`). **Documented evidence:** `core/logging_setup.py`, `core/request_logger.py`, `reports/04_known_weaknesses.md`. |
| **Week 6** | **Evals — Measuring Improvement (Set F):** Build and run an evaluation workflow (golden set/dataset), assess answer quality, iterate prompts/judge logic, and track improvements with reproducible commands and results. | **Yes** | **Research/implementation branches:** `week6` (`6a2f32f evaluation basic implementation`), `week-11` (`4861118 feature golden set evaluation...`, `06099c1 60 percent accuracy during evaluation`, `0472410`, `bdf37fa`, `7a57112 72 to 8.7 incremented`). **Artifacts:** `evals/golden_set.json`, `evals/dataset.json`, `evals/run_eval.py`, judge/result files in `evals/`, and `reports/02_eval_dataset.md`, `reports/03_eval_results.md`. |
| **Week 7** | **Agent Loops & Tool-Oriented Workflow (Set F):** Implement agent/tool-calling behavior for contract tasks, support routed execution paths, and compare/validate workflow choices through iterative development. | **Yes** | **Research/implementation branches:** `week7` (`640a405 tools workflows`), `production-grade-application` (`7f92704 implemented the classifier agent`), `whole-doc-implementation` (`e25e216 implemented whole doc`). **Current implementation:** `router/classifier.py`, `router/agent.py`, `router/doc_loader.py`, `mcp/client.py`, `pipeline/rag.py`. |
| **Week 8** | **Agent Failure Modes & Reliability Improvements (Set F):** Strengthen agent behavior by handling failure cases, improving routing/tool execution paths, and refining reliability through iterative fixes and trace-informed changes. | **Yes** | **Branch evidence:** `week7` history includes `f4e0f86 week 8 task` and `433c2da week 8 rest of the files`; improvements carried forward in `restructuring-branch`, `whole-doc-implementation`, `week-11`, and `documents`. **Supporting implementation:** structured logging/traceability and routed agent flows in `router/agent.py`, `core/request_logger.py`, `core/logging_setup.py`. |
| **Week 9** | **MCP — Standard Tool/Data Integration (Set F):** Integrate MCP-based capabilities with runtime tool discovery and config-driven connectivity so the agent can use external contract/standards tools. | **Yes** | **Implementation branch:** `origin/week9` (`42a7660 week 9 completed mcp server integration`, merged through later branch history). **Current implementation:** `mcp/client.py` (dynamic tool discovery/calls), MCP route handling in `router/agent.py` and `router/classifier.py`, `MCP_URL` configuration in `.env.example`/`core/config.py`/`README.md`. |
| **Week 10** | **Multi-Step / Multi-Route Orchestration with Evidence (Set F):** Extend the system beyond single-path RAG to routed/orchestrated flows (RAG, MCP, combined, full-document paths) and validate behavior through evaluation and runtime checks. | **Yes** | **Implementation branch evidence:** `origin/week10` (`61f8c53 week 10 race completed`, merged in history). **Current implementation:** route classification and orchestration in `router/classifier.py`, `router/agent.py`, full-doc handling in `router/doc_loader.py`, and RAG/MCP integration across `pipeline/` and `mcp/`. |
| **Week 11** | **Production — Observability, Cost Awareness & Failure→Test Loop (Set F):** Add production-style logging/tracing, improve diagnosability, iterate on eval failures, and maintain tests/reports for reliability and regression tracking. | **Yes** | **Implementation branches:** `whole-doc-implementation` (`3d4f3ff completed the week 11 by logging the things`), `restructuring-branch` (`b4473e1 implemented industry level logging`), `week-11` (eval/prompt/judge improvements), `documents` (`ee42d16 observability implemented`, `d469f7f test suite added using pytest`). **Artifacts:** `core/logging_setup.py`, `core/request_logger.py`, `core/observability.py`, `tests/`, `evals/`, `reports/03_eval_results.md`, `reports/04_known_weaknesses.md`. |
| **Week 12** | **Capstone — Final Reports, Documentation & Packaging:** Consolidate architecture, dataset/eval results, known weaknesses, documentation, and readiness artifacts for final review/demo. | **Yes (Completed, in current branch `documents`)** | **Completed on current branch (`documents`):** `f8662ba week 12 reports`, `df3b98b added docurization`, `3c5d990 readme file added`, `fb57c88 requirement text updated`, plus supporting test/observability/UI/MCP commits. Cross-branch review performed across `week6`, `week7`, `week9`, `week10`, `week-11`, `whole-doc-implementation`, `restructuring-branch`, and `feature-hybrid-approch-rpf-failed-one`. |

---

## Detailed Task Breakdown by Week

### Week 3 — Document Ingestion & Core RAG Foundation
- **Task focus:** Build a contract-focused “ask my documents” app with grounded responses and citations.
- **Implementation highlights:**
  - Streamlit UI for PDF upload, indexing, and chat (`app.py`, `ui/chat.py`, `ui/history.py`)
  - PDF/document ingestion and chunking (`ingestion/pdf_loader.py`, `ingestion/chunker.py`)
  - Embeddings, vector storage, and BM25 retrieval foundations (`retrieval/embeddings.py`, `retrieval/vector_store.py`, `retrieval/bm25_search.py`)
  - RAG response generation with prompts and guardrails (`pipeline/rag.py`, `generation/generator.py`, `generation/prompts.py`, `guardrails/checker.py`)
- **Research / choice evidence:** Early retrieval/hybrid exploration in `feature-hybrid-approch-rpf-failed-one`.

### Week 4 — Hybrid Retrieval, Reranking & Result Quality
- **Task focus:** Improve retrieval quality and separate retrieval behavior from generation outcomes.
- **Implementation highlights:**
  - Hybrid retrieval combining semantic and keyword signals (`retrieval/hybrid_search.py`)
  - MMR diversification (`retrieval/mmr.py`)
  - Cross-encoder reranking (`reranking/reranker.py`)
  - Retrieval pipeline integration for higher-quality context selection
- **Research / choice evidence:** `feature-hybrid-approch-rpf-failed-one` (hybrid attempt) and `week6` (MMR/cross-encoder implementation).

### Week 5 — Trace Reading, Error Analysis & Prioritization
- **Task focus:** Understand real failure patterns and convert observations into fix priorities.
- **Implementation highlights:**
  - Structured logging and request traceability (`core/logging_setup.py`, `core/request_logger.py`)
  - Classification/request logging in routed flows (`router/agent.py`, `router/classifier.py`)
  - Evaluation-driven iteration and documented weakness taxonomy (`evals/`, `reports/04_known_weaknesses.md`)
- **Notes:** Findings and priorities are captured through logs, eval iterations, and report artifacts used in later fixes.

### Week 6 — Evaluation Workflow & Quality Measurement
- **Task focus:** Establish a repeatable evaluation process and improve answer quality with measurable results.
- **Implementation highlights:**
  - Golden set/dataset and eval runner (`evals/golden_set.json`, `evals/dataset.json`, `evals/run_eval.py`)
  - Judge prompts/results and iterative score improvements (`evals/judge_v1.txt`, `evals/results_v1.json`, `evals/results_v2.json`, `week-11` commits)
  - Evaluation documentation (`reports/02_eval_dataset.md`, `reports/03_eval_results.md`)
- **Research / choice evidence:** `week6` and `week-11` branches show progressive evaluation and judge/prompt refinement.

### Week 7 — Agent/Tool Routing & Workflow Execution
- **Task focus:** Move from single-path answering to routed, tool-aware execution for different contract question types.
- **Implementation highlights:**
  - Query classification and route dispatch (`router/classifier.py`, `router/agent.py`)
  - RAG, MCP, combined, and full-document handling paths (`pipeline/rag.py`, `router/doc_loader.py`, `mcp/client.py`)
  - Agent/tool workflow evolution from branch history into the current structured app
- **Research / choice evidence:** `week7` (`tools workflows`) and `production-grade-application` (classifier agent).

### Week 8 — Reliability & Failure-Mode Hardening
- **Task focus:** Improve robustness of agent/routing behavior based on observed failures and execution patterns.
- **Implementation highlights:**
  - Week 8 task commits preserved in branch history (`f4e0f86`, `433c2da`)
  - Reliability improvements carried into structured routing/logging architecture (`router/agent.py`, `core/request_logger.py`, `core/logging_setup.py`)
  - Iterative fixes across MCP/routing/application refactor branches
- **Notes:** Reliability work is reflected in the integrated codebase and commit history, with traceability supported by logs and eval feedback loops.

### Week 9 — MCP Integration (Standardized Tool Access)
- **Task focus:** Enable standardized external tool/data access through MCP.
- **Implementation highlights:**
  - MCP HTTP client with runtime tool discovery and calls (`mcp/client.py`)
  - MCP-aware routing and execution in agent flows (`router/agent.py`, `router/classifier.py`)
  - Config-driven MCP endpoint support (`core/config.py`, `.env.example`, `README.md`)
  - External MCP server integration reference: `https://github.com/vinslin/contarct_mcp_server`
- **Research / choice evidence:** `origin/week9` (`42a7660 week 9 completed mcp server integration`).

### Week 10 — Multi-Route Orchestration & Evidence-Based Operation
- **Task focus:** Orchestrate multiple execution paths (RAG/MCP/full-doc/combined) and validate behavior with practical evidence.
- **Implementation highlights:**
  - Route-based orchestration and decision logic (`router/classifier.py`, `router/agent.py`)
  - Combined retrieval + MCP and full-document analysis flows
  - Continued integration with evaluation/logging for validation and debugging
- **Research / choice evidence:** `origin/week10` (`61f8c53 week 10 race completed`) with integration reflected in later branches.

### Week 11 — Production Observability & Failure→Test Loop
- **Task focus:** Improve diagnosability, traceability, and regression prevention in a production-oriented workflow.
- **Implementation highlights:**
  - Structured logging, trace IDs, request/classification logs (`core/logging_setup.py`, `core/request_logger.py`)
  - Optional observability integration (`core/observability.py`)
  - Test suite and evaluation loop for regression checks (`tests/`, `evals/run_eval.py`)
  - Failure taxonomy and improvement documentation (`reports/03_eval_results.md`, `reports/04_known_weaknesses.md`)
- **Research / choice evidence:** Logging/observability and eval-iteration work across `restructuring-branch`, `whole-doc-implementation`, `week-11`, and `documents`.

### Week 12 — Capstone Completion (Reports & Packaging)
- **Task focus:** Finalize architecture/evaluation/weakness documentation and package the project for review/demo.
- **Implementation highlights (completed):**
  - Capstone reports: `reports/01_application_architecture.md`, `reports/02_eval_dataset.md`, `reports/03_eval_results.md`, `reports/04_known_weaknesses.md`
  - Documentation and setup updates (`README.md`, `.env.example`, requirements)
  - Supporting quality and runtime improvements (tests, observability, UI/MCP fixes)
- **Completion evidence:** `f8662ba week 12 reports` and subsequent commits on `documents` branch.

---

## Cross-Branch Evidence Reviewed

- `feature-hybrid-approch-rpf-failed-one` — early hybrid/RRF and retrieval research attempts
- `week6` — MMR, cross-encoder reranking, evaluation groundwork
- `week7` — tools/workflow and Week 8 task commits in history (`f4e0f86`, `433c2da`)
- `origin/week9` — MCP integration (`42a7660`)
- `origin/week10` — Week 10 completion/race evidence (`61f8c53`)
- `week-11` — golden set/judge/eval iteration improvements
- `whole-doc-implementation` — whole-doc handling and Week 11 logging completion
- `restructuring-branch` — structured refactor and industry-level logging
- `documents` (current) — Week 12 reports, docs, tests, observability, and final packaging updates

## Simple Notes (Polite Clarifications)

- Week 12 is fully completed in the current repository with the capstone report set and supporting documentation.
- For some earlier curriculum milestones, a few week-specific submission files (such as separately named race logs/tables or standalone trace worksheets) may be stored in other branch contexts or captured within integrated code/log/report artifacts; the implementation history and outcomes are reflected in the branches and repository evidence reviewed above.
- This report is grounded in the **actual codebase and git history** of this project (Streamlit/Groq/Chroma/BM25/MCP architecture), and all branch references are included to help trace where each task/research was implemented.