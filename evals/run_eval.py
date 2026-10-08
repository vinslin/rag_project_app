#!/usr/bin/env python3
"""
Evaluation Runner — Shared Corpus Golden Set (32 cases)

Reads evals/golden_set.json and runs every case through the RAG pipeline.
Reports pass rate by class:
    direct_lookup, amendment_supersession, counterparty_disambiguation,
    out_of_scope, defined_term_chase, multi_hop_dependent, computation,
    version_comparison, cross_document

Failure breakdown:
    fail_wrong_version, fail_wrong_counterparty,
    fail_hallucination, fail_false_refusal

Usage:
    python evals/run_eval.py              # use cached predictions
    python evals/run_eval.py --recompute  # regenerate all predictions
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH    = os.path.join(BASE_DIR, "golden_set.json")
PREDICTION_PATH = os.path.join(BASE_DIR, "prediction.txt")
JUDGE_V1_PATH   = os.path.join(BASE_DIR, "judge_v1.txt")
JUDGE_V2_PATH   = os.path.join(BASE_DIR, "judge_v2.txt")
RESULTS_V1_PATH = os.path.join(BASE_DIR, "results_v1.json")
RESULTS_V2_PATH = os.path.join(BASE_DIR, "results_v2.json")

APP_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

try:
    from pipeline.rag import answer_question
except ImportError:
    answer_question = None


# ── Index auto-build ──────────────────────────────────────────────────────────

def ensure_index_built():
    """Index all documents in documents/ if ChromaDB is empty."""
    import glob
    try:
        from ingestion.pdf_loader import load_document
        from ingestion.chunker import chunk_document
        from retrieval.vector_store import build_index, clear_index, chroma_client
        from core import config

        docs_dir  = os.path.join(APP_ROOT, "documents")
        doc_files = sorted(
            glob.glob(os.path.join(docs_dir, "*.pdf")) +
            glob.glob(os.path.join(docs_dir, "*.docx")) +
            glob.glob(os.path.join(docs_dir, "*.md"))
        )
        if not doc_files:
            print("Warning: No documents found in documents/ folder.")
            return

        try:
            col = chroma_client.get_collection(config.COLLECTION_NAME)
            if col.count() > 0:
                return
        except Exception:
            pass

        print(f"Indexing {len(doc_files)} document(s) from documents/...")
        clear_index(config.COLLECTION_NAME)
        total_chunks = 0
        for doc_path in doc_files:
            pages = load_document(doc_path)
            for page in pages:
                meta = {k: page.get(k, "") for k in
                        ["source_doc", "counterparty", "effective_date", "doc_type"]}
                chunks = chunk_document(page["text"], metadata=meta)
                build_index(chunks, config.COLLECTION_NAME,
                            source=page["source"], page=page["page"])
                total_chunks += len(chunks)
        print(f"Indexed {total_chunks} chunks from {len(doc_files)} document(s).")
    except Exception as e:
        print(f"Warning: Could not auto-build index: {e}")


# ── Prediction generation ─────────────────────────────────────────────────────

def run_pipeline_predictions(dataset: list) -> dict:
    """Run every case through the RAG pipeline; return {id: {question, prediction}}."""
    import time
    ensure_index_built()
    print("Generating pipeline predictions for all cases...")
    predictions = {}

    for i, case in enumerate(dataset):
        case_id  = case["id"]
        question = case["question"]
        if answer_question:
            try:
                res    = answer_question(question)
                answer = res.get("answer", "")
            except Exception as e:
                answer = f"Error: {e}"
        else:
            answer = "Pipeline unavailable."

        predictions[case_id] = {"question": question, "prediction": answer}
        print(f"  [{i+1}/{len(dataset)}] {case_id} done")
        if i < len(dataset) - 1:
            time.sleep(180)  # 3 min gap — lets Groq 200k/day rolling window recover

    with open(PREDICTION_PATH, "w", encoding="utf-8") as fh:
        for cid, pdata in predictions.items():
            fh.write(json.dumps({
                "id": cid,
                "question": pdata["question"],
                "predicted_answer": pdata["prediction"],
            }, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {PREDICTION_PATH}")
    return predictions


def load_predictions(dataset: list) -> dict:
    """Load cached predictions or run pipeline if cache is missing."""
    if not os.path.exists(PREDICTION_PATH):
        return run_pipeline_predictions(dataset)

    predictions = {}
    with open(PREDICTION_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                predictions[item["id"]] = {
                    "question":   item.get("question", ""),
                    "prediction": item.get("predicted_answer", ""),
                }
            except Exception:
                continue

    missing = [c["id"] for c in dataset if c["id"] not in predictions]
    if missing:
        print(f"Missing predictions for {len(missing)} case(s). Recomputing...")
        return run_pipeline_predictions(dataset)

    return predictions


# ── LLM judge ─────────────────────────────────────────────────────────────────

def _clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].startswith("```") else lines
        raw = "\n".join(lines).strip()
    return raw


def get_groq_client():
    """Return a Groq client if GROQ_API_KEY is set, else None."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except Exception as e:
        print(f"Warning: Could not init Groq client: {e}")
        return None


def evaluate_case(client, judge_template: str, case: dict,
                  predicted_answer: str) -> dict:
    """Evaluate one case. Uses Groq LLM judge if available, else deterministic."""
    prompt = (
        judge_template
        .replace("{question}",        case["question"])
        .replace("{expected_answer}", case["expected_answer"])
        .replace("{predicted_answer}", predicted_answer)
        .replace("{taxonomy}",        case.get("class", "unassigned"))
    )

    if client:
        try:
            from core import config
            resp = client.chat.completions.create(
                model=config.GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text   = _clean_json(resp.choices[0].message.content)
            parsed = json.loads(text)
            return {
                "status":         parsed.get("status", "FAIL").upper(),
                "score":          float(parsed.get("score", 0.0)),
                "reasoning":      parsed.get("reasoning", "LLM judge"),
                "error_category": parsed.get("error_category"),
            }
        except Exception:
            pass  # fall through to deterministic

    # ── Deterministic fallback ────────────────────────────────────────────────
    pred_lower = predicted_answer.lower()
    refusal_phrases = [
        "could not find", "not in the corpus", "no such", "not exist",
        "no apex", "no amendments", "no separate",
    ]

    if case.get("refusal_expected"):
        passed = any(p in pred_lower for p in refusal_phrases)
        error  = None if passed else "fail_false_refusal"
    else:
        expected_values = case.get("expected_values", [])
        if expected_values:
            passed = all(v.lower() in pred_lower for v in expected_values)
        else:
            passed = case["expected_answer"].lower() in pred_lower
        if not passed:
            # Classify failure type
            sup = (case.get("superseded_value") or "").lower()
            dis = (case.get("distractor_value") or "").lower()
            if sup and any(w in pred_lower for w in sup.split() if len(w) > 3):
                error = "fail_wrong_version"
            elif dis and any(w in pred_lower for w in dis.split() if len(w) > 3):
                error = "fail_wrong_counterparty"
            else:
                error = "fail_hallucination"
        else:
            error = None

    return {
        "status":         "PASS" if passed else "FAIL",
        "score":          1.0 if passed else 0.0,
        "reasoning":      "Deterministic fallback.",
        "error_category": error,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Legal RAG Golden Set Evaluation")
    parser.add_argument("--judge",     choices=["v1", "v2"], default="v1")
    parser.add_argument("--recompute", action="store_true",
                        help="Regenerate all predictions (ignores cache)")
    args = parser.parse_args()

    print(f"=== Legal RAG Evaluation — Golden Set (Judge: {args.judge}) ===\n")

    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        sys.exit(1)

    with open(DATASET_PATH, encoding="utf-8") as fh:
        raw = json.load(fh)
    dataset = raw["cases"]   # golden_set.json wraps cases under "cases" key
    print(f"Loaded {len(dataset)} cases from golden_set.json")

    template_path = JUDGE_V1_PATH if args.judge == "v1" else JUDGE_V2_PATH
    if not os.path.exists(template_path):
        print(f"Warning: Judge template not found at {template_path}. Using deterministic fallback.")
        judge_template = ""
    else:
        with open(template_path, encoding="utf-8") as fh:
            judge_template = fh.read()

    predictions = run_pipeline_predictions(dataset) if args.recompute \
        else load_predictions(dataset)

    client = get_groq_client()
    if client:
        print("Using Groq LLM judge.")
    else:
        print("GROQ_API_KEY not set — using deterministic fallback judge.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    eval_results   = []
    class_stats    = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
    failure_counts = defaultdict(int)
    total_passed   = 0

    print("\nEvaluating cases...\n")
    for case in dataset:
        case_id   = case["id"]
        cls       = case.get("class", "unassigned")
        pred_data = predictions.get(case_id, {})
        pred      = pred_data.get("prediction", "")

        if pred.startswith("Error:"):
            result = {
                "status": "FAIL",
                "score": 0.0,
                "reasoning": f"Pipeline error: {pred[:200]}",
                "error_category": "pipeline_error",
            }
        else:
            result = evaluate_case(client, judge_template, case, pred)
        passed  = result["status"] == "PASS"
        err_cat = result.get("error_category")

        if passed:
            total_passed += 1
            class_stats[cls]["passed"] += 1
        else:
            class_stats[cls]["failed"] += 1
            if err_cat:
                failure_counts[err_cat] += 1

        class_stats[cls]["total"] += 1

        eval_results.append({
            "id":               case_id,
            "class":            cls,
            "question":         case["question"],
            "expected_answer":  case["expected_answer"],
            "predicted_answer": pred,
            "status":           result["status"],
            "score":            result["score"],
            "reasoning":        result["reasoning"],
            "error_category":   err_cat,
            "source_doc":       case.get("source_doc"),
            "refusal_expected": case.get("refusal_expected", False),
        })

    total_cases       = len(dataset)
    overall_pass_rate = (total_passed / total_cases * 100.0) if total_cases else 0.0

    # ── Save results ──────────────────────────────────────────────────────────
    class_breakdown = {}
    for cls, stats in class_stats.items():
        tot  = stats["total"]
        pas  = stats["passed"]
        rate = (pas / tot * 100.0) if tot else 0.0
        class_breakdown[cls] = {
            "total":            tot,
            "passed":           pas,
            "failed":           stats["failed"],
            "pass_rate_percent": round(rate, 2),
        }

    output = {
        "judge_version":            args.judge,
        "total_cases":              total_cases,
        "total_passed":             total_passed,
        "total_failed":             total_cases - total_passed,
        "overall_pass_rate_percent": round(overall_pass_rate, 2),
        "pass_rate_by_class":       class_breakdown,
        "failure_breakdown":        dict(failure_counts),
        "cases":                    eval_results,
    }

    results_path = RESULTS_V1_PATH if args.judge == "v1" else RESULTS_V2_PATH
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"Results saved to {results_path}\n")

    # ── Console report ────────────────────────────────────────────────────────
    W = 62
    print("=" * W)
    print(f"  GOLDEN SET EVALUATION REPORT  (Judge: {args.judge})")
    print("=" * W)
    print(f"  Total cases : {total_cases}")
    print(f"  Passed      : {total_passed}")
    print(f"  Failed      : {total_cases - total_passed}")
    print(f"  Pass rate   : {overall_pass_rate:.1f}%")
    print("-" * W)
    print(f"  {'CLASS':<30}  {'TOT':>3}  {'PASS':>4}  {'RATE':>7}")
    print("-" * W)

    CLASS_ORDER = [
        "direct_lookup", "amendment_supersession", "counterparty_disambiguation",
        "out_of_scope", "defined_term_chase", "multi_hop_dependent",
        "computation", "version_comparison", "cross_document",
    ]
    for cls in CLASS_ORDER:
        if cls not in class_breakdown:
            continue
        s = class_breakdown[cls]
        bar = "#" * s["passed"] + "." * s["failed"]
        print(f"  {cls:<30}  {s['total']:>3}  {s['passed']:>4}  {s['pass_rate_percent']:>6.1f}%  {bar}")

    if failure_counts:
        print("-" * W)
        print("  Failure breakdown:")
        for cat, cnt in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"    {cat:<35} {cnt}")

    print("=" * W + "\n")


if __name__ == "__main__":
    main()
