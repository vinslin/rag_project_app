#!/usr/bin/env python3
"""
Evaluation Runner for Legal Contract RAG

Runs evaluation over evals/dataset.json using LLM Judge (judge_v1.txt or judge_v2.txt).
Prints overall pass rate and pass rates broken down by Week 5 taxonomy mode.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.json")
LABELS_PATH = os.path.join(BASE_DIR, "labels_25.json")
PREDICTION_PATH = os.path.join(BASE_DIR, "prediction.txt")
JUDGE_V1_PATH = os.path.join(BASE_DIR, "judge_v1.txt")
JUDGE_V2_PATH = os.path.join(BASE_DIR, "judge_v2.txt")
RESULTS_V1_PATH = os.path.join(BASE_DIR, "results_v1.json")
RESULTS_V2_PATH = os.path.join(BASE_DIR, "results_v2.json")

# Ensure rag package is in python path
APP_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

try:
    from rag.pipeline import answer_question
except ImportError:
    answer_question = None


def get_genai_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Warning: Failed to initialize genai Client: {e}")
        return None


def ensure_index_built():
    """Ensure ChromaDB & BM25 index is built from documents/*.pdf if empty."""
    import glob
    try:
        from rag.ingestion.pdf_loader import load_pdf
        from rag.ingestion.chunker import chunk_document
        from rag.retrieval.vector_store import build_index, clear_index, chroma_client
        from rag import config

        docs_dir = os.path.join(APP_ROOT, "documents")
        pdf_files = sorted(glob.glob(os.path.join(docs_dir, "*.pdf")))
        if not pdf_files:
            print("Warning: No PDF files found in documents/ folder.")
            return

        try:
            col = chroma_client.get_collection(config.COLLECTION_NAME)
            if col.count() > 0:
                return
        except Exception:
            pass

        print(f"Indexing {len(pdf_files)} PDF(s) from documents/...")
        clear_index(config.COLLECTION_NAME)
        all_chunks = []
        for pdf_path in pdf_files:
            pages = load_pdf(pdf_path)
            for page in pages:
                chunks = chunk_document(page["text"], config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS)
                build_index(chunks, config.COLLECTION_NAME, source=page["source"], page=page["page"])
                all_chunks.extend(chunks)
        print(f"Successfully indexed {len(all_chunks)} chunks.")
    except Exception as e:
        print(f"Warning: Could not auto-build index: {e}")


def run_pipeline_predictions(dataset):
    """Run RAG pipeline over dataset questions and return dict of predictions."""
    ensure_index_built()
    print("Generating pipeline predictions for dataset...")
    predictions = {}
    
    for case in dataset:
        case_id = case["id"]
        question = case["question"]
        if answer_question:
            try:
                res = answer_question(question)
                answer = res.get("answer", "")
            except Exception as e:
                answer = f"Error generating answer: {e}"
        else:
            answer = "Pipeline unavailable."
        
        predictions[case_id] = {
            "question": question,
            "prediction": answer
        }
    
    # Save to prediction.txt as formatted text / JSONL
    with open(PREDICTION_PATH, "w", encoding="utf-8") as f:
        for cid, pdata in predictions.items():
            f.write(json.dumps({"id": cid, "question": pdata["question"], "predicted_answer": pdata["prediction"]}, ensure_ascii=False) + "\n")
            
    print(f"Saved predictions to {PREDICTION_PATH}")
    return predictions


def load_predictions(dataset):
    """Load existing predictions from prediction.txt or run pipeline if missing."""
    if not os.path.exists(PREDICTION_PATH):
        return run_pipeline_predictions(dataset)
    
    predictions = {}
    with open(PREDICTION_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                predictions[item["id"]] = {
                    "question": item.get("question", ""),
                    "prediction": item.get("predicted_answer", "")
                }
            except Exception:
                continue
                
    # If any case is missing, recompute
    missing = [c["id"] for c in dataset if c["id"] not in predictions]
    if missing:
        print(f"Missing predictions for {len(missing)} cases. Recomputing...")
        return run_pipeline_predictions(dataset)
        
    return predictions


def clean_json_response(raw_text):
    """Clean markdown backticks from model output before parsing JSON."""
    raw = raw_text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def evaluate_case_with_llm(client, judge_template, case, predicted_answer, judge_version="v1"):
    """Evaluate a single test case using Gemini LLM judge."""
    prompt = (
        judge_template
        .replace("{question}", case["question"])
        .replace("{expected_answer}", case["expected_answer"])
        .replace("{predicted_answer}", predicted_answer)
        .replace("{taxonomy}", case.get("taxonomy", "unassigned"))
    )

    if client:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            text = clean_json_response(response.text)
            parsed = json.loads(text)
            return {
                "status": parsed.get("status", "FAIL").upper(),
                "score": float(parsed.get("score", 0.0)),
                "reasoning": parsed.get("reasoning", "Evaluated by LLM Judge"),
                "error_category": parsed.get("error_category", None)
            }
        except Exception as e:
            # Deterministic fallback evaluation if JSON parsing or API fails
            pass

    # Deterministic rule-based fallback judge logic
    exp_lower = case["expected_answer"].lower()
    pred_lower = predicted_answer.lower()
    tax = case.get("taxonomy", "")

    if tax == "guardrail_trigger":
        passed = ("rejected" in pred_lower or "out of scope" in pred_lower or "cannot" in pred_lower or "not able" in pred_lower or "prohibited" in pred_lower)
    elif tax == "amendment_override_failure":
        # Check key terms for overrides
        if "15" in exp_lower:
            passed = ("15 days" in pred_lower or "fifteen" in pred_lower)
        elif "45" in exp_lower:
            passed = ("45" in pred_lower or "forty-five" in pred_lower)
        else:
            passed = any(w in pred_lower for w in ["amendment", "supersedes", "replaced"])
    else:
        # Key string overlaps
        keywords = [w for w in exp_lower.replace(",", "").replace(".", "").split() if len(w) > 4]
        overlap = sum(1 for k in keywords if k in pred_lower)
        passed = (overlap >= max(1, len(keywords) // 3)) or (exp_lower in pred_lower)

    status = "PASS" if passed else "FAIL"
    score = 1.0 if passed else 0.0
    reasoning = f"Rule-based evaluation fallback (status: {status})."

    return {
        "status": status,
        "score": score,
        "reasoning": reasoning,
        "error_category": None if passed else tax
    }


def main():
    parser = argparse.ArgumentParser(description="Legal RAG Evaluation Harness")
    parser.add_argument("--judge", choices=["v1", "v2"], default="v1", help="Judge version to run (v1 or v2)")
    parser.add_argument("--recompute", action="store_true", help="Recompute pipeline predictions instead of reading cache")
    args = parser.parse_args()

    print(f"=== Running Legal RAG Evaluation (Judge: {args.judge}) ===")

    # Load Dataset
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} evaluation cases from {DATASET_PATH}")

    # Load Judge Prompt Template
    template_path = JUDGE_V1_PATH if args.judge == "v1" else JUDGE_V2_PATH
    with open(template_path, "r", encoding="utf-8") as f:
        judge_template = f.read()

    # Load or generate predictions
    if args.recompute:
        predictions = run_pipeline_predictions(dataset)
    else:
        predictions = load_predictions(dataset)

    # Initialize Gemini client
    client = get_genai_client()
    if client:
        print("Initialized Gemini Client for LLM Judge.")
    else:
        print("Warning: GOOGLE_API_KEY not set or client unavailable. Using fallback deterministic evaluator.")

    # Evaluate cases
    eval_results = []
    taxonomy_stats = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
    total_passed = 0

    print("\nRunning evaluation on test cases...")
    for case in dataset:
        case_id = case["id"]
        tax = case.get("taxonomy", "unassigned")
        pred_data = predictions.get(case_id, {})
        pred_answer = pred_data.get("prediction", "")

        eval_res = evaluate_case_with_llm(client, judge_template, case, pred_answer, judge_version=args.judge)

        status = eval_res["status"]
        passed = (status == "PASS")

        if passed:
            total_passed += 1
            taxonomy_stats[tax]["passed"] += 1
        else:
            taxonomy_stats[tax]["failed"] += 1
        taxonomy_stats[tax]["total"] += 1

        record = {
            "id": case_id,
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "predicted_answer": pred_answer,
            "taxonomy": tax,
            "contract": case.get("contract", "N/A"),
            "status": status,
            "score": eval_res["score"],
            "reasoning": eval_res["reasoning"],
            "error_category": eval_res.get("error_category")
        }
        eval_results.append(record)

    total_cases = len(dataset)
    overall_pass_rate = (total_passed / total_cases * 100.0) if total_cases > 0 else 0.0

    # Summary payload
    taxonomy_breakdown = {}
    for mode, stats in taxonomy_stats.items():
        tot = stats["total"]
        pas = stats["passed"]
        rate = (pas / tot * 100.0) if tot > 0 else 0.0
        taxonomy_breakdown[mode] = {
            "total": tot,
            "passed": pas,
            "failed": stats["failed"],
            "pass_rate_percent": round(rate, 2)
        }

    output_payload = {
        "judge_version": args.judge,
        "total_cases": total_cases,
        "total_passed": total_passed,
        "total_failed": total_cases - total_passed,
        "overall_pass_rate_percent": round(overall_pass_rate, 2),
        "pass_rate_by_mode": taxonomy_breakdown,
        "cases": eval_results
    }

    # Save results json
    results_path = RESULTS_V1_PATH if args.judge == "v1" else RESULTS_V2_PATH
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nSaved evaluation results to {results_path}")

    # Print Clean Console Summary Report
    print("\n" + "=" * 60)
    print(f"       EVALUATION SUMMARY REPORT (Judge: {args.judge})")
    print("=" * 60)
    print(f"Total Test Cases:       {total_cases}")
    print(f"Total Passed:           {total_passed}")
    print(f"Total Failed:           {total_cases - total_passed}")
    print(f"Overall Pass Rate:      {overall_pass_rate:.2f}%\n")
    print("-" * 60)
    print(f"{'TAXONOMY MODE':<30} | {'TOTAL':<6} | {'PASSED':<6} | {'PASS RATE':<10}")
    print("-" * 60)

    for mode in sorted(taxonomy_breakdown.keys()):
        stats = taxonomy_breakdown[mode]
        print(f"{mode:<30} | {stats['total']:<6} | {stats['passed']:<6} | {stats['pass_rate_percent']:.2f}%")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
