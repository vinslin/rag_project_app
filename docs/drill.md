# Support Drill — Wrong Termination Clause

**Complaint received:** "A lawyer said it cited the wrong clause on termination, maybe Thursday."

---

## Time-to-Find

| | |
|---|---|
| **Start** | Timer started |
| **Find** | 01:42 |
| **Slice that found it** | `answer` field in `data/logs/requests.jsonl` |

**Search command used:**
```python
from obs.request_logger import search_logs
hits = search_logs("Section 8.2", field="answer")
# OR
hits = search_logs("sixty", field="answer")
```

The wrong answer contained `"sixty (60) days"` and `"Section 8.2"` — both visible
in the indexed `answer` field. Searching the `query` field alone would have found
nothing because the complaint is about the *output*, not the input.

---

## What Was Found

```json
{
  "trace_id": "a3f1c2d4-...",
  "timestamp": "2026-10-03T11:47:22+00:00",
  "prompt_version": "v1.0",
  "route": "rag",
  "query": "According to the termination clause, what notice period must a party give to terminate for convenience?",
  "answer": "According to Section 8.2, either party may terminate for convenience by providing sixty (60) days' written notice.",
  "retrieved_context_ids": ["hybrid_search_legal_contract_test.pdf_p3_4", "hybrid_search_legal_contract_test.pdf_p7_2"]
}
```

**The bug:** Section 8.2 was retrieved and cited. Amendment No. 2 (15 days) was also
in the index but ranked lower — the model cited the base clause without checking for
superseding amendments.

---

## Missing Log Field That Would Have Made the Find Instant

Without the `answer` field indexed, you cannot search by output text.
This drill took **01:42** instead of under 30 seconds because the log had to be
scanned line-by-line. With the `answer` field now present in all log records,
a `grep "sixty" data/logs/requests.jsonl` finds it in milliseconds.
