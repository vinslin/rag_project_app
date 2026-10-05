# 10x Query Volume — What Breaks First

**At 10x today's query volume, the Groq rate limit breaks first.**

---

## Numbers

| Metric | Current (1x) | Projected (10x) | Groq Free-Tier Limit |
|---|---|---|---|
| Queries / day | ~20 | ~200 | 14,400 req/day |
| Queries / minute (peak) | ~2 | ~20 | **30 req/min** |
| Tokens / query | ~2,159 | ~2,159 | — |
| Daily token spend | ~43,180 | ~431,800 | 131,072 tokens/day |
| Daily cost (paid tier) | ~$0.007 | ~$0.07 | — |

**The number that proves it:** Groq free tier caps at **30 req/min** and **131,072 tokens/day**.
At 10x volume (~200 req/day, ~20 req/min peak), the daily token limit
(~431,800 tokens needed vs 131,072 allowed) is exceeded before the request-per-minute
cap is hit during normal usage. Either constraint fires before cost becomes meaningful
($0.07/day on the paid tier is negligible).

---

## What to Do at 10x

1. **Upgrade to Groq paid tier** — removes the daily token cap.
2. **Add semantic caching** — identical or near-identical queries skip the LLM call
   entirely; estimated 30–40% cache-hit rate on FAQ-style legal questions.
3. **Route cheap questions to a smaller model** (e.g. Llama 3 8B) — saves ~80% on
   generation cost for guardrail-adjacent or simple lookup queries.
