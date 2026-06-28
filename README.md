# HireRank AI

Explainable offline hybrid ranker for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## Why this approach

The challenge rewards top-100 candidate ranking quality and requires the ranking code to run CPU-only, offline, without hosted LLM/API calls. HireRank AI therefore uses a streaming hybrid scoring engine instead of per-candidate LLM calls.

## Pipeline

1. Stream `candidates.jsonl` without loading the full file into memory.
2. Score each candidate using:
   - semantic/domain keyword evidence
   - core AI/retrieval/vector database skill coverage
   - production ML/ranking evidence
   - seniority and experience fit
   - behavioral Redrob signals
   - location/relocation fit
   - trap/disqualifier penalties
3. Keep only the top 100 candidates using a heap.
4. Generate a submission CSV with candidate_id, rank, score, and evidence-based reasoning.

## Run

```bash
python src/rank_candidates.py --candidates /path/to/candidates.jsonl --output outputs/team_xxx.csv
python validate_submission.py outputs/team_xxx.csv
```

## Files

```text
src/rank_candidates.py       Main offline ranking engine
outputs/team_xxx.csv         Generated top-100 submission
validate_submission.py       Official validator
submission_metadata_template.yaml
```

## Design notes

- No network calls during ranking.
- No GPU required.
- Pure Python standard library for reproducibility.
- Deterministic tie-breaking by candidate_id.
- Reasoning strings are grounded in candidate features and include concerns where applicable.

## Scoring weights

Final score combines semantic fit, skills, production experience, experience fit, behavior, and location. A penalty layer reduces candidates with weak domain fit, non-production-only AI exposure, keyword-stuffing patterns, very low availability, or long notice periods.
