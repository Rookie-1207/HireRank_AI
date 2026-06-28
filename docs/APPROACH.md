# Approach

## Objective
Rank the best 100 candidates for the Senior AI Engineer / Founding Team role.

## Key interpretation of the JD
The strongest candidates should show:

- production embeddings / retrieval / ranking experience
- vector database or hybrid search experience
- strong Python and code ownership
- ranking evaluation knowledge such as NDCG, MRR, MAP, A/B testing
- product-engineering bias, not research-only background
- good availability and recruiter engagement signals

## Ranking method

HireRank AI uses a multi-factor hybrid formula:

```text
score =
  semantic/domain fit
+ skill coverage
+ production ML evidence
+ experience fit
+ Redrob behavioral signals
+ location/relocation fit
- trap/disqualifier penalties
```

## Why no hosted LLM
The challenge rules require offline, CPU-only, no-network ranking within the runtime limit. The ranker therefore avoids OpenAI/Gemini/Claude API calls and uses deterministic local scoring.

## Trap handling
The penalty layer reduces candidates with:

- very low or very high experience for the senior IC role
- non-technical current roles
- keyword-stuffed skills but weak production descriptions
- research-only language without deployment evidence
- only generic ChatGPT/LLM usage and no retrieval/ranking depth
- long notice period, poor recent activity, poor recruiter response

## Reasoning generation
The reasoning column is built from actual candidate facts: title, years of experience, matched skills, production evidence, and concerns.
