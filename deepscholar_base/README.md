# DeepScholar Base

DeepScholar Base is our baseline research synthesis pipeline that transforms a research topic into a comprehensive, well-organized literature review with proper citations. Built on top of the [LOTUS framework](https://github.com/lotus-data/lotus) for LLM-powered data processing, it demonstrates a modular approach to automated deep research.

## How It Works

The pipeline takes a **research topic** (and optionally an **end date** to limit searches to papers before a certain date—useful for reproducibility or backdating research) and produces a **Markdown report** with categorized references, summaries, and inline citations.

### Pipeline Overview

```
                                  ┌────────────┐
                                  │   INPUT    │
                                  │   topic    │
                                  │  end_date  │
                                  │  configs   │
                                  └─────┬──────┘
                                        │
                                        ▼
          ┌───────────────────────────────────────────────────────┐
          │                    STEP 1: SEARCH                     │
          │  ┌─────────────────────────────────────────────────┐  │
          │  │         use_agentic_search = True?              │  │
          │  └────────────────────┬────────────────────────────┘  │
          │              ┌────────┴────────┐                      │
          │              ▼                 ▼                      │
          │     ┌──────────────┐   ┌───────────────┐              │
          │     │   Agentic    │   │   Recursive   │              │
          │     │   Search     │   │    Search     │              │
          │     │  (Agent+     │   │  (Multi-step  │              │
          │     │   Tools)     │   │   queries)    │              │
          │     └──────┬───────┘   └───────┬───────┘              │
          │            │                   │                      │
          │            └─────────┬─────────┘                      │
          │                      │                                │
          │                      ▼                                │
          │              ┌──────────────┐                         │
          │              │   docs_df    │   DataFrame with        │
          │              │   queries    │   title, url, snippet,  │
          │              │   background │   date, authors, etc.   │
          │              └──────────────┘                         │
          └──────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
          ┌───────────────────────────────────────────────────────┐
          │                    STEP 2: FILTER                     │
          │  ┌─────────────────────────────────────────────────┐  │
          │  │  if use_sem_filter:                             │  │
          │  │    sem_filter() - Keep only relevant docs       │  │
          │  │                                                 │  │
          │  │  if use_sem_topk:                               │  │
          │  │    sem_topk(K=final_max_results_count)          │  │
          │  │    - Select top K most relevant docs            │  │
          │  └─────────────────────────────────────────────────┘  │
          │                                                       │
          │  If docs_df.empty → Retry search (max_search_retries) │
          └──────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────────────┐
          │               STEP 3: GENERATE INTRO                  │
          │  ┌─────────────────────────────────────────────────┐  │
          │  │  generate_intro_section()                       │  │
          │  │   - Uses sem_agg() to summarize all docs        │  │
          │  │   - Creates cohesive background/intro section   │  │
          │  │   - Includes inline citations [author, date]    │  │
          │  └─────────────────────────────────────────────────┘  │
          └──────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────────────┐
          │                 STEP 4: TAXONOMIZE                    │
          │  ┌─────────────────────────────────────────────────┐  │
          │  │  if categorize_references:                      │  │
          │  │    1. generate_categories()                     │  │
          │  │       - LLM creates ≤10 distinct categories     │  │
          │  │                                                 │  │
          │  │    2. match_references_to_categories()          │  │
          │  │       - sem_map() assigns each doc a category   │  │
          │  │                                                 │  │
          │  │    3. if generate_category_summary:             │  │
          │  │       - sem_agg() per category → summaries      │  │
          │  └─────────────────────────────────────────────────┘  │
          └──────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────────────┐
          │               STEP 5: GENERATE INSIGHTS               │
          │  ┌─────────────────────────────────────────────────┐  │
          │  │  if generate_insights:                          │  │
          │  │    sem_extract() on each doc                    │  │
          │  │    - Extracts: "key idea/summary"               │  │
          │  │    - Adds new column(s) to docs_df              │  │
          │  └─────────────────────────────────────────────────┘  │
          └──────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────────────┐
          │             STEP 6: GENERATE FINAL REPORT             │
          │  ┌─────────────────────────────────────────────────┐  │
          │  │  generate_final_report()                        │  │
          │  │                                                 │  │
          │  │  If categorized:                                │  │
          │  │    - Creates outline with category links        │  │
          │  │    - Groups papers by category                  │  │
          │  │    - Adds category summaries                    │  │
          │  │                                                 │  │
          │  │  Formats papers as Markdown tables              │  │
          │  │  Combines: intro + outline + categorized refs   │  │
          │  └─────────────────────────────────────────────────┘  │
          └──────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
                        ┌───────────────┐
                        │    OUTPUT     │
                        │  final_report │ (Markdown string)
                        │    docs_df    │ (DataFrame)
                        │     stats     │ (Dict with usage)
                        └───────────────┘
```

### Search Modes

The pipeline begins by searching for relevant literature. You can choose between two search strategies:

#### Agentic Search (`use_agentic_search=True`, default)

An AI agent iteratively searches arXiv and the web using tools. The agent decides what to search, reads promising results, refines its queries, and synthesizes a background section—all autonomously over up to 100 turns.

```
                              ┌─────────────────────────────────────┐
                              │           AI AGENT                  │
                              │   (up to 100 autonomous turns)      │
                              └──────────────┬──────────────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    ▼                        ▼                        ▼
          ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
          │  search_arxiv() │      │  search_web()   │      │ read_abstracts()│
          │                 │      │    (Tavily)     │      │ read_webpages() │
          └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
                   │                        │                        │
                   └────────────────────────┼────────────────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────────┐
                              │   Agent reviews results,    │
                              │   decides next action:      │
                              │   • Search more?            │
                              │   • Read specific papers?   │
                              │   • Done collecting?        │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │  Synthesize background +    │
                              │  Return collected papers_df │
                              └─────────────────────────────┘
```

#### Recursive Search (`use_agentic_search=False`)

A structured multi-step approach where the LLM generates queries, searches multiple corpuses in parallel, and uses accumulated results to generate better queries in subsequent iterations.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RECURSIVE SEARCH LOOP                                   │
│                      (num_search_steps iterations)                              │
└─────────────────────────────────────────────────────────────────────────────────┘

Step 1                        Step 2                        Step N
───────                       ───────                       ───────

┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│ Generate     │             │ Generate     │             │ Generate     │
│ Queries      │             │ Queries      │             │ Queries      │
│ (from topic) │             │ (topic +     │             │ (topic +     │
└──────┬───────┘             │  background) │             │  background) │
       │                     └──────┬───────┘             └──────┬───────┘
       ▼                            ▼                            ▼
┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│   Search     │             │   Search     │             │   Search     │
│  ┌────────┐  │             │  ┌────────┐  │             │  ┌────────┐  │
│  │ arXiv  │  │             │  │ arXiv  │  │             │  │ arXiv  │  │
│  ├────────┤  │             │  ├────────┤  │             │  ├────────┤  │
│  │ Tavily │  │             │  │ Tavily │  │             │  │ Tavily │  │
│  ├────────┤  │             │  ├────────┤  │             │  ├────────┤  │
│  │ Google │  │             │  │ Google │  │             │  │ Google │  │
│  └────────┘  │             │  └────────┘  │             │  └────────┘  │
└──────┬───────┘             └──────┬───────┘             └──────┬───────┘
       │                            │                            │
       ▼                            ▼                            ▼
┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│  Dedupe &    │             │  Dedupe &    │             │  Dedupe &    │
│  Accumulate  │────────────▶│  Accumulate  │────...─────▶│  Accumulate  │
│  Results     │             │  Results     │             │  Results     │
└──────┬───────┘             └──────┬───────┘             └──────┬───────┘
       │                            │                            │
       ▼                            ▼                            ▼
┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│  Summarize   │             │  Summarize   │             │    Final     │
│  Background  │────────────▶│  Background  │────...─────▶│   Output     │
└──────────────┘             └──────────────┘             └──────────────┘
       │                            │
       │    background informs      │    background informs
       └────── next queries ────────┘────── next queries ──────▶
```

Both modes support an optional `end_date` parameter that filters results to only include papers published before that date. Web search can be disabled with `enable_web_search=False` to search only arXiv.

## Usage

```python
from deepscholar_base import deepscholar_base
from deepscholar_base.configs import Configs
from lotus.models import LM
from datetime import datetime
import asyncio

# Configure the pipeline with a base LM
configs = Configs(
    lm=LM(model="gpt-4o", temperature=1.0, max_tokens=10000)
)

# Or use different LMs for different stages
configs = Configs(
    search_lm=LM(model="gpt-4o", temperature=0.7),      # For query generation & agentic search
    filter_lm=LM(model="gpt-4o-mini", temperature=0),   # For semantic filtering
    taxonomize_lm=LM(model="gpt-4o", temperature=0.5),  # For categorization
    generation_lm=LM(model="gpt-4o", temperature=0.7),  # For summaries & report
)

# Run the pipeline
async def main():
    final_report, docs_df, stats = await deepscholar_base(
        topic="What are the latest developments in retrieval-augmented generation?",
        end_date=datetime(2025, 1, 1),  # Only papers before this date
        configs=configs,
    )
    print(final_report)
    print(f"Found {len(docs_df)} papers")
    print(f"Total tokens used: {stats['total_usage']['total_tokens']}")

asyncio.run(main())
```

## Configuration Reference

### Search Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_agentic_search` | `True` | Use autonomous AI agent for search (vs structured recursive search) |
| `enable_web_search` | `True` | Include web results via Tavily alongside arXiv |
| **(Only for recursive search)** | | |
| `per_query_max_search_results_count` | `10` | Maximum results to fetch per query |
| `max_search_retries` | `3` | Retries if search+filter yields no results |
| `num_search_steps` | `3` | Iterations for recursive search (ignored if agentic) |
| `num_search_queries_per_step_per_corpus` | `2` | Queries per step in recursive search |
| `web_corpuses` | `[TAVILY]` | Web search providers for recursive search |
| **(Only for agentic search)** | | |
| `use_responses_model` | `None` | Force OpenAI Responses vs Chat Completions API |

### Filter Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_sem_filter` | `True` | Apply semantic relevance filtering |
| `use_sem_topk` | `True` | Rank and select top-K most relevant papers |
| `final_max_results_count` | `30` | Maximum papers to keep after filtering |
| `sem_filter_kwargs` | `{strategy: COT}` | Arguments for LOTUS `sem_filter()` |
| `sem_topk_kwargs` | `{strategy: COT}` | Arguments for LOTUS `sem_topk()` |

### Generation Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `categorize_references` | `True` | Organize papers into thematic categories |
| `generate_category_summary` | `True` | Write summary paragraphs per category |
| `generate_insights` | `True` | Extract key ideas from each paper |
| `use_structured_output` | `True` | Use Pydantic models for LLM outputs |

### Language Model Settings

You can provide a single `lm` parameter, which will be used as the default for all stages. Alternatively, you can specify different models for each stage to optimize for cost, speed, or capability:

| Parameter | Purpose |
|-----------|---------|
| `lm` | Default LM used when stage-specific LMs aren't provided |
| `search_lm` | Query generation and agentic search reasoning |
| `filter_lm` | Semantic filtering and ranking |
| `taxonomize_lm` | Category creation and paper-to-category mapping |
| `generation_lm` | Introduction, summaries, insights, and final report |

## Output

The pipeline returns a tuple of three values:

1. **`final_report`** (str): The complete Markdown document ready for display or export
2. **`docs_df`** (DataFrame): All filtered papers with columns: `id`, `title`, `url`, `snippet`, `date`, `authors`, `category`, `key idea/summary`
3. **`stats`** (dict): Detailed statistics including per-stage token usage, intermediate results, and any errors

