# AI-Driven Research for Databases

Three case studies that use ADRS evolutionary search to optimize core PostgreSQL components: buffer management, index selection, and query rewriting.

## Projects

| Directory | Target Subsystem | Description |
|-----------|-----------------|-------------|
| [`buffer_cache/`](buffer_cache/) | Buffer eviction policies | Co-evolution of simulator configurations and eviction policies for PostgreSQL buffer management |
| [`index_selection/`](index_selection/) | Index advisor algorithms | Co-evolution of index selection algorithms and evaluation metrics for PostgreSQL |
| [`query_rewrite/`](query_rewrite/) | SQL rewrite rule selection | Co-evolution of Apache Calcite rewrite rule combinations and query classification strategies |

## Submodules

After cloning, initialize submodules:

```bash
git submodule update --init --recursive
```

| Submodule | Path | Source |
|-----------|------|--------|
| postgres-pbm | `buffer_cache/postgres-pbm` | [TheoVanderkooy/postgres-pbm](https://github.com/TheoVanderkooy/postgres-pbm) |
| Index Selection Evaluation | `index_selection/deps/Index_EAB` | [hyrise/index_selection_evaluation](https://github.com/hyrise/index_selection_evaluation) |
| OpenEvolve | `index_selection/deps/openevolve` | [algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve) |
| LearnedRewrite (R-Bot) | `query_rewrite/rbot` | [XuanheZhou/LearnedRewrite](https://github.com/XuanheZhou/LearnedRewrite) |

## Setup

Each project has its own README with detailed setup instructions. Common requirements:

- Python 3.10+
- PostgreSQL with benchmark data (TPC-H, TPC-DS, JOB)
- LLM API keys (OpenAI, Anthropic, and/or Gemini depending on the project)
