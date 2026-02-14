<h1 align="center"> 
    🌐🔍DeepScholar-Bench: Build and Benchmark Generative Research Synthesis
</h1>


<!-- [![Dataset](https://img.shields.io/badge/Dataset-deepscholar--bench%2FDeepScholarBench-blue)](https://huggingface.co/datasets/deepscholar-bench/DeepScholarBench)
[![GitHub](https://img.shields.io/badge/GitHub-deepscholar--bench-green)](https://github.com/guestrin-lab/deepscholar-bench)
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/guestrin-lab/deepscholar-bench/blob/main/LICENSE)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-DeepScholar%20Bench-orange)](https://guestrin-lab.github.io/deepscholar-leaderboard/leaderboard/deepscholar_bench_leaderboard.html)
-->

<!-- **📊 Dataset**: [deepscholar-bench/DeepScholarBench](https://huggingface.co/datasets/deepscholar-bench/DeepScholarBench)  
**🔗 GitHub**: [guestrin-lab/deepscholar-bench](https://github.com/guestrin-lab/deepscholar-bench)
**🏆 Leaderboard**: [DeepScholar Bench Leaderboard](https://guestrin-lab.github.io/deepscholar-leaderboard/leaderboard/deepscholar_bench_leaderboard.html) -->


<p align="center">
<a href="https://guestrin-lab.github.io/deepscholar-leaderboard/leaderboard/deepscholar_bench_leaderboard.html"><b> 🏆 Live Leaderboard</b></a> | <a href="https://deep-scholar.vercel.app"><b> 🤖 DeepScholar Live Preview </b></a> 
</p>

<p align="center">
<a href="https://huggingface.co/datasets/deepscholar-bench/DeepScholarBench"><b> 📊 Dataset </b></a> | <a href="https://arxiv.org/abs/2508.20033"><b>📄 Paper</b></a> | <a href="https://discord.gg/ZWQBurm5bt"><b> 🎮Discord </b></a>
</p>

---

DeepScholar-Bench provides a live benchmark dataset and holistic evaluation of generative research synthesis, an emerging capability among AI systems designed for DeepResearch. We also developed DeepScholar-base, a strong open-source reference pipeline/

This repository provides:
1. **[Dataset Scripts](data_pipeline/README.md)** - which allow you to collect new datasets from recent, high-quality Arxiv papers using our automated data-collection pipeline. You can set your own configurations (e.g., choice of valid date ranges and valid Arxiv domains) to customize your dataset
2. **[An Evaluation Suite](eval/README.md)** - for measuring performance of long-form research synthesis answers. Our evaluation framework supports a holistic set of metrics, which demonstrate high agreement with human annotations. Our eval suite is built using the [LOTUS framework for LLM-based data processing](https://github.com/lotus-data/lotus), which  provides a library for LLM-based evaluations and can be used directly to instantiate [your custom LLM-judges](https://lotus-ai.readthedocs.io/en/latest/evals.html#).
3. **[DeepScholar-base](deepscholar_base/README.md)** - our open-source reference pipeline for generative research synthesis. It is built on top of the [LOTUS framework](https://github.com/lotus-data/lotus), which introduces and serves semantic operators for LLM-powered data processing. LOTUS' semantic operators provide a rich-set of primitives, providing a superset of RAG that goes beyond search() and LM() calls. On DeepScholar-bench, our reference pipeline achieves competitive performance with OpenAI's DeepResearch, while running 2x faster.

If you run into any problems with the code in this repo, leaderboard, or dataset, please feel free to raise an issue and we will address it promptly. If you would like to add your AI system to the DeepScholar-bench leaderboard, please fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLSeug4igDHhVUU3XnrUSeMVRUJFKlHP28i8fcBAu_LHCkqdV1g/viewform).



## 🚀 Quick Start

To get started, make sure you are using Python 3.10, simply clone the repository and install dependencies as follows:

```bash
# Clone the repository
git clone git@github.com:guestrin-lab/deepscholar-bench.git
cd deepscholar-bench

# Install dependencies
conda create -n dsbench python=3.10 -y
conda activate dsbench
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY and TAVILY_API_KEY
```

### 🔄 Complete Pipeline

`scripts/run_all.sh` orchestrates data collection → report generation → evaluation end-to-end:

```bash
# Full pipeline for CS papers from Jan 2025
./scripts/run_all.sh --field cs --start-date 2025-01-01 --output-base runs/cs_jan2025

# Single paper
./scripts/run_all.sh --paper-id 2502.07374 --output-base runs/single_paper

# Skip collection, evaluate existing results only
./scripts/run_all.sh --skip-data-pipeline --skip-deepscholar --output-base runs/cs_jan2025
```

Output structure:

```
{output-base}/
├── dataset/                         # Collected papers
│   ├── papers_with_related_works.csv
│   └── queries.csv
├── results/                         # Generated reports (one folder per query)
│   ├── 0/
│   │   ├── final_report.md
│   │   ├── papers.csv
│   │   └── stats.json
│   └── summary.json
├── evaluation/
│   └── results.csv
└── pipeline_config.txt
```

### 📊 Benchmark Usage
You can start scraping your own datasets and running our holistic, automated evaluation suite using the commands below. For more details and a full introduction, please continue to our **[Dataset Scripts Description](data_pipeline/README.md)** and/or our **[Evaluation library Description](eval/README.md)**.


#### 1. Scraping Data

```bash
# Collect recent AI papers since May 1, 2025
python -m data_pipeline.main \
    --categories cs.AI \
    --start-date 2025-05-01

# Collect by field shorthand (cs, bio, econ, phy, stat)
python -m data_pipeline.main --field cs --field bio --start-date 2025-01-01
```

#### 2. Evaluate Research Generation Systems

```bash
# Evaluate deepscholar_base outputs with gpt-4o as judge
python -m eval.main \
    --modes deepscholar_base \
    --evals organization nugget_coverage reference_coverage cite_p \
    --input-folder tests/baselines_results/deepscholar_base_gpt_4.1 \
    --output-folder results \
    --dataset-path dataset/related_works_combined.csv \
    --model-name gpt-4o

# Compare multiple systems
python -m eval.main \
    --modes deepscholar_base openscholar \
    --input-folder results/deepscholar/ results/openscholar/ \
    --output-folder eval/ \
    --evals all
```



### 📚 DeepScholar-Base

DeepScholar-Base is our reference pipeline research synthesis pipeline that generates comprehensive literature reviews from a research query. It serves as a strong, open-source baseline and is built on [LOTUS](https://github.com/lotus-data/lotus) for efficient LLM-based data processing. For detailed documentation see the **[DeepScholar Base README](deepscholar_base/README.md)**.

```python
from deepscholar_base import deepscholar_base
from deepscholar_base.configs import Configs
from lotus.models import LM
from datetime import datetime
import asyncio

configs = Configs(lm=LM(model="gpt-4o", temperature=1.0, max_tokens=10000))

async def main():
    final_report, docs_df, stats = await deepscholar_base(
        topic="What are the latest developments in retrieval-augmented generation?",
        end_date=datetime(2025, 1, 1),  # Only papers before this date
        configs=configs,
    )
    print(final_report)

asyncio.run(main())
```


## 🔌 MCP Integration

DeepScholar-Base can be used as an [MCP (Model Context Protocol)](https://spec.modelcontextprotocol.io/) server, letting you call it from Cursor, Claude Desktop, or any MCP-compatible client. See [scripts/mcp/README.md](scripts/mcp/README.md) for full setup, Cursor config, and usage.

```bash
# Start the MCP server (from project root)
python scripts/mcp/server.py
```

## 🤝 Contributing

We welcome contributions to DeepScholarBench! Please feel free to submit a PR for code contributions. If you would like to add your AI system to the DeepScholar-bench leaderboard, please fill out this [this form](https://docs.google.com/forms/d/e/1FAIpQLSeug4igDHhVUU3XnrUSeMVRUJFKlHP28i8fcBAu_LHCkqdV1g/viewform).


## Citation
If you use DeepScholar-Bench in an academic work, we would greatly appreciate it if you can cite this work as follows:
```bibtex
@article{patel2025deepscholarbench,
      title={DeepScholar-Bench: A Live Benchmark and Automated Evaluation for Generative Research Synthesis}, 
      author={Liana Patel and Negar Arabzadeh and Harshit Gupta and Ankita Sundar and Ion Stoica and Matei Zaharia and Carlos Guestrin},
      year={2025},
      url={https://arxiv.org/abs/2508.20033}, 
}
```
