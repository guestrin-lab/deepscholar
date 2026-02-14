#!/usr/bin/env python3
"""
Script to run deepscholar_base on a generated dataset.

Usage:
    python -m deepscholar_base.main --queries-file dataset/queries.csv --output-folder results/
    python -m deepscholar_base.main --output-folder results/ --model gpt-4o --start-idx 0 --end-idx 10
    python -m deepscholar_base.main --config-yaml configs/deepscholar_base.yaml --output-folder results/
"""

import argparse
import asyncio
import json
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from lotus.models import LM
try:
    from deepscholar_base import deepscholar_base
    from deepscholar_base.configs import Configs
except ImportError:
    from ..deepscholar_base import deepscholar_base
    from ..deepscholar_base.configs import Configs

_DEFAULT_CONFIG_YAML = "configs/deepscholar_base.yaml"

# Keys in the YAML that map to runner-level args (not Configs fields).
# These are applied via parser.set_defaults(); everything else in the YAML
# is handled by Configs.from_yaml().
_RUNNER_YAML_KEYS = {"queries_file", "output_folder", "start_idx", "end_idx"}


def _load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a dict. Returns {} on any error."""
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            print(f"Warning: {path} is not a YAML mapping; ignoring.", file=sys.stderr)
            return {}
        return data
    except Exception as e:
        print(f"Warning: Could not load config {path}: {e}", file=sys.stderr)
        return {}


def _configs_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build non-LM Configs kwargs from CLI args. Only include keys where value is not None."""
    kwargs: dict[str, Any] = {}
    for key, attr in [
        ("use_agentic_search", "use_agentic_search"),
        ("max_search_retries", "max_search_retries"),
        ("use_structured_output", "use_structured_output"),
        ("enable_web_search", "enable_web_search"),
        ("per_query_max_search_results_count", "per_query_max_search_results_count"),
        ("use_responses_model", "use_responses_model"),
        ("num_search_steps", "num_search_steps"),
        ("num_search_queries_per_step_per_corpus", "num_search_queries_per_step_per_corpus"),
        ("web_corpuses", "web_corpuses"),
        ("use_sem_filter", "use_sem_filter"),
        ("use_sem_topk", "use_sem_topk"),
        ("final_max_results_count", "final_max_results_count"),
        ("categorize_references", "categorize_references"),
        ("generate_category_summary", "generate_category_summary"),
        ("generate_insights", "generate_insights"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            kwargs[key] = val
    return kwargs


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    # Pre-parse to get --config-yaml before building the main parser
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-yaml", default=_DEFAULT_CONFIG_YAML)
    pre_args, _ = pre_parser.parse_known_args()

    yaml_values = _load_yaml(pre_args.config_yaml)

    # Extract only runner-level keys to apply as argparse defaults.
    # Configs-level keys (lm, search_mode, etc.) remain in the YAML and are
    # consumed by Configs.from_yaml() during main().
    runner_yaml = {k: yaml_values[k] for k in _RUNNER_YAML_KEYS if k in yaml_values}

    parser = argparse.ArgumentParser(
        description="Run deepscholar_base on a dataset and save results"
    )
    # I/O
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder to store results",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default="dataset/queries.csv",
        help="Path to the query CSV file (default: dataset/queries.csv)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index of queries to process (default: 0)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index of queries to process (default: None, process all)",
    )
    # Configs
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=pre_args.config_yaml,
        help=(
            f"Path to YAML config file (default: {_DEFAULT_CONFIG_YAML}). "
            "Provides base values for both runner args and Configs; "
            "any explicitly-passed CLI arg overrides the YAML. "
            "Silently ignored if the file does not exist."
        ),
    )
    # LM — only model name; use --config-yaml for temperature, max_tokens, etc.
    parser.add_argument("--model", type=str, default=None, help="Model name to use")

    # Search
    parser.add_argument(
        "--search-mode",
        type=str,
        choices=["agentic", "recursive"],
        default=None,
        help="Search mode: agentic or recursive",
    )
    parser.add_argument("--max-search-retries", type=int, default=None, help="Max search retries")
    parser.add_argument(
        "--use-structured-output",
        "--no-structured-output",
        dest="use_structured_output",
        action=_OptionalBooleanAction,
        default=None,
        help="Use structured output",
    )
    parser.add_argument(
        "--enable-web-search",
        "--no-web-search",
        dest="enable_web_search",
        action=_OptionalBooleanAction,
        default=None,
        help="Enable web search",
    )
    parser.add_argument(
        "--per-query-max-search-results",
        type=int,
        default=None,
        dest="per_query_max_search_results_count",
        help="Max search results per query",
    )
    parser.add_argument(
        "--use-responses-model",
        action="store_true",
        default=None,
        dest="use_responses_model",
        help="Use responses model (agentic only)",
    )
    parser.add_argument("--num-search-steps", type=int, default=None, help="Search steps (recursive only)")
    parser.add_argument(
        "--num-search-queries-per-step",
        type=int,
        default=None,
        dest="num_search_queries_per_step_per_corpus",
        help="Queries per step per corpus (recursive only)",
    )
    parser.add_argument(
        "--web-corpuses",
        type=str,
        nargs="+",
        default=None,
        help="Web corpuses, e.g. TAVILY ARXIV",
    )

    # Filter
    parser.add_argument(
        "--use-sem-filter",
        "--no-sem-filter",
        dest="use_sem_filter",
        action=_OptionalBooleanAction,
        default=None,
        help="Use semantic filter",
    )
    parser.add_argument(
        "--use-sem-topk",
        "--no-sem-topk",
        dest="use_sem_topk",
        action=_OptionalBooleanAction,
        default=None,
        help="Use semantic top-k",
    )
    parser.add_argument(
        "--final-max-results",
        type=int,
        default=None,
        dest="final_max_results_count",
        help="Final max results count",
    )

    # Taxonomization
    parser.add_argument(
        "--categorize-references",
        "--no-categorize-references",
        dest="categorize_references",
        action=_OptionalBooleanAction,
        default=None,
        help="Categorize references",
    )
    parser.add_argument(
        "--generate-category-summary",
        "--no-generate-category-summary",
        dest="generate_category_summary",
        action=_OptionalBooleanAction,
        default=None,
        help="Generate category summary",
    )
    parser.add_argument(
        "--generate-insights",
        "--no-generate-insights",
        dest="generate_insights",
        action=_OptionalBooleanAction,
        default=None,
        help="Generate insights",
    )

    # Apply runner-level YAML values as defaults so CLI args naturally override them
    if runner_yaml:
        parser.set_defaults(**runner_yaml)

    args = parser.parse_args()

    # Validate required runner args (can come from YAML or CLI)
    if not args.output_folder:
        parser.error(
            "--output-folder is required (or set 'output_folder' in the config YAML)"
        )

    args.use_agentic_search = (args.search_mode == "agentic") if args.search_mode is not None else None
    return args


def load_queries(queries_file: str) -> pd.DataFrame:
    """Load queries from the input folder."""
    query_path = Path(queries_file)

    if not query_path.exists():
        # Try to find papers_with_related_works.csv and generate queries
        papers_path = query_path.parent / "papers_with_related_works.csv"
        if not papers_path.exists():
            raise FileNotFoundError(
                f"Neither {query_path} nor {papers_path} found. "
                f"Please ensure the input folder contains query data."
            )

        print(f"⚠️  {queries_file} not found. Generating queries from papers_with_related_works.csv...")
        papers_df = pd.read_csv(papers_path)

        # Generate queries using the template
        QUERY_TEMPLATE = """Your task is to write a Related Works section for an academic paper given the paper's abstract. Your response should provide the Related Works section and references. Only include references from arXiv that are published before {cutoff_date}. Mention them in a separate, numbered reference list at the end and use the reference numbers to provide in-line citations in the Related Works section for all claims referring to a source (e.g., description of source [3]. Further details [6][7][8][9][10].) Each in-line citation must consist of a single reference number within a pair of brackets. Do not use any other citation format. Do not exceed 600 words for the related works section. Here is the paper abstract: {abstract}"""

        queries_df = papers_df.copy()
        queries_df["query"] = queries_df.apply(
            lambda row: QUERY_TEMPLATE.format(
                cutoff_date=row["published_date"],
                abstract=row["abstract"]
            ),
            axis=1
        )

        # Save generated queries
        queries_df.to_csv(query_path, index=False)
        print(f"✅ Generated and saved queries to {query_path}")

        return queries_df

    return pd.read_csv(query_path)


async def process_query(
    idx: int,
    query: str,
    arxiv_id: str,
    end_date: datetime | None,
    configs: Configs,
    output_folder: Path,
) -> dict[str, Any]:
    """Process a single query and save results."""
    query_output_folder = output_folder / str(idx)
    query_output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing query {idx}: {arxiv_id}")
    print(f"{'='*60}")

    try:
        # Run deepscholar_base
        final_report, docs_df, stats = await deepscholar_base(
            topic=query,
            end_date=end_date,
            configs=configs,
        )

        # Save final report
        final_report_path = query_output_folder / "final_report.md"
        with open(final_report_path, "w") as f:
            f.write(final_report)
        print(f"✅ Saved final report to {final_report_path}")

        # Save intro section
        intro_section = stats["intro_section"]
        intro_section_path = query_output_folder / "intro.md"
        with open(intro_section_path, "w") as f:
            f.write(intro_section)
        print(f"✅ Saved intro section to {intro_section_path}")

        # Save papers dataframe
        if docs_df is not None:
            papers_path = query_output_folder / "paper.csv"
            docs_df.to_csv(papers_path, index=False)
            print(f"✅ Saved papers to {papers_path}")

        # Save stats
        stats_path = query_output_folder / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"✅ Saved stats to {stats_path}")

        return {
            "idx": idx,
            "arxiv_id": arxiv_id,
            "status": "success",
            "num_papers": len(docs_df) if docs_df is not None else 0,
            "final_report": final_report,
        }

    except Exception as e:
        print(f"❌ Error processing query {idx}: {e}")
        import traceback
        error_path = query_output_folder / "error.txt"
        with open(error_path, "w") as f:
            f.write(f"Error: {e}\n\n")
            f.write(traceback.format_exc())

        return {
            "idx": idx,
            "arxiv_id": arxiv_id,
            "status": "error",
            "error": str(e),
        }


async def main():
    """Main function."""
    args = parse_args()

    # Load environment variables
    load_dotenv()

    # Create output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DeepScholar-Base Runner")
    print(f"{'='*60}")
    print(f"Queries file:         {args.queries_file}")
    print(f"Output folder:        {args.output_folder}")
    _yaml_path = Path(args.config_yaml) if args.config_yaml else None
    _yaml_exists = bool(_yaml_path and _yaml_path.exists())
    if _yaml_exists:
        print(f"Config:               {args.config_yaml} (YAML)")
    if args.model:
        print(f"Model override:       {args.model}")
    if args.search_mode and not _yaml_exists:
        print(f"Search mode:          {args.search_mode}")
    print(f"{'='*60}\n")

    # Load queries
    print(f"Loading queries from {args.queries_file}...")
    queries_df = load_queries(args.queries_file)
    print(f"✅ Loaded {len(queries_df)} queries")

    # Filter by indices
    if args.end_idx is not None:
        queries_df = queries_df.iloc[args.start_idx:args.end_idx]
    else:
        queries_df = queries_df.iloc[args.start_idx:]

    print(f"Processing queries {args.start_idx} to {args.start_idx + len(queries_df)}")

    # Build configs: YAML provides the base (if given), CLI args override only what was explicitly passed.
    cfg_kwargs = _configs_kwargs_from_args(args)
    yaml_path = Path(args.config_yaml) if args.config_yaml else None
    if yaml_path and yaml_path.exists():
        configs = Configs.from_yaml(yaml_path)
        if cfg_kwargs or args.model is not None:
            base = configs.model_dump(
                exclude={"filter_lm", "search_lm", "taxonomize_lm", "generation_lm", "logger"},
                mode="python",
            )
            base.update(cfg_kwargs)
            if args.model is not None:
                # Override only the model name; preserve all other LM settings from YAML
                yaml_lm_kwargs: dict[str, Any] = {
                    "model": args.model,
                    **configs.generation_lm.kwargs,
                }
                base["lm"] = yaml_lm_kwargs
            configs = Configs(**base)
    else:
        if args.model is not None:
            cfg_kwargs["lm"] = LM(model=args.model)
        configs = Configs(**cfg_kwargs)

    # Process queries
    results = []
    for idx, row in queries_df.iterrows():
        query = row["query"]
        arxiv_id = row.get("arxiv_id", f"unknown_{idx}")

        # Parse end date if available
        end_date = None
        if "published_date" in row and pd.notna(row["published_date"]):
            try:
                end_date = pd.to_datetime(row["published_date"])
            except Exception:
                pass

        result = await process_query(
            idx=idx,
            query=query,
            arxiv_id=arxiv_id,
            end_date=end_date,
            configs=configs,
            output_folder=output_folder,
        )
        results.append(result)

    # Save summary
    summary_path = output_folder / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Processing complete!")
    print(f"{'='*60}")
    print(f"Processed {len(results)} queries")
    print(f"Success: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Errors:  {sum(1 for r in results if r['status'] == 'error')}")
    print(f"Summary saved to {summary_path}")
    print(f"{'='*60}\n")


class _OptionalBooleanAction(argparse.Action):
    """Action for optional boolean flags: default None, set True/False only when flag is present (--x -> True, --no-x -> False)."""

    def __init__(self, option_strings, dest, default=None, **kwargs):
        super().__init__(option_strings=option_strings, dest=dest, default=default, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string):
        val = not option_string.startswith("--no-")
        setattr(namespace, self.dest, val)



if __name__ == "__main__":
    asyncio.run(main())
