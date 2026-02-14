#!/usr/bin/env python3
"""
MCP Server for DeepScholar-Base

Usage (from project root):
    python scripts/mcp/server.py                                      # stdio (default)
    python scripts/mcp/server.py --transport sse --port 8080          # SSE
    python scripts/mcp/server.py --config path/to/my_config.yaml      # custom config
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastmcp import FastMCP

from deepscholar_base import deepscholar_base
from deepscholar_base.configs import Configs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_YAML = "configs/deepscholar_base.yaml"

load_dotenv()

mcp = FastMCP("deepscholar-base")


def _load_configs(path: str) -> Configs:
    p = Path(path)
    if p.exists():
        logger.info(f"Loaded config from {p}")
        return Configs.from_yaml(p)
    logger.warning(f"Config not found: {p}; using defaults.")
    return Configs()


_configs = _load_configs(_DEFAULT_CONFIG_YAML)


@mcp.tool()
async def run_deepscholar_base(
    topic: str,
    end_date: Optional[str] = None,
) -> str:
    """Run the DeepScholar-Base pipeline to generate a research synthesis report.

    Searches arXiv and the web for relevant papers, semantically filters and ranks
    them, categorizes references, and generates a related works report with citations.

    Args:
        topic: Research topic or query (question, topic description, or paper abstract).
        end_date: Optional cutoff date in YYYY-MM-DD format. Only papers before this
                  date are included.

    Returns:
        JSON string with keys: 'final_report' (str), 'papers' (list of paper dicts),
        'stats' (pipeline statistics dict).
    """
    parsed_end_date = None
    if end_date:
        try:
            parsed_end_date = datetime.fromisoformat(end_date)
        except ValueError:
            raise ValueError(f"Invalid date format: {end_date!r}. Expected YYYY-MM-DD.")

    final_report, docs_df, stats = await deepscholar_base(
        topic=topic,
        end_date=parsed_end_date,
        configs=_configs,
    )

    papers = docs_df.to_dict(orient="records") if docs_df is not None else []

    return json.dumps(
        {"final_report": final_report, "papers": papers, "stats": stats},
        default=str,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP Server for DeepScholar-Base")
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG_YAML,
        help=f"Path to YAML config file (default: {_DEFAULT_CONFIG_YAML})",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for SSE/HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for SSE/HTTP transport (default: 8080)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Re-load configs if a non-default path was given
    if args.config != _DEFAULT_CONFIG_YAML:
        _configs = _load_configs(args.config)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
