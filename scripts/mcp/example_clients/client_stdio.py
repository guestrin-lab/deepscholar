#!/usr/bin/env python3
"""Example MCP client using stdio transport. Run from project root: python scripts/mcp/examples/client_stdio.py"""

import asyncio
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

server_params = StdioServerParameters(
    command="python",
    args=["scripts/mcp/server.py"],
    cwd=str(PROJECT_ROOT),
)


async def main() -> None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "run_deepscholar_base",
                {"topic": "retrieval-augmented generation", "end_date": "2025-01-01"},
            )

            data = json.loads(result.content[0].text)
            print("final_report (first 500 chars):")
            print((data.get("final_report") or "")[:500])
            print("\nstats:", data.get("stats"))


if __name__ == "__main__":
    asyncio.run(main())
