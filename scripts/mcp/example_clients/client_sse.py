#!/usr/bin/env python3
"""Example MCP client using SSE transport. Start the server first:
  python scripts/mcp/server.py --transport sse --port 8080
Then run from project root: python scripts/mcp/examples/client_sse.py
"""

import asyncio
import json

from mcp import ClientSession
from mcp.client.sse import sse_client


async def main() -> None:
    async with sse_client("http://localhost:8080/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "run_deepscholar_base",
                {"topic": "LLM agents"},
            )
            data = json.loads(result.content[0].text)
            print("final_report (first 500 chars):")
            print((data.get("final_report") or "")[:500])
            print("\nstats:", data.get("stats"))


if __name__ == "__main__":
    asyncio.run(main())
