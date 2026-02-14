# DeepScholar-Base MCP Server

Exposes the DeepScholar-Base pipeline as a single MCP tool (`run_deepscholar_base`) via [FastMCP](https://github.com/jlowin/fastmcp).

## Installation

From the **project root** (e.g. `deepscholar-bench/`):

```bash
# Install main project dependencies and MCP server deps
pip install -r requirements.txt
pip install -r scripts/mcp/requirements.txt
```

Ensure your environment is activated (e.g. `conda activate dsbench`) and that a `.env` with `OPENAI_API_KEY` and `TAVILY_API_KEY` exists in the project root.

## Running the Server

Run the server from the **project root** so that `configs/` and `deepscholar_base` resolve correctly.

```bash
# From project root
cd /path/to/deepscholar-bench

# Stdio (default — for Claude Desktop, Cursor, etc.)
python scripts/mcp/server.py

# SSE transport (for HTTP clients)
python scripts/mcp/server.py --transport sse --port 8080

# Streamable HTTP
python scripts/mcp/server.py --transport streamable-http --port 8080

# Custom config
python scripts/mcp/server.py --config path/to/my_config.yaml

# All options
python scripts/mcp/server.py --help
```


## Configuration

The server reuses the same YAML config as `deepscholar_base/main.py`. The default is `configs/deepscholar_base.yaml` in the project root. Override any setting there (model, search mode, filters, etc.).

```yaml
# configs/deepscholar_base.yaml
lm:
  model: gpt-4o
  temperature: 1.0
  max_tokens: 10000

search_mode: agentic
enable_web_search: true
final_max_results_count: 30
```

## Tool: `run_deepscholar_base`

Runs the full DeepScholar-Base pipeline (search → filter → taxonomize → generate report).

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `topic` | string | yes | Research topic, question, or paper abstract |
| `end_date` | string | no | Paper cutoff date in `YYYY-MM-DD` format |

**Returns:** JSON string with three keys:

```json
{
  "final_report": "# Related Works\n...",
  "papers": [
    {"title": "...", "url": "...", "snippet": "...", ...}
  ],
  "stats": {"search_time": 12.3, "num_papers_found": 47, ...}
}
```

## Testing the Server and Clients

From the project root, with dependencies installed:

1. **Server only** — verify the server starts:
   ```bash
   python scripts/mcp/server.py --help
   ```

2. **Stdio client** — spawns the server and calls the tool (needs `mcp` from `scripts/mcp/requirements.txt`):
   ```bash
   python scripts/mcp/example_clients/client_stdio.py
   ```
   This runs the full pipeline once; ensure `.env` has `OPENAI_API_KEY` and `TAVILY_API_KEY`.

3. **SSE client** — start the server in another terminal, then run the client:
   ```bash
   # Terminal 1
   python scripts/mcp/server.py --transport sse --port 8080
   # Terminal 2
   python scripts/mcp/example_clients/client_sse.py
   ```
