import asyncio
from enum import Enum
from typing import Callable, Coroutine, Any
from datetime import datetime, timezone
import arxiv
from agents import function_tool, RunContextWrapper, RunConfig
from openai.types.responses import ResponseInputItemParam
from agents.models.chatcmpl_converter import Converter
from dataclasses import dataclass
from agents import Agent, Runner
from agents.run import ModelInputData, CallModelData
from agents.util._json import _to_dump_compatible
import pandas as pd
from lotus.types import LMStats
import logging
import tavily
import os
from lotus.models import LM
from openai import AsyncOpenAI
from agents import OpenAIResponsesModel, OpenAIChatCompletionsModel, ModelSettings
from openai.types.shared import Reasoning
import re

try:
    from deepscholar_base.utils.prompts import (
        openai_sdk_arxiv_search_system_prompt,
        openai_sdk_arxiv_search_system_prompt_without_cutoff,
        openai_sdk_search_system_prompt,
        openai_sdk_search_system_prompt_without_cutoff,
    )
    from deepscholar_base.configs import Configs
except ImportError:
    from ..utils.prompts import (
        openai_sdk_arxiv_search_system_prompt,
        openai_sdk_arxiv_search_system_prompt_without_cutoff,
        openai_sdk_search_system_prompt,
        openai_sdk_search_system_prompt_without_cutoff,
    )
    from ..configs import Configs

arxiv_client = arxiv.Client()
arxiv_logger = logging.getLogger("arxiv")
arxiv_logger.setLevel(logging.WARNING)

tavily_client = tavily.AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@dataclass
class AgentContext:
    configs: Configs
    end_date: datetime | None  # YYYYMMDDhhmm (UTC)
    papers_df: pd.DataFrame | None
    queries: list[list[str]]


# All Search Tools
class ToolTypes(Enum):
    ARXIV = "arxiv"
    WEB = "web"

    def to_parsed_results(
        self,
        results: list[dict] | list[arxiv.Result],
        query: str,
        current_papers_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self == ToolTypes.ARXIV:
            assert all(isinstance(result, arxiv.Result) for result in results), (
                "Results must be a list of arxiv.Result objects"
            )
            info = [
                {
                    "id": result.entry_id.split("/")[-1],  # type: ignore
                    "title": result.title,  # type: ignore
                    "url": result.entry_id,  # type: ignore
                    "snippet": result.summary,  # type: ignore
                    "date": result.published.strftime("%Y-%m-%d %H:%M:%S%z"),  # type: ignore
                    "authors": ", ".join([author.name for author in result.authors]),  # type: ignore
                    "query": query,  # type: ignore
                    "context": f"{result.title}[{result.entry_id}]: {result.summary}",  # type: ignore
                }
                for result in results
            ]
        elif self == ToolTypes.WEB:
            assert all(isinstance(result, dict) for result in results), (
                "Results must be a list of dict objects"
            )
            if current_papers_df is not None:
                maps = {
                    row["url"]: row
                    for row in current_papers_df.to_dict(orient="records")
                }
            else:
                maps = {}
            info = [
                {
                    "id": result["url"],  # type: ignore
                    "title": result.get(
                        "title", maps.get(result["url"], {}).get("title", "Untitled")
                    ),  # type: ignore
                    "url": result["url"],  # type: ignore
                    "snippet": result.get(
                        "content",
                        result.get(
                            "raw_content",
                            maps.get(result["url"], {}).get("snippet", "No content"),
                        ),
                    ),  # type: ignore
                    "query": query,  # type: ignore
                    "date": result.get("date", None),
                }
                for result in results
            ]
            for inf in info:
                inf["context"] = f"{inf['title']}[{inf['url']}]: {inf['snippet']}"
        else:
            raise ValueError(f"Invalid search type: {self}")
        return pd.DataFrame(info)

    def to_search_function(
        self,
    ) -> Callable[
        [Configs, datetime | None, str],
        Coroutine[Any, Any, tuple[str, list[dict] | list[arxiv.Result]]],
    ]:  # type: ignore
        if self == ToolTypes.ARXIV:
            return _search_arxiv  # type: ignore
        elif self == ToolTypes.WEB:
            return _search_tavily  # type: ignore
        else:
            raise ValueError(f"Invalid search type: {self}")

    def to_read_function(
        self,
    ) -> Callable[
        [Configs, datetime | None, list[str]],
        Coroutine[Any, Any, tuple[str, list[dict] | list[arxiv.Result]]],
    ]:  # type: ignore
        if self == ToolTypes.ARXIV:
            return _read_arxiv_abstracts  # type: ignore
        elif self == ToolTypes.WEB:
            return _read_webpage_full_text  # type: ignore
        else:
            raise ValueError(f"Invalid search type: {self}")


async def _search_arxiv(
    configs: Configs, cutoff: datetime | None, query: str
) -> tuple[str, list[arxiv.Result]]:
    # Enforce cutoff server-side using arXiv's query syntax
    # Example: (quantum computing) AND submittedDate:[* TO 202201010000]
    if cutoff is not None:
        date_clause = f"submittedDate:[* TO {cutoff.strftime('%Y%m%d%H%M')}]"
        full_query = f"({query}) AND {date_clause}"
    else:
        full_query = query

    search = arxiv.Search(
        query=full_query,
        max_results=configs.per_query_max_search_results_count,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = []
    raw_results = []
    for r in arxiv_client.results(search):
        results.append(f"{r.title} ({r.published.date()}): {r.entry_id}")
        raw_results.append(r)

    return "\n".join(
        results
    ) if results else "No results found on or before the cutoff date.", raw_results


async def _search_tavily(
    configs: Configs, cutoff: datetime | None, query: str, score_threshold: float = 0.5
) -> tuple[str, list[dict]]:
    try:
        response = await tavily_client.search(
            query,
            search_depth="basic",
            end_date=cutoff.strftime("%Y-%m-%d") if cutoff else None,
            max_results=configs.per_query_max_search_results_count,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
        )
    except Exception as e:
        configs.logger.error(f"Tavily error: {e}")
        raise e

    results = [
        result for result in response["results"] if result["score"] >= score_threshold
    ]
    results_section = "\n".join(
        [f"{result['title']}: {result['url']}" for result in results]
    )
    return results_section, results


async def _handle_one_search_query(
    ctx: RunContextWrapper[AgentContext],
    search_type: ToolTypes,
    i: int,
    cutoff: datetime | None,
    query: str,
) -> tuple[str, str | None]:
    successful_query = None
    try:
        query_results, raw_results = await search_type.to_search_function()(
            ctx.context.configs, cutoff, query
        )
        df = search_type.to_parsed_results(raw_results, query, ctx.context.papers_df)
        if ctx.context.papers_df is None or len(ctx.context.papers_df) == 0:
            ctx.context.papers_df = df.copy()
        else:
            ctx.context.papers_df = pd.concat(
                [ctx.context.papers_df, df], ignore_index=True
            )
        successful_query = query
    except Exception as e:
        ctx.context.configs.logger.error(f"Error searching {search_type.value} for query {query}: {e}")
        query_results = f"Error searching {search_type.value} for query {query}: {e}"
    # Format the results with query header
    query_section = f"=== QUERY {i}: {query} ===\n{query_results}"

    return query_section, successful_query


async def _search(
    ctx: RunContextWrapper[AgentContext], search_type: ToolTypes, queries: list[str]
) -> str:
    ctx.context.configs.logger.info(f"Searching {search_type.value} for queries: {queries}")
    cutoff = ctx.context.end_date
    all_results_sections = await asyncio.gather(
        *[
            _handle_one_search_query(
                ctx, search_type, i, cutoff, query
            )
            for i, query in enumerate(queries, 1)
        ]
    )
    successful_queries = [f"{search_type.value}_search"] + [
        successful_query
        for _, successful_query in all_results_sections
        if successful_query is not None
    ]
    ctx.context.configs.logger.info(f"Successful queries: {successful_queries}, collected total references: {len(ctx.context.papers_df)}")
    all_results = [query_section for query_section, _ in all_results_sections]
    ctx.context.queries.append(successful_queries)

    return "\n\n".join(all_results)


@function_tool
async def search_arxiv(ctx: RunContextWrapper[AgentContext], queries: list[str]) -> str:
    """
    Search arXiv for literature that matches the provided queries.
    Your query will be passed to the arXiv API, so it should be in the format of the arXiv API syntax.

    Returns up to 25 entries per query, with clear separation showing which results
    correspond to which query. Each entry is formatted as
    "Title (YYYY-MM-DD): https://arxiv.org/abs/<id>".
    Guidelines for Constructing Effective arXiv Search Queries:
    - Use multiple, concise, and focused queries targeting distinct aspects such as:
      key methods, technologies, canonical author names, and major terms central to the target paper.
    - Design queries to maximize breadth and relevance, not just quantity—use at least two different queries
      to ensure comprehensive coverage of prior work.
    - Leverage synonyms or alternative terms to broaden coverage, but avoid redundant or overly broad queries.
    - Do NOT include the word "arXiv" in any search string.

    Args:
        queries: A list of arXiv query strings using the arXiv API syntax.
            Example: ["attention mechanisms", "transformer architecture", "neural machine translation"]
    """
    return await _search(ctx, ToolTypes.ARXIV, queries)


@function_tool
async def search_web(ctx: RunContextWrapper[AgentContext], queries: list[str]) -> str:
    """
    Search the web for literature that matches the provided queries.
    Your query will be passed to the Tavily API.

    Returns up to 10 entries per query, with clear separation showing which results correspond to which query.
    Each entry is formatted as "Title: URL".
    Guidelines for Constructing Effective Web Search Queries:
    - Use multiple, concise, and focused queries targeting distinct aspects such as:
      key methods, technologies, canonical author names, and major terms central to the target paper.
    - Design queries to maximize breadth and relevance, not just quantity—use at least two different queries to ensure comprehensive coverage of prior work.
    - Leverage synonyms or alternative terms to broaden coverage, but avoid redundant or overly broad queries.
    - Keep the queries under 400 characters.

    Args:
        queries: A list of web search query strings.
            Example: ["attention mechanisms", "transformer architecture", "neural machine translation"]
    """
    return await _search(ctx, ToolTypes.WEB, queries)


# All Read Tools
async def _read_arxiv_abstracts(
    configs: Configs, cutoff: datetime | None, paper_ids: list[str]
) -> tuple[str, list[arxiv.Result]]:
    if cutoff is not None:
        cutoff_dt = datetime.strptime(cutoff, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    else:
        cutoff_dt = None
    search = arxiv.Search(id_list=paper_ids)
    results = []
    raw_results: list[arxiv.Result] = []
    for result in arxiv_client.results(search):
        if cutoff_dt is not None and result.published > cutoff_dt:
            results.append(f"{result.entry_id}: Paper published after the cutoff date.")
        else:
            results.append(f"{result.title}\n\n{result.summary.strip()}")
            raw_results.append(result)
    return (
        "\n\n---\n\n".join(results)
        if results
        else "No valid results found on or before the cutoff date.",
        raw_results,
    )


async def _read_webpage_full_text(
    configs: Configs, cutoff: datetime | None, urls: list[str]
) -> tuple[str, list[dict]]:
    try:
        extracted_data = await tavily_client.extract(urls, extract_depth="basic", format="text")
    except Exception as e:
        configs.logger.error(f"Error extracting content: {e}")
        return f"Error extracting content: {e}", []

    url_to_results = {r["url"]: r for r in extracted_data["results"]}

    results = [
        f"{url}: {url_to_results[url]['raw_content'][:1000]}"
        if url in url_to_results
        else f"{url}: Error extracting content"
        for url in urls
    ]
    raw_results = [url_to_results[url] for url in urls if url in url_to_results]
    return "\n\n---\n\n".join(results) if results else "No content found", raw_results


async def _read_content(
    ctx: RunContextWrapper[AgentContext], tool_type: ToolTypes, input: list[str]
) -> str:
    ctx.context.configs.logger.info(f"Reading content from {tool_type.value} for input: {input}")
    cutoff = ctx.context.end_date
    successful_inputs = [f"{tool_type.value}_read"]
    try:
        answer, raw_results = await tool_type.to_read_function()(
            ctx.context.configs, cutoff, input
        )
        if raw_results:
            df = tool_type.to_parsed_results(
                raw_results, f"{tool_type.value}_read", ctx.context.papers_df
            )
            if ctx.context.papers_df is None or len(ctx.context.papers_df) == 0:
                ctx.context.papers_df = df.copy()
            else:
                ctx.context.papers_df = pd.concat(
                    [ctx.context.papers_df, df], ignore_index=True
                )
            successful_inputs.extend(input)
    except Exception as e:
        answer = f"Error extracting content from {tool_type.value} for {input}: {e}"

    ctx.context.configs.logger.info(f"Successful inputs: {successful_inputs}, collected total references: {len(ctx.context.papers_df)}")
    ctx.context.queries.append(successful_inputs)
    return answer


@function_tool
async def read_arxiv_abstracts(
    ctx: RunContextWrapper[AgentContext], paper_ids: list[str]
) -> str:
    """
    Retrieve the titles and abstracts for a list of arXiv papers.
    There may be a cutoff automatically enforced server-side so works released after the target paper's release date may not be included.

    Use this after `search_arxiv` surfaces promising identifiers to extract
    verifiable details for synthesis. The output includes each paper's title,
    followed by a blank line and the abstract text, separated by ---.

    Args:
        paper_ids: A list of arXiv identifiers, such as ["2408.14717", "2407.11418v3"].
    """
    return await _read_content(ctx, ToolTypes.ARXIV, paper_ids)


@function_tool
async def read_webpage_full_text(
    ctx: RunContextWrapper[AgentContext], urls: list[str]
) -> str:
    """
    Retrieve the full text of a list of web pages.
    Use this after `search_web` surfaces promising URLs to extract
    verifiable details for synthesis. The output includes each webpage's URL,
    followed by a blank line and the full text of the webpage, separated by ---.

    Args:
        urls: A list of web page URLs. Example: ["https://www.google.com", "https://www.wikipedia.org"]
    """
    return await _read_content(ctx, ToolTypes.WEB, urls)


def _call_model_input_filter(input: CallModelData[AgentContext]) -> ModelInputData:
    """
    This function is used to trim input to less than search_lm's max_ctx_len.
    """
    configs = input.context.configs
    instructions = input.model_data.instructions or ""
    input_items = input.model_data.input

    configs.logger.debug(f"Instructions: {instructions}")
    configs.logger.debug(f"Input: {input_items}")

    input_allowed_length = (
        configs.search_lm.max_ctx_len
        - len(instructions)
        - configs.search_lm.max_tokens
    )
    if input_allowed_length < 0:
        raise ValueError(
            f"Input is too long. Max allowed length is {input_allowed_length} tokens. {configs.search_lm.max_ctx_len} is the max context length and {configs.search_lm.max_tokens} is the max tokens."
        )

    user_message_index = [
        index
        for index, message in enumerate(input_items)
        if (
            Converter.maybe_input_message(message)
            or Converter.maybe_easy_input_message(message)
        )
        and message["role"] == "user"
    ]
    configs.logger.debug(f"User message index: {user_message_index}")
    if user_message_index:
        # Keep the last user message in context
        last_user_message_index = user_message_index[-1]
        input_allowed_length = input_allowed_length - len(
            str(input_items[last_user_message_index])
        )
    else:
        last_user_message_index = -1

    final_input: list[ResponseInputItemParam] = []
    for i, message in enumerate(reversed(input_items)):
        if i == len(input_items) - 1 - last_user_message_index:
            final_input.append(message)
            continue
        message_length = len(str(_to_dump_compatible(message)))
        configs.logger.debug(f"Message length: {message_length}")
        if message_length > input_allowed_length:
            continue
        final_input.append(message)
        input_allowed_length -= message_length
    final_input.reverse()
    configs.logger.debug(f"Final input: {final_input}")
    return ModelInputData(instructions=instructions, input=final_input)


async def agentic_search(
    configs: Configs,
    topic: str,
    end_date: datetime | None = None,
) -> tuple[list[list[str]], pd.DataFrame, str]:
    context = AgentContext(
        configs=configs,
        end_date=end_date,
        papers_df=None,
        queries=[],
    )
    model, model_configs = _lotus_lm_to_openai_lm(configs, configs.search_lm)
    tools = [search_arxiv, read_arxiv_abstracts]
    prompt = (
        openai_sdk_arxiv_search_system_prompt_without_cutoff
        if not end_date
        else openai_sdk_arxiv_search_system_prompt
    )
    if configs.enable_web_search:
        configs.logger.info(
            "Web search is enabled, adding web search tools and prompt."
        )
        tools.append(search_web)
        tools.append(read_webpage_full_text)
        prompt = (
            openai_sdk_search_system_prompt_without_cutoff
            if not end_date
            else openai_sdk_search_system_prompt
        )
    agent = Agent(
        name="Research Assistant",
        instructions=prompt,
        tools=tools,
        model=model,
        model_configs=model_configs,
    )
    result = await Runner.run(
        agent,
        input=topic,
        context=context,
        max_turns=100,
        run_config=RunConfig(call_model_input_filter=_call_model_input_filter),
    )
    docs_df = (
        result.context_wrapper.context.papers_df
        if result.context_wrapper.context.papers_df is not None
        else pd.DataFrame(
            columns=["title", "url", "snippet", "query", "context", "date"]
        )
    )
    configs.search_lm.stats.virtual_usage = LMStats.TotalUsage(
        prompt_tokens=result.context_wrapper.usage.input_tokens,
        completion_tokens=result.context_wrapper.usage.output_tokens,
        total_tokens=result.context_wrapper.usage.total_tokens,
    )
    configs.search_lm.stats.physical_usage = LMStats.TotalUsage(
        prompt_tokens=result.context_wrapper.usage.input_tokens,
        completion_tokens=result.context_wrapper.usage.output_tokens,
        total_tokens=result.context_wrapper.usage.total_tokens,
    )
    if len(docs_df) > 0:
        # Deduplicate results by URL
        docs_df = docs_df.drop_duplicates(subset=["url"])
    return result.context_wrapper.context.queries, docs_df, result.final_output


def _lotus_lm_to_openai_lm(
    configs: Configs, lm: LM,
) -> tuple[OpenAIResponsesModel | OpenAIChatCompletionsModel, ModelSettings]:
    client = AsyncOpenAI(
        base_url=lm.kwargs.get("api_base"),
    )
    if _is_responses_model(configs, lm.model):
        configs.logger.info(
            f"Converting Lotus LM to OpenAI Responses Model: {lm.model}"
        )
        model = OpenAIResponsesModel(
            model=lm.model,
            openai_client=client,
        )
    else:
        configs.logger.info(
            f"Converting Lotus LM to OpenAI Chat Completions Model: {lm.model}"
        )
        model = OpenAIChatCompletionsModel(
            model=lm.model,
            openai_client=client,
        )

    model_configs = ModelSettings(
        temperature=lm.kwargs.get("temperature"),
        top_p=lm.kwargs.get("top_p"),
        frequency_penalty=lm.kwargs.get("frequency_penalty"),
        presence_penalty=lm.kwargs.get("presence_penalty"),
        # max_tokens=lm.kwargs.get("max_tokens"),
        reasoning=Reasoning(
            effort=lm.kwargs.get("reasoning_effort", "low"),
        )
        if lm.kwargs.get("reasoning_effort")
        else None,
        truncation="auto",
    )
    configs.logger.info(
        f"Using OpenAI LM: {lm.model} {model_configs}"
    )

    return model, model_configs


def _is_responses_model(configs: Configs, model: str) -> bool:
    if configs.use_responses_model is not None:
        return configs.use_responses_model
    if "gpt" in model.lower():
    # Extracts the first model number from the string, e.g., "gpt-4-turbo" -> 4, "gpt-4.1-extended" -> 4.1
        if "oss" in model.lower():
            return True
        match = re.search(r"(\d+(?:\.\d+)?)", model)
        if match:
            try:
                model_number = float(match.group())
                return model_number >= 5.0
            except ValueError:
                return False
        return False

