import pandas as pd
from typing import Any
from dataclasses import asdict
from datetime import datetime
try:
    from deepscholar_base.configs import Configs
    from deepscholar_base.search import search
    from deepscholar_base.filter import filter
    from deepscholar_base.final_generation import generate_intro_section, generate_insights, generate_final_report
    from deepscholar_base.taxonomize import categorize_references
except ImportError:
    from .configs import Configs
    from .search import search
    from .filter import filter
    from .final_generation import generate_intro_section, generate_insights, generate_final_report
    from .taxonomize import categorize_references

async def deepscholar_base(
    configs: Configs,
    topic: str,
    end_date: datetime | None = None,
) -> tuple[str, pd.DataFrame | None, dict[str, Any]]:
    """
    Runs the DeepScholarBase pipeline and returns the intro section, the final document dataframe, and a dictionary of statistics.
    Args:
        configs: Configs object
        topic: The user's query
        end_date: The end date for the search

    Returns:
        intro_section: The intro section of the report
        docs_df: The final document dataframe
        stats: A dictionary of intermediate results
    """
    stats = {"configs": configs.log()}
    docs_df = None
    final_report = ""
    try:
        configs.logger.info("Starting DeepScholarBase pipeline.")
        # Search and filter
        for i in range(configs.max_search_retries):
            per_retry_stats = {}
            try:
                configs.logger.info(f"Step 1: Performing search (attempt {i+1}/{configs.max_search_retries}).")
                # Search
                queries, docs_df, background = await search(configs, topic, end_date)
                per_retry_stats["search_queries"] = queries
                per_retry_stats["pre_filter_search_results"] = docs_df.to_dict(orient="records")
                per_retry_stats["search_background"] = background

                configs.logger.info(f"Step 2: Filtering search results (attempt {i+1}/{configs.max_search_retries}).")
                # Filter
                docs_df = filter(configs, docs_df, topic)
                per_retry_stats["post_filter_search_results"] = docs_df.to_dict(orient="records")
                if docs_df.empty:
                    if i < configs.max_search_retries - 1:
                        configs.logger.warning(f"No results found after {i+1} retries, retrying...")
                    continue
                else:
                    stats["search_usage"] = asdict(configs.search_lm.stats)
                    stats["filter_usage"] = asdict(configs.filter_lm.stats)
                    break
            except Exception as e:
                per_retry_stats["error"] = str(e)
                if i < configs.max_search_retries - 1:
                    configs.logger.warning(f"Error in search or filter: {e}, retrying...")
                    continue
                else:
                    raise Exception(f"Failed to search and filter after {configs.max_search_retries} retries")
            finally:
                stats[f"search_try_{i+1}"] = per_retry_stats

        if docs_df.empty:
            configs.logger.error(f"No results found after {configs.max_search_retries} retries")
            raise Exception(f"No results found after {configs.max_search_retries} retries")
        
        # Generate intro section
        configs.logger.info(f"Step 3: Found {len(docs_df)} results after search and filter. Generating intro section.")
        intro_section = await generate_intro_section(topic, docs_df, background, configs)
        stats["intro_section"] = intro_section

        # Taxonomize
        if configs.categorize_references:
            configs.logger.info("Step 4: Categorizing references.")
            docs_df, category_summaries = await categorize_references(topic, intro_section, docs_df, configs)
            cat_count = len(category_summaries) if category_summaries is not None else 0
            configs.logger.info(f"Categorized {len(docs_df)} results into {cat_count} categories.")
        else:
            configs.logger.info("Step 4: Skipping categorization as per configuration.")
            category_summaries = None

        stats["category_summaries"] = category_summaries.to_dict(orient="records") if category_summaries is not None else None
        stats["taxonomize_usage"] = asdict(configs.taxonomize_lm.stats)

        # Generate insights
        if configs.generate_insights:
            configs.logger.info("Step 5: Generating insights from documents.")
            docs_df = await generate_insights(docs_df, configs)
        else:
            configs.logger.info("Step 5: Skipping insight generation as per configuration.")

        configs.logger.info("Step 6: Generating final report.")
        final_report = await generate_final_report(docs_df, category_summaries, intro_section, configs)    
        stats["final_report"] = final_report
        stats["generation_usage"] = asdict(configs.generation_lm.stats)
        stats["total_usage"] = asdict(configs.search_lm.stats + configs.filter_lm.stats + configs.taxonomize_lm.stats + configs.generation_lm.stats)
        configs.logger.info("DeepScholarBase pipeline complete.")
    except Exception as e:
        stats["error"] = str(e)
        configs.logger.error(f"Error in DeepScholarBase pipeline: {e}")
    return final_report, docs_df, stats