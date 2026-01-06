import pandas as pd
import lotus
from tabulate import tabulate

from deepscholar_base.utils.summary_generation import generate_section_summary_with_citations
try:
    from deepscholar_base.configs import Configs
    from deepscholar_base.utils.prompts import intro_section_instructions, insight_columns_to_descriptions
    from deepscholar_base.utils.summary_generation import generate_section_summary_with_citations
except ImportError:
    from .configs import Configs
    from .utils.prompts import intro_section_instructions, insight_columns_to_descriptions
    from .utils.summary_generation import generate_section_summary_with_citations
    
async def generate_intro_section(
    topic: str,
    docs_df: pd.DataFrame,
    existing_content: str,
    configs: Configs,
) -> str:
    intro_section: str = await generate_section_summary_with_citations(
        topic,
        docs_df,
        intro_section_instructions,
        existing_content,
        lm=configs.generation_lm,
    )
    return intro_section


async def generate_insights(
    docs_df: pd.DataFrame,
    configs: Configs,
) -> pd.DataFrame:
    previous_lm = lotus.settings.lm
    lotus.settings.configure(
        lm=configs.generation_lm,
    )
    docs_df = docs_df.sem_extract(["snippet"], insight_columns_to_descriptions)
    lotus.settings.configure(
        lm=previous_lm,
    )
    return docs_df


async def generate_final_report(
    docs_df: pd.DataFrame,
    category_summaries: pd.DataFrame | None,
    intro_section: str,
    configs: Configs,
) -> str:
    papers_string = ""
    outline = ""
    
    if configs.categorize_references and "category" in docs_df.columns:
        unique_categories: list[str] = pd.unique(docs_df.category).tolist()
        configs.logger.info(f"Unique categories: {unique_categories}")

        outline = "### Outline\n"
        papers_string = "\n"
        for category in unique_categories:
            category_df = docs_df[docs_df.category == category]
            category_df = category_df.fillna("")
            num_papers = len(docs_df[docs_df.category == category])
            papers_table = _format_df_to_string(category_df, configs)
            if category_summaries is not None:
                category_summary: str = category_summaries[
                    category_summaries.category == category
                ].summary.iloc[0] + "\n"
            else:
                category_summary = ""
            category_id = (
                category.lower()
                .replace(" ", "-")
                .replace(":", "")
                .replace(",", "")
                .strip("'")
                .strip('"')
            )
            outline += f"- [{category} [{num_papers} papers]](#category-{category_id})\n"
            papers_string += f"### <a id='category-{category_id}'></a>{category}: {len(category_df)} papers found.\n{category_summary}{papers_table}\n"
    else:
        papers_string = "\n" + _format_df_to_string(docs_df, configs)

    papers_string = intro_section + "\n" + "## Research papers\n" + outline + papers_string

    return papers_string

def _format_df_to_string(df: pd.DataFrame, configs: Configs) -> str:
    headers = ["reference", "date"] + list(insight_columns_to_descriptions.keys())
    docs_df = df.copy()
    docs_df["reference"] = docs_df.apply(
        lambda x: f"[{x['title']}]({x['url']})", axis=1
    )
    # Ensure only existing columns are selected, warn if any headers are missing
    existing_headers = [col for col in headers if col in docs_df.columns]
    missing_headers = [col for col in headers if col not in docs_df.columns]
    if missing_headers:
        configs.logger.warning(f"Columns missing in DataFrame and excluded from output: {missing_headers}")

    display_df = docs_df[existing_headers].reset_index(drop=True)
    if "date" in display_df.columns:
        display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%m/%d/%y")
    
    # Escape '|' in all values in display_df
    for col in display_df.columns:
        display_df[col] = display_df[col].astype(str).str.replace('|', '\\|')
    
    table = tabulate(
        display_df,
        headers=existing_headers,
        tablefmt="github",
        stralign="left",
        maxcolwidths=[None] * len(existing_headers),
    )
    return table
