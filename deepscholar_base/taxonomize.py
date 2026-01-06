import pandas as pd
import re
from pydantic import BaseModel, Field
import lotus
try:
    from deepscholar_base.configs import Configs
    from deepscholar_base.utils.prompts import taxonomize_category_creation_prompt, match_references_to_categories_prompt
    from deepscholar_base.utils.summary_generation import input_sanitization
    from deepscholar_base.utils.summary_generation import generate_category_summary_with_citations
except ImportError:
    from .configs import Configs
    from .utils.prompts import taxonomize_category_creation_prompt, match_references_to_categories_prompt
    from .utils.summary_generation import input_sanitization
    from .utils.summary_generation import generate_category_summary_with_citations



async def categorize_references(
    topic: str,
    intro_section: str,
    docs_df: pd.DataFrame,
    configs: Configs,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    categories = await generate_categories(topic, intro_section, configs)
    docs_df = match_references_to_categories(docs_df, categories, configs)
    if configs.generate_category_summary:
        category_summaries = await generate_category_summary_with_citations(docs_df, topic, configs.taxonomize_lm)
        return docs_df, category_summaries
    return docs_df, None

class Categories(BaseModel):
    categories: list[str] = Field(
        description="list of categories.",
    )

async def generate_categories(
    topic: str,
    intro_section: str,
    configs: Configs,
) -> list[str]:
    system_prompt = taxonomize_category_creation_prompt
    user_prompt = f"""
    Query: {topic}
    Answer: {intro_section}
    """
    if configs.use_structured_output:
        categories = configs.taxonomize_lm.get_completion(system_prompt, user_prompt, response_format=Categories)
        assert isinstance(categories, Categories)
    else:
        category_str = configs.taxonomize_lm.get_completion(system_prompt, user_prompt)
        category_str = re.sub(r"^\d+\.\s+", "", category_str, flags=re.MULTILINE)
        categories = Categories(categories=category_str.split("\n"))
    return categories.categories


def match_references_to_categories(
    docs_df: pd.DataFrame, 
    categories: list[str],
    configs: Configs,
) -> pd.DataFrame:
    previous_lm = lotus.settings.lm
    lotus.settings.configure(
        lm=configs.taxonomize_lm,
    )
    def get_category(x):
        try:
            # remove any non-numeric characters
            x = re.sub(r"\D", "", x)
            int_x = int(x)
            return categories[int_x - 1]
        except Exception:
            return "Others"

    categories_str = input_sanitization(
        "\n".join([f"{i + 1}. {cat}" for i, cat in enumerate(categories)])
    )
    docs_df = docs_df.sem_map(
        match_references_to_categories_prompt.format(categories_str=categories_str),
        suffix="category_number",
    )
    docs_df["category"] = docs_df["category_number"].map(get_category)
    docs_df.drop(columns=["category_number"], inplace=True)
    lotus.settings.configure(
        lm=previous_lm,
    )
    return docs_df
