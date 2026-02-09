import pandas as pd
import datetime
import re
import lotus
from lotus.models import LM
try:
    from deepscholar_base.utils.prompts import section_writer_instructions, citation_guidelines, category_summarization_instruction
except ImportError:
    from .prompts import section_writer_instructions, citation_guidelines, category_summarization_instruction

async def generate_section_summary(
    topic: str,
    docs_df: pd.DataFrame,
    section_instructions: str,
    existing_content: str,
    citation_guidelines: str = "",
    group_by: list[str] | None = None,
    lm: LM | None = None,
) -> str | pd.DataFrame:
    """
    Generate a summary of the documents for a report section. docs_df is assumed to have a column "context" that contains the title, url, and snippet of the document.
    """
    previous_lm = lotus.settings.lm
    if lm is not None:
        lotus.settings.configure(
            lm=lm,
        )
    topic = input_sanitization(topic)
    section_instructions = input_sanitization(section_instructions)
    existing_content = input_sanitization(existing_content)
    citation_guidelines = input_sanitization(citation_guidelines)
    agg_instruction = section_writer_instructions.format(
        topic=topic,
        section_instructions=section_instructions,
        existing_content=existing_content,
        context="{context}",
        citation_guidelines=citation_guidelines,
    )
    res: pd.DataFrame = docs_df.sem_agg(
        agg_instruction, suffix="summary", group_by=group_by
    )
    if lm is not None:
        lotus.settings.configure(
            lm=previous_lm,
        )
    if group_by is None:
        return res.iloc[0].summary
    else:
        return res


async def generate_section_summary_with_citations(
    topic: str,
    docs_df: pd.DataFrame,
    section_instructions: str,
    existing_content: str,
    group_by: list[str] | None = None,
    lm: LM | None = None,
) -> str | pd.DataFrame:
    """
    Generate a summary of the documents for a report section, with citations.
    """

    docs_df = _prepare_df_for_citation(docs_df)
    summary = await generate_section_summary(
        topic,
        docs_df,
        section_instructions,
        existing_content,
        citation_guidelines,
        group_by,
        lm,
    )
    if group_by is None:
        summary_with_urls = _postprocess_citation(docs_df, summary)
        return summary_with_urls
    else:
        summary["summary"] = summary["summary"].map(
            lambda x: _postprocess_citation(docs_df, x)
        )
        return summary

async def generate_category_summary_with_citations(
    docs_df: pd.DataFrame,
    topic: str,
    lm: LM | None = None,
):
    previous_lm = lotus.settings.lm
    if lm is not None:
        lotus.settings.configure(
            lm=lm,
        )
    df = _prepare_df_for_citation(docs_df)
    agg_instruction = category_summarization_instruction.format(
        query=topic, citation_guidelines=citation_guidelines
    )
    summaries: pd.DataFrame = df.sem_agg(
        agg_instruction, group_by=["category"], suffix="summary"
    )
    summaries["summary"] = summaries["summary"].map(
        lambda x: _postprocess_citation(df, x)
    )
    if lm is not None:
        lotus.settings.configure(
            lm=previous_lm,
        )
    return summaries

def _prepare_df_for_citation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame for citation by adding citation numbers and URLs.
    """
    if "_citation_number" not in df.columns:
        df = df.copy()
        df["_citation_number"] = [i + 1 for i in range(len(df))]
    return df


def _postprocess_citation(df: pd.DataFrame, summary: str) -> str:
    """
    Postprocess the citation by replacing the citation numbers with the actual citations.
    """

    def get_citation_string(row: pd.Series, number: int):
        citation_string = f"{number}"
        if (
            "authors" in row.index
            and row["authors"] is not None
            and row["authors"] != ""
            and row["authors"] != "nan"
        ):
            if isinstance(row["authors"], str):
                citation_string = row["authors"].split(",")[0]
            elif isinstance(row["authors"], list) or isinstance(row["authors"], tuple):
                citation_string = row["authors"][0]
            else:
                citation_string = f"{row['authors']}"
        elif (
            "id" in row.index
            and row["id"] is not None
            and row["id"] != ""
            and row["id"] != "nan"
        ):
            citation_string = row["id"]

        if (
            "date" in row.index
            and row["date"] is not None
            and row["date"] != ""
            and row["date"] != "nan"
        ):
            try:
                date_str = datetime.datetime.strptime(
                    row["date"], "%Y-%m-%d %H:%M:%S%z"
                ).strftime("%Y-%m-%d")
                citation_string = f"{citation_string}' {date_str}"
            except Exception as e:
                pass
        if citation_string == "nan":
            citation_string = f"{number}"
        return citation_string

    def replace_citation(match):
        citation_num = int(match.group(1)) - 1
        if 0 <= citation_num < len(df):
            url = df[df["_citation_number"] == citation_num + 1]["url"].values[0]
            citation_string = get_citation_string(
                df[df["_citation_number"] == citation_num + 1].iloc[0], citation_num + 1
            )
            return f"\[[{citation_string}]({url})\]"
        return ""  # Remove citation if not found

    # parse 【】, unicode = 12304, 12305
    summary = summary.replace("【", "[").replace("】", "]")
    summary = re.sub(r"\[(\d+)\]", replace_citation, summary)

    return summary



def input_sanitization(input: str) -> str:
    input = input.strip()
    # Regular expression pattern to match variables in brackets not escaped by double brackets
    pattern = r"(?<!\{)\{(?!\{)(.*?)(?<!\})\}(?!\})"
    # Find all matches in the text
    matches = re.findall(pattern, input)
    # Escape all matches by replacing {match} with {{match}}
    for match in matches:
        input = input.replace(f"{{{match}}}", f"{{{{{match}}}}}")
    return input
