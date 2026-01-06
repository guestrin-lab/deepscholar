from deepscholar_base.settings import Configs
from deepscholar_base import deepscholar_base
from dotenv import load_dotenv
from lotus.models import LM
import asyncio
load_dotenv()

settings = Configs(
    lm=LM(model="gpt-5-mini", temperature=1.0, reasoning_effort="low", max_tokens=10000)
)


async def main():
    final_output, docs_df, stats = await deepscholar_base(
        settings, 
        "What are the latest developments in the field of AI."
    )
    print(f"Final output: {final_output}")
    print(f"Docs DataFrame: {len(docs_df)}")
    print(docs_df.head())
    print(f"Stats: {stats}")

if __name__ == "__main__":
    asyncio.run(main())