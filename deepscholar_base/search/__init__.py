from deepscholar_base.configs import Configs
import pandas as pd
from datetime import datetime

async def search(
    configs: Configs,
    topic: str,
    end_date: datetime | None = None,
) -> tuple[list[str], pd.DataFrame, str]:
    if configs.use_agentic_search:
        try:
            from deepscholar_base.search.agentic_search import agentic_search
        except ImportError:
            from .agentic_search import agentic_search
        return await agentic_search(configs, topic, end_date)
    else:
        try:
            from deepscholar_base.search.recursive_search import recursive_search
        except ImportError:
            from .recursive_search import recursive_search
        return await recursive_search(configs, topic, end_date)