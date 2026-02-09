import pandas as pd
import lotus
try:
    from deepscholar_base.configs import Configs
    from deepscholar_base.utils.prompts import sem_filter_instruction, sem_topk_instruction
except ImportError:
    from .configs import Configs
    from .utils.prompts import sem_filter_instruction, sem_topk_instruction

def filter(
    configs: Configs,
    docs_df: pd.DataFrame,
    topic: str,
) -> pd.DataFrame:
    previous_lm = lotus.settings.lm
    lotus.settings.configure(
        lm=configs.filter_lm,
    )
    if docs_df.empty:
        return docs_df
    res_df = docs_df
    if configs.use_sem_filter:
        filter_instruction = sem_filter_instruction.format(
            user_query=topic, snippet="{snippet}"
        )
        res_df = res_df.sem_filter(
            filter_instruction, 
            **configs.sem_filter_kwargs
        )
    if configs.use_sem_topk:
        topk_instruction = sem_topk_instruction.format(
            user_query=topic, snippet="{snippet}"
        )
        configs.sem_topk_kwargs["K"] = configs.final_max_results_count
        res_df = res_df.sem_topk(
            topk_instruction, 
            **configs.sem_topk_kwargs
        )
    lotus.settings.configure(
        lm=previous_lm,
    )
    configs.logger.info(f"Filtered {len(docs_df)} results to {len(res_df)} results")
    return res_df