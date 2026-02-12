from pydantic import BaseModel, model_validator
import logging
from lotus.models import LM
from pydantic import Field
from lotus import WebSearchCorpus
from lotus.types import ReasoningStrategy
from typing import Any

logger = logging.getLogger("deepscholar_base")

class Configs(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    logger: logging.Logger
    
    use_agentic_search: bool = True
    max_search_retries: int = 3
    use_structured_output: bool = True
    
    # Common for both agentic and recursive search
    enable_web_search: bool = True
    enable_exa_search: bool = False
    per_query_max_search_results_count: int = 10

    # Only for agentic search
    use_responses_model: bool | None = None

    # Only for recursive search
    num_search_steps: int = 3
    num_search_queries_per_step_per_corpus: int = 2
    web_corpuses: list[WebSearchCorpus] = Field(
        default_factory=lambda: [WebSearchCorpus.TAVILY]
    )
    
    # Filtering configs
    use_sem_filter: bool = True
    use_sem_topk: bool = True
    final_max_results_count: int = 30
    # Defaulted in initialize_sem_filter_kwargs and initialize_sem_topk_kwargs
    sem_filter_kwargs: dict[str, Any]
    sem_topk_kwargs: dict[str, Any]

    # Taxonomization configs
    categorize_references: bool = True
    generate_category_summary: bool = True
    
    # Generation configs
    generate_insights: bool = True

    # LM configs (defaulted in initialize_lms)
    filter_lm: LM
    search_lm: LM
    taxonomize_lm: LM
    generation_lm: LM

    @model_validator(mode="before")
    def initialize_logger(value: dict[str, Any]):
        if not value.get("logger"):
            value["logger"] = logger
        return value
    
    @model_validator(mode="before")
    def initialize_sem_filter_kwargs(value: dict[str, Any]):
        # Default to COT reasoning and return explanations
        value["sem_filter_kwargs"] = {
            "strategy": ReasoningStrategy.COT,
            "return_explanations": True,
            **value.get("sem_filter_kwargs", {}),
        }
        return value
    
    @model_validator(mode="before")
    def initialize_sem_topk_kwargs(value: dict[str, Any]):
        # Default to COT reasoning and return explanations
        value["sem_topk_kwargs"] = {
            "strategy": ReasoningStrategy.COT,
            "return_explanations": True,
            **value.get("sem_topk_kwargs", {}),
        }
        return value
    
    @model_validator(mode="before")
    def initialize_lms(value: dict[str, Any]):
        configured_lm: LM = value.get("lm") or LM(model="gpt-5-mini", temperature=1.0, reasoning_effort="low", max_tokens=10000)
        assert isinstance(configured_lm, LM), "configured_lm must be a Lotus LM"
        for lm in ["filter_lm", "search_lm", "taxonomize_lm", "generation_lm"]:
            if not value.get(lm):
                value[lm]: LM = LM(
                    model=configured_lm.model,
                    max_ctx_len=configured_lm.max_ctx_len,
                    max_batch_size=configured_lm.max_batch_size,
                    rate_limit=configured_lm.rate_limit,
                    tokenizer=configured_lm.tokenizer,
                    cache=configured_lm.cache,
                    physical_usage_limit=configured_lm.physical_usage_limit,
                    virtual_usage_limit=configured_lm.virtual_usage_limit,
                    **configured_lm.kwargs,
                )
        return value
    
    
    def log(self):
        model_dump = self.model_dump(
            mode="json",
            exclude={"logger", "search_lm", "taxonomize_lm", "generation_lm", "filter_lm"},
        )
        for lm in ["search_lm", "taxonomize_lm", "generation_lm", "filter_lm"]:
            lm_instance = getattr(self, lm)
            model_dump[lm] = {
                "model": lm_instance.model,
                "max_ctx_len": lm_instance.max_ctx_len,
                "kwargs": lm_instance.kwargs,
            }
        return model_dump