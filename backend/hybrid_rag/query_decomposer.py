"""
Query Decomposer Module
=======================
Decomposes complex multi-part, comparative, or multi-hop queries into atomic,
independently searchable sub-queries with type classification.
"""

from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
import logging

logger = logging.getLogger(__name__)

SubQueryType = Literal["factual", "comparative", "temporal", "synthesis"]

class SubQuery(BaseModel):
    query: str = Field(description="The standalone atomic search query")
    query_type: SubQueryType = Field(description="The category of sub-query: factual, comparative, temporal, or synthesis")

class DecompositionResult(BaseModel):
    is_complex: bool = Field(description="True if query requires multiple sub-search queries")
    sub_queries: List[SubQuery] = Field(description="List of decomposed sub-queries")

class QueryDecomposer:
    """Decomposes complex user queries into structured sub-queries."""

    DECOMPOSE_PROMPT = """You are a Search Engine Query Decomposition Agent.
Given a user query and optional conversation history, evaluate if the query is complex (multi-part, comparative, or multi-hop).
If complex, decompose it into 2 to 4 atomic standalone sub-queries.

User Query: {query}
Context: {context}

Output JSON matching the DecompositionResult schema:
- is_complex: boolean
- sub_queries: list of {{ "query": string, "query_type": "factual"|"comparative"|"temporal"|"synthesis" }}
"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        try:
            self.structured_llm = llm.with_structured_output(DecompositionResult)
        except Exception as e:
            logger.warning(f"Could not initialize structured output for QueryDecomposer: {e}")
            self.structured_llm = None

    def decompose(self, query: str, context: str = "") -> DecompositionResult:
        if not self.structured_llm:
            return DecompositionResult(is_complex=False, sub_queries=[SubQuery(query=query, query_type="factual")])

        try:
            prompt = PromptTemplate(
                template=self.DECOMPOSE_PROMPT,
                input_variables=["query", "context"]
            ).format(query=query, context=context or "None")

            res = self.structured_llm.invoke(prompt)
            if isinstance(res, DecompositionResult):
                return res
            return DecompositionResult(is_complex=False, sub_queries=[SubQuery(query=query, query_type="factual")])
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return DecompositionResult(is_complex=False, sub_queries=[SubQuery(query=query, query_type="factual")])
