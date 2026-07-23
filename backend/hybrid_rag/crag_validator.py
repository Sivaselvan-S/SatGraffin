"""
CRAG (Corrective RAG) Validator
================================
Grades retrieved context chunks for relevance to the query.
If overall context relevance is insufficient, triggers query reformulation or web search fallback.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
import logging

logger = logging.getLogger(__name__)

class ChunkGrade(BaseModel):
    chunk_index: int = Field(description="Index of the chunk being evaluated")
    relevance_score: float = Field(description="Relevance score between 0.0 (irrelevant) and 1.0 (highly relevant)")
    reason: str = Field(description="Brief justification for the relevance score")

class CRAGEvaluation(BaseModel):
    grades: List[ChunkGrade] = Field(description="List of grades for each evaluated chunk")
    action: str = Field(description="Recommended action: 'CORRECT' (proceed), 'AMBIGUOUS' (use available), 'REFORMULATE' (web search fallback required)")
    refined_query: Optional[str] = Field(default=None, description="Refined query string if action is REFORMULATE")

class CRAGValidator:
    """Corrective RAG relevance grader and fallback trigger."""

    CRAG_PROMPT = """You are a Corrective RAG (CRAG) Relevance Evaluator.
Given a user query and candidate retrieved text passages, score each passage's relevance to answering the query.

User Query: {query}

Passages to evaluate:
{passages_block}

Instructions:
1. Assign a relevance_score (0.0 to 1.0) to each passage.
2. If at least 2 passages score >= 0.5, set action to 'CORRECT'.
3. If some passages score 0.3-0.5, set action to 'AMBIGUOUS'.
4. If most passages score < 0.3, set action to 'REFORMULATE' and provide a refined_query optimized for web search.
"""

    def __init__(self, llm: BaseChatModel, min_score_threshold: float = 0.3):
        self.llm = llm
        self.min_score_threshold = min_score_threshold
        try:
            self.structured_llm = llm.with_structured_output(CRAGEvaluation)
        except Exception as e:
            logger.warning(f"Could not initialize structured output for CRAGValidator: {e}")
            self.structured_llm = None

    def evaluate(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate retrieved chunks.
        Returns dict with:
          - filtered_chunks: List of chunks meeting score threshold
          - action: 'CORRECT' | 'AMBIGUOUS' | 'REFORMULATE'
          - refined_query: Optional[str]
        """
        if not chunks:
            return {"filtered_chunks": [], "action": "REFORMULATE", "refined_query": query}

        if not self.structured_llm:
            # Fallback if structured output unavailable: keep all chunks
            return {"filtered_chunks": chunks, "action": "CORRECT", "refined_query": None}

        try:
            passages_lines = []
            for idx, c in enumerate(chunks[:8]): # Grade top 8 chunks max
                txt = c.get("text", "")[:300].replace("\n", " ")
                passages_lines.append(f"Passage [{idx}]: {txt}")

            passages_block = "\n".join(passages_lines)
            prompt = PromptTemplate(
                template=self.CRAG_PROMPT,
                input_variables=["query", "passages_block"]
            ).format(query=query, passages_block=passages_block)

            eval_res: CRAGEvaluation = self.structured_llm.invoke(prompt)
            
            # Map grades back to chunks
            grade_map = {g.chunk_index: g.relevance_score for g in eval_res.grades}
            
            filtered = []
            for idx, chunk in enumerate(chunks):
                score = grade_map.get(idx, 0.5) # default 0.5 if omitted
                if score >= self.min_score_threshold:
                    chunk_copy = dict(chunk)
                    chunk_copy["crag_score"] = score
                    filtered.append(chunk_copy)

            if not filtered:
                filtered = chunks[:2] # safety fallback so pipeline isn't starved

            logger.info(f"CRAG Evaluation: action={eval_res.action}, passed={len(filtered)}/{len(chunks)}")
            return {
                "filtered_chunks": filtered,
                "action": eval_res.action,
                "refined_query": eval_res.refined_query
            }

        except Exception as e:
            logger.error(f"CRAG evaluation failed: {e}")
            return {"filtered_chunks": chunks, "action": "CORRECT", "refined_query": None}

    def local_evaluate(self, query: str, chunks: list) -> dict:
        """
        Local CPU-only relevance grading using cross-encoder ``ce_score``.

        The CrossEncoderReranker already attached a ``ce_score`` (raw logit from
        ms-marco-MiniLM-L-6-v2) to every chunk.  We use fixed thresholds that
        closely mirror what the LLM CRAG would decide:

            ce_score > 0.0   → highly relevant  (CORRECT)
            ce_score > -2.0  → possibly relevant (AMBIGUOUS)
            ce_score ≤ -2.0  → irrelevant        (REFORMULATE)

        No new model load needed — scores are already computed.
        Latency: < 0.1 ms  (simple list comprehension).
        """
        if not chunks:
            return {"filtered_chunks": [], "action": "REFORMULATE", "refined_query": query}

        HIGH_THRESH  =  0.0   # ms-marco: > 0 → passage is clearly relevant
        LOW_THRESH   = -2.0   # ms-marco: < -2 → passage is noise

        scored = []
        for chunk in chunks:
            ce = chunk.get("ce_score", 0.0)
            chunk_copy = dict(chunk)
            chunk_copy["crag_score"] = float(ce)
            scored.append((ce, chunk_copy))

        high_rel  = [c for (s, c) in scored if s >  HIGH_THRESH]
        mid_rel   = [c for (s, c) in scored if LOW_THRESH < s <= HIGH_THRESH]
        irrelevant = [c for (s, c) in scored if s <= LOW_THRESH]

        if len(high_rel) >= 2:
            action = "CORRECT"
            filtered = high_rel + mid_rel
        elif high_rel or mid_rel:
            action = "AMBIGUOUS"
            filtered = high_rel + mid_rel
        else:
            action = "REFORMULATE"
            filtered = chunks[:2]   # safety: always return something

        logger.info(
            "[LocalCRAG] action=%s high=%d mid=%d irrelevant=%d",
            action, len(high_rel), len(mid_rel), len(irrelevant),
        )
        return {
            "filtered_chunks": filtered,
            "action": action,
            "refined_query": None,
        }
