"""
HyDE (Hypothetical Document Embeddings) Generator
================================================
Generates a plausible hypothetical answer document for a given user query.
Embedding the hypothetical document instead of raw query vector improves dense retrieval recall.
"""

import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

HYDE_PROMPT_TEMPLATE = """You are a domain expert writing an informative reference article passage.
Write a detailed, factual, and complete hypothetical answer paragraph for the following user question.
Do NOT say "I don't know" or "as of my last update". Imagine what an ideal reference document answering this question would state.

Question: {query}

Hypothetical Answer Document:"""

class HyDEGenerator:
    """Generates hypothetical document text for dense retrieval."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = PromptTemplate(
            template=HYDE_PROMPT_TEMPLATE,
            input_variables=["query"]
        )

    def generate_hypothetical_doc(self, query: str) -> str:
        """
        Generate hypothetical answer document string.
        Falls back to original query on failure.
        """
        if not self.llm:
            return query

        try:
            formatted_prompt = self.prompt.format(query=query)
            response = self.llm.invoke(formatted_prompt)
            hypothetical_text = response.content if hasattr(response, "content") else str(response)
            
            if hypothetical_text and len(hypothetical_text.strip()) > 20:
                logger.info(f"Generated HyDE doc ({len(hypothetical_text)} chars) for query: '{query}'")
                return hypothetical_text.strip()
            return query
        except Exception as e:
            logger.warning(f"HyDE document generation failed: {e}. Falling back to original query.")
            return query

    def expand_query(self, query: str) -> str:
        """
        Local keyword-expansion alternative to full HyDE (API Saver Mode).

        Strips common stop-words, extracts meaningful tokens, and builds a
        slightly enriched query string for FAISS dense search.  Keeps the
        3-lane retrieval architecture intact without any API call.

        Latency: < 0.5 ms
        """
        import re

        _STOP_WORDS = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "of", "in", "on", "at", "to", "for", "with", "by", "from",
            "about", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "each", "this", "that", "these",
            "those", "i", "me", "my", "we", "our", "you", "your",
            "what", "which", "who", "whom", "how", "when", "where", "why",
        }

        # Extract meaningful tokens (alphanumeric, len >= 3)
        tokens = re.findall(r"[a-zA-Z0-9]{3,}", query.lower())
        keywords = [t for t in tokens if t not in _STOP_WORDS]

        if not keywords:
            return query

        # Build expanded string: original query + unique keywords (deduped)
        seen = set(query.lower().split())
        extras = [k for k in keywords if k not in seen]
        expanded = query + (" " + " ".join(extras) if extras else "")

        logger.info("[HyDE-local] Expanded query: '%s'", expanded)
        return expanded
