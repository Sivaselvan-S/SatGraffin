"""
Prompt Builder
==============
Formats packed context chunks with [Source: URL] labels and instructs the
LLM to cite sources in its answer.  Preserves the existing disambiguation
logic from the original SatGraffin prompt.
"""

from __future__ import annotations

from typing import Any
from datetime import datetime


class PromptBuilder:
    """Builds a citation-aware prompt for the LLM."""

    SYSTEM_TEMPLATE = """You are SatGraffin, an intelligent AI research assistant. Answer the user's question comprehensively and accurately based on the provided context.

Current Date and Time: {current_datetime}

Rules:
1. Use the source context below to answer. Do not include source indexes or citations in your answer.
2. If the context is insufficient, supplement with your general knowledge but clearly note it.
3. Never say "according to the provided text" — just answer naturally.
4. Provide clear, well-structured answers with examples when helpful.
5. Do not list sources at the end of your response.
6. Always consider the Current Date provided above when interpreting terms like "current", "previous", "recent", or "upcoming".

DISAMBIGUATION RULE:
If the query is ambiguous (could apply to multiple fields/domains):
- Start with "<<DISAMBIGUATION>>" on its own line
- List each interpretation as "[[OPTION: Field Name]]" followed by the explanation
- End with "<<END_DISAMBIGUATION>>"

If the user has a CONTEXT PREFERENCE set (shown below), focus your answer ONLY on that domain.
If no context preference is set and the query is specific enough, answer normally.

{conversation_block}
{preference_block}
--- SOURCE CONTEXT ---
{context_block}
--- END CONTEXT ---

User's Question: {question}

Answer:"""

    def build(
        self,
        query: str,
        packed_chunks: list[dict[str, Any]],
        conversation_context: str = "",
        context_preference: str | None = None,
    ) -> str:
        """
        Build the final prompt string.

        Parameters
        ----------
        query : str
            The user's question.
        packed_chunks : list[dict]
            Chunks from ContextPacker, each with "text" and "metadata".
        conversation_context : str
            Formatted previous conversation turns (may be empty).
        context_preference : str or None
            User's disambiguation preference (e.g. "Distributed Computing").

        Returns
        -------
        str
            The fully formatted prompt ready for the LLM.
        """
        # Build source blocks
        context_parts: list[str] = []
        for i, chunk in enumerate(packed_chunks, start=1):
            source = chunk.get("metadata", {}).get("source", "unknown")
            title = chunk.get("metadata", {}).get("title", "")
            label = f"[Source: {source}]"
            if title:
                label += f" ({title})"
            context_parts.append(f"{label}\n{chunk['text']}")

        context_block = "\n\n".join(context_parts) if context_parts else "(No source context available)"

        # Conversation block
        conversation_block = ""
        if conversation_context:
            conversation_block = f"Previous conversation:\n{conversation_context}\n"

        # Preference block
        preference_block = ""
        if context_preference:
            preference_block = f"[CONTEXT PREFERENCE: {context_preference}]\n"

        return self.SYSTEM_TEMPLATE.format(
            question=query,
            context_block=context_block,
            conversation_block=conversation_block,
            preference_block=preference_block,
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    @staticmethod
    def extract_source_map(packed_chunks: list[dict[str, Any]]) -> dict[int, str]:
        """Return a {1: url, 2: url, ...} map for post-processing citations."""
        return {
            i: chunk.get("metadata", {}).get("source", "unknown")
            for i, chunk in enumerate(packed_chunks, start=1)
        }
