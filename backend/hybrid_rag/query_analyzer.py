"""
Query Analyzer
==============
Uses an LLM to analyze user queries, resolve context, and output a structured
JSON indicating whether a web search is needed, and if so, the optimized search string.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

class QueryAnalysis(BaseModel):
    """Structured output for the Query Analyzer."""
    optimized_search_query: str = Field(
        description="The rewritten query optimized for a web search engine like DuckDuckGo. Must include necessary context like temporal/spatial keywords."
    )
    is_time_sensitive: bool = Field(
        description="True if the query implies current, recent, past, or future events where the specific time matters."
    )
    spatial_context: Optional[str] = Field(
        description="The geographical or spatial context if present, else null.",
        default=None
    )
    requires_web_search: bool = Field(
        description="True if this query requires searching the web for facts. False if it's a casual greeting, a math problem, or something that doesn't need external facts."
    )

class QueryAnalyzer:
    """Analyzes user queries to generate optimized search terms and intents."""
    
    SYSTEM_TEMPLATE = """You are an expert Search Engine Optimizer and Query Analyzer.
Your task is to analyze a user's query, considering any conversation history, and output a structured JSON response.

Current Date and Time: {current_datetime}

Instructions:
1. Identify if the query requires external knowledge (facts, news, entities) or if it's just a greeting/simple chat.
2. If it requires external knowledge, rewrite the query into an `optimized_search_query` that works well in DuckDuckGo.
3. If the user uses relative time words ("current", "previous", "latest", "now", "outgoing"), resolve them into absolute context (e.g. use the current year {current_year}) and append "latest news" or similar terms if it helps.
4. Keep the optimized search query concise but highly specific.
5. If there's a conversation history, use it to resolve pronouns or missing context (e.g., if history talks about Tamil Nadu, and query says "who is the CM", the optimized query must be "Chief Minister of Tamil Nadu {current_year}").

Conversation History:
{conversation_block}

User Query: {query}
"""

    def __init__(self, llm: BaseChatModel):
        # Bind the Pydantic model to force structured JSON output
        self.structured_llm = llm.with_structured_output(QueryAnalysis)
        self.prompt = PromptTemplate(
            template=self.SYSTEM_TEMPLATE,
            input_variables=["current_datetime", "current_year", "conversation_block", "query"]
        )

    def analyze(self, query: str, conversation_context: str = "") -> QueryAnalysis:
        """
        Analyze the query and return a structured QueryAnalysis object.
        """
        now = datetime.now()
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
        current_year = now.year
        
        conversation_block = conversation_context if conversation_context else "None"

        # Format the prompt string
        prompt_text = self.prompt.format(
            current_datetime=current_datetime,
            current_year=current_year,
            conversation_block=conversation_block,
            query=query
        )
        
        # Call the LLM with structured output
        result = self.structured_llm.invoke(prompt_text)
        return result
