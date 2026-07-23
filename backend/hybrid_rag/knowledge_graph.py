"""
Mini Knowledge Graph Store
===========================
Lightweight in-memory entity-relationship graph using networkx.
Extracts (entity -> relationship -> entity) triples using Gemini structured output,
enabling multi-hop entity traversal and multi-entity context synthesis.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import networkx as nx
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
import logging

logger = logging.getLogger(__name__)

class EntityTriple(BaseModel):
    subject: str = Field(description="Subject entity name")
    relation: str = Field(description="Relationship or connection verb/phrase")
    object: str = Field(description="Object entity or property name")

class TripleExtraction(BaseModel):
    triples: List[EntityTriple] = Field(description="Extracted knowledge graph triples")

class KnowledgeGraphStore:
    """Stores and queries entity relationship triples."""

    EXTRACTION_PROMPT = """Extract up to 6 key entity relationship triples from the text below.

Text: {text}

Output JSON with 'triples': list of {{ "subject": string, "relation": string, "object": string }}
"""

    def __init__(self, storage_path: Path | str = "data/knowledge_graph.json"):
        self.storage_path = Path(storage_path)
        self.graph = nx.DiGraph()
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for edge in data.get("edges", []):
                        self.graph.add_edge(edge["s"], edge["o"], relation=edge["r"])
                logger.info(f"Loaded Knowledge Graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            except Exception as e:
                logger.warning(f"Failed to load knowledge graph: {e}")

    def save(self):
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            edges = [
                {"s": u, "o": v, "r": data.get("relation", "connected_to")}
                for u, v, data in self.graph.edges(data=True)
            ]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"edges": edges}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")

    def extract_and_add(self, text: str, llm: BaseChatModel) -> int:
        """Extract triples from text using LLM and add to graph."""
        if not text or len(text.strip()) < 50:
            return 0
        try:
            structured_llm = llm.with_structured_output(TripleExtraction)
            prompt = PromptTemplate(
                template=self.EXTRACTION_PROMPT,
                input_variables=["text"]
            ).format(text=text[:1500])

            res: TripleExtraction = structured_llm.invoke(prompt)
            added_count = 0
            for t in res.triples:
                s, r, o = t.subject.strip(), t.relation.strip(), t.object.strip()
                if s and o:
                    self.graph.add_edge(s, o, relation=r)
                    added_count += 1
            if added_count > 0:
                self.save()
            return added_count
        except Exception as e:
            logger.debug(f"Knowledge graph triple extraction skipped: {e}")
            return 0

    def query_entity_context(self, entities: List[str], depth: int = 1) -> str:
        """
        Find entity subgraphs using BFS up to `depth` hops and format as text context.
        depth=1 returns direct neighbours; depth=2 also returns neighbours-of-neighbours, etc.
        """
        triples_found = []
        visited_edges: set = set()

        for entity in entities:
            # Find all nodes whose name contains the entity string (case-insensitive)
            seed_nodes = [n for n in self.graph.nodes if entity.lower() in str(n).lower()]

            # BFS frontier: list of (node, current_depth)
            frontier = [(n, 0) for n in seed_nodes]
            visited_nodes: set = set(seed_nodes)

            while frontier:
                node, current_depth = frontier.pop(0)
                if current_depth >= depth:
                    continue
                for neighbor in self.graph.neighbors(node):
                    edge_key = (node, neighbor)
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        rel = self.graph[node][neighbor].get("relation", "is related to")
                        triples_found.append(f"({node}) --[{rel}]--> ({neighbor})")
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        frontier.append((neighbor, current_depth + 1))

        if triples_found:
            return "Knowledge Graph Relationships:\n" + "\n".join(triples_found[:15])
        return ""

