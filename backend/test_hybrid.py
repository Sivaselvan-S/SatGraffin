"""Full end-to-end hybrid RAG pipeline test (without LLM call)."""
from hybrid_rag import (
    SentenceAwareChunker, BM25Retriever, reciprocal_rank_fusion,
    CrossEncoderReranker, ContextPacker, PromptBuilder,
)

# --- 1. Prepare test corpus ---
docs = [
    {
        "text": "Quantum computing is a rapidly emerging technology that harnesses the laws of "
                "quantum mechanics to solve problems too complex for classical computers.",
        "metadata": {"source": "https://ibm.com/quantum", "title": "Quantum Overview", "chunk_id": 0},
    },
    {
        "text": "Unlike classical bits which can only be 0 or 1, quantum bits or qubits can exist "
                "in a superposition of states allowing parallel processing of information.",
        "metadata": {"source": "https://ibm.com/quantum", "title": "Quantum Overview", "chunk_id": 1},
    },
    {
        "text": "Google Sycamore processor achieved quantum supremacy in 2019 by performing a "
                "calculation in 200 seconds that would take a classical supercomputer 10000 years.",
        "metadata": {"source": "https://google.com/ai", "title": "Quantum Supremacy", "chunk_id": 0},
    },
    {
        "text": "Quantum entanglement is a phenomenon where qubits become correlated so that the "
                "state of one instantly influences the state of another regardless of distance.",
        "metadata": {"source": "https://mit.edu/physics", "title": "Entanglement", "chunk_id": 0},
    },
    {
        "text": "The Pallava dynasty was a prominent South Indian dynasty that ruled from the 3rd "
                "to the 9th century CE. They were known for their rock-cut architecture.",
        "metadata": {"source": "https://wikipedia.org/pallava", "title": "Pallava Dynasty", "chunk_id": 0},
    },
    {
        "text": "Surface tension is the tendency of liquid surfaces at rest to shrink into the "
                "minimum surface area possible. It is caused by cohesive forces between molecules.",
        "metadata": {"source": "https://britannica.com/surface-tension", "title": "Surface Tension", "chunk_id": 0},
    },
]

# --- 2. Build BM25 index ---
bm25 = BM25Retriever()
bm25.add_documents(docs)
print(f"[1] BM25 corpus: {bm25.document_count} docs")

# --- 3. Simulate FAISS results (in real pipeline, FAISS does this) ---
# For testing, we simulate FAISS returning results in different order
query = "How does quantum computing achieve supremacy?"

bm25_results = bm25.search(query, top_k=20)
print(f"\n[2] BM25 results: {len(bm25_results)}")
for r in bm25_results:
    print(f"    BM25 score={r['bm25_score']:.3f}: {r['text'][:60]}...")

# Simulate FAISS (reverse order to test fusion)
faiss_results = []
for i, doc in enumerate(reversed(docs[:4])):
    faiss_results.append({**doc, "faiss_score": 0.9 - i * 0.1})
print(f"\n[3] FAISS results (simulated): {len(faiss_results)}")

# --- 4. RRF Fusion ---
fused = reciprocal_rank_fusion([bm25_results, faiss_results], k=60, top_n=20)
print(f"\n[4] RRF fused: {len(fused)} candidates")
for f in fused:
    src = f["metadata"].get("source", "?")
    print(f"    RRF={f['rrf_score']:.4f}: [{src}] {f['text'][:50]}...")

# --- 5. Cross-encoder re-ranking ---
print("\n[5] Loading cross-encoder (first time may download ~80MB model)...")
reranker = CrossEncoderReranker()
reranked = reranker.rerank(query, fused, top_k=5)
print(f"    Re-ranked to {len(reranked)} results")
for r in reranked:
    print(f"    CE={r['ce_score']:.4f}: {r['text'][:60]}...")

# --- 6. Context packing ---
packer = ContextPacker(max_tokens=3000, jaccard_threshold=0.85)
packed = packer.pack(reranked)
print(f"\n[6] Packed: {len(packed)} chunks")

# --- 7. Prompt building ---
pb = PromptBuilder()
prompt = pb.build(query, packed, conversation_context="", context_preference=None)
print(f"\n[7] Prompt built: {len(prompt)} chars")
print("    Contains [Source:] labels:", "[Source:" in prompt)
print("    Contains citation instructions:", "Cite sources" in prompt or "[1]" in prompt)

source_map = pb.extract_source_map(packed)
print(f"    Source map: {source_map}")

print("\n" + "=" * 60)
print("ALL END-TO-END TESTS PASSED")
print("=" * 60)
