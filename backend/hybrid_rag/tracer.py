"""
RAG Observability & Pipeline Tracer
===================================
Records detailed step timing, search hit count, CRAG score metrics, and token generation speeds.
Persists trace logs to JSONL for quality analysis and latency profiling.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class PipelineTracer:
    """Trace collector for a single query execution."""

    def __init__(self, query: str, user_id: Optional[str] = None):
        self.query = query
        self.user_id = user_id
        self.start_time = time.time()
        self.steps: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def log_step(self, step_name: str, duration_ms: float, details: Optional[Dict[str, Any]] = None):
        self.steps.append({
            "step": step_name,
            "duration_ms": round(duration_ms, 2),
            "details": details or {}
        })

    def finish(self, total_tokens: int = 0) -> Dict[str, Any]:
        total_duration = round((time.time() - self.start_time) * 1000, 2)
        trace_record = {
            "query": self.query,
            "user_id": self.user_id,
            "timestamp": time.time(),
            "total_duration_ms": total_duration,
            "total_tokens": total_tokens,
            "steps": self.steps,
            "metadata": self.metadata
        }
        self._append_to_file(trace_record)
        return trace_record

    def _append_to_file(self, record: Dict[str, Any]):
        try:
            # Anchor to the backend directory (two levels up from this module file)
            path = Path(__file__).parent.parent / "routing_store" / "traces.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"Failed to record trace log: {e}")

def get_recent_traces(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve last N traces from disk."""
    path = Path(__file__).parent.parent / "routing_store" / "traces.jsonl"

    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        traces = [json.loads(line) for line in lines[-limit:]]
        traces.reverse()
        return traces
    except Exception as e:
        logger.error(f"Error reading trace logs: {e}")
        return []

# Alias for backwards compatibility
QueryTracer = PipelineTracer

