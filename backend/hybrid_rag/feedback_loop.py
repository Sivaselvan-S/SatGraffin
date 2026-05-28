import json
import os
from pathlib import Path
from typing import Dict, List, Any
import urllib.parse
import logging

logger = logging.getLogger("satgraffin.feedback")

class FeedbackLoop:
    def __init__(self, feedback_file: str = "data/feedback.jsonl"):
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)

    def submit_feedback(self, query: str, answer: str, source_links: List[str], is_thumbs_up: bool):
        """Save a user's feedback to the JSONL log file."""
        feedback_entry = {
            "query": query,
            "answer": answer,
            "sources": source_links,
            "is_thumbs_up": is_thumbs_up
        }
        try:
            with open(self.feedback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(feedback_entry) + "\n")
            logger.info(f"Feedback saved: Thumbs {'Up' if is_thumbs_up else 'Down'}")
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")

    def run_optimization_job(self, current_trusted_domains: Dict[str, int]) -> Dict[str, int]:
        """
        The 'MLOps' background job.
        Reads all feedback, analyzes which domains lead to thumbs up vs thumbs down,
        and adjusts their trust scores accordingly.
        """
        if not self.feedback_file.exists():
            return current_trusted_domains

        domain_scores = {}
        
        try:
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    sources = entry.get("sources", [])
                    is_thumbs_up = entry.get("is_thumbs_up", True)
                    
                    for url in sources:
                        try:
                            domain = urllib.parse.urlparse(url).netloc.lower()
                            if domain.startswith("www."):
                                domain = domain[4:]
                            
                            if domain not in domain_scores:
                                domain_scores[domain] = 0
                            
                            if is_thumbs_up:
                                domain_scores[domain] += 0.5  # Small reward
                            else:
                                domain_scores[domain] -= 2.0  # Heavy penalty
                        except Exception:
                            continue
        except Exception as e:
            logger.error(f"Failed to read feedback file: {e}")
            return current_trusted_domains
            
        # Update the trusted domains dictionary
        updated_domains = dict(current_trusted_domains)
        
        for domain, adjustment in domain_scores.items():
            if adjustment == 0:
                continue
                
            # Find the closest matching domain key in our dictionary if it exists
            matched_key = domain
            for key in updated_domains.keys():
                if key in domain or domain in key:
                    matched_key = key
                    break
                    
            current_score = updated_domains.get(matched_key, 3) # Default is 3
            new_score = current_score + int(adjustment)
            
            # Bound the score between 1 and 10
            new_score = max(1, min(10, new_score))
            updated_domains[matched_key] = new_score
            logger.info(f"MLOps: Adjusted {matched_key} score from {current_score} to {new_score}")
            
        # Clear the feedback file so we don't double count next time
        try:
            # We could archive it, but for simplicity we clear it
            open(self.feedback_file, 'w').close()
            logger.info("Feedback log cleared after optimization run.")
        except Exception as e:
            logger.error(f"Failed to clear feedback file: {e}")
            
        return updated_domains
