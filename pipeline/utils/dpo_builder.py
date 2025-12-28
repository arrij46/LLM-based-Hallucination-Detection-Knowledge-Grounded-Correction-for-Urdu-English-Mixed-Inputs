"""
DPO Training Data Builder (UPDATED)
Owner: Rimsha Azam

UPDATES:
- Lowered confidence threshold from 0.8 to 0.7 (matches Stage 4 & 5)
- Better quality checks
- Added data validation
- Improved statistics
"""

import json
from datetime import datetime
from pathlib import Path


class DPOBuilder:
    """
    Builds Direct Preference Optimization (DPO) training data pairs.
    Only saves high-quality correction pairs for model fine-tuning.
    """

    def __init__(self, output_file="dpo_training_data.jsonl"):
        self.output_file = output_file
        self.preference_pairs = []
        self.rejected_pairs = []  # Track rejected pairs for debugging

    def create_preference_pair(
        self,
        query,
        hallucinated_response,
        corrected_response,
        verified_fact,
        error_type,
        verification_confidence,
        quality_passed
    ):
        """
        Create a DPO training pair if quality checks pass.
        
        Args:
            query: Original user query
            hallucinated_response: Incorrect response
            corrected_response: Corrected response
            verified_fact: Verified fact used for correction
            error_type: Type of hallucination
            verification_confidence: Confidence score from Stage 4
            quality_passed: Whether quality checks passed
        
        Returns:
            Created pair dict or None if rejected
        """
        
        # SAFETY CHECKS (order matters)
        
        # Check 1: Quality must pass
        if not quality_passed:
            self._reject_pair(query, "quality_check_failed")
            return None
        
        # Check 2: Confidence threshold (LOWERED from 0.8 to 0.7)
        if verification_confidence < 0.70:
            self._reject_pair(query, f"low_confidence_{verification_confidence:.2f}")
            return None
        
        # Check 3: No verification errors
        if error_type == "verification_error":
            self._reject_pair(query, "verification_error")
            return None
        
        # Check 4: Valid responses
        if not corrected_response or not hallucinated_response:
            self._reject_pair(query, "empty_response")
            return None
        
        # Check 5: Responses must be different
        if corrected_response.strip().lower() == hallucinated_response.strip().lower():
            self._reject_pair(query, "identical_responses")
            return None
        
        # Check 6: Corrected response must contain verified fact content
        if not self._contains_fact_content(corrected_response, verified_fact):
            self._reject_pair(query, "fact_not_in_correction")
            return None
        
        # All checks passed - create pair
        pair = {
            "prompt": query,
            "chosen": corrected_response,
            "rejected": hallucinated_response,
            "metadata": {
                "verified_fact": verified_fact,
                "error_type": error_type,
                "verification_confidence": round(verification_confidence, 3),
                "created_at": datetime.now().isoformat(),
                "quality_score": 1.0 if quality_passed else 0.0
            }
        }
        
        self.preference_pairs.append(pair)
        return pair

    def _contains_fact_content(self, correction, fact):
        """Check if correction contains significant fact content"""
        if not correction or not fact:
            return False
        
        correction_words = set(correction.lower().split())
        fact_words = set(fact.lower().split())
        
        # Remove common words
        stop_words = {'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 
                     'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                     'hai', 'hain', 'ka', 'ki', 'ke'}
        
        correction_words -= stop_words
        fact_words -= stop_words
        
        # At least 40% overlap required
        if not fact_words:
            return True
        
        overlap = len(correction_words & fact_words)
        return overlap >= len(fact_words) * 0.4

    def _reject_pair(self, query, reason):
        """Track rejected pairs for debugging"""
        self.rejected_pairs.append({
            "query": query[:50] + "..." if len(query) > 50 else query,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })

    def save_to_file(self, output_path=None):
        """
        Save all preference pairs to JSONL file.
        
        Args:
            output_path: Optional custom output path
        
        Returns:
            Path to saved file
        """
        path = output_path or self.output_file
        
        if not self.preference_pairs:
            print("⚠️ No DPO pairs to save.")
            return None
        
        # Create parent directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSONL format
        with open(path, "w", encoding="utf-8") as f:
            for pair in self.preference_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        
        print(f"✅ Saved {len(self.preference_pairs)} DPO pairs to {path}")
        return path

    def get_statistics(self):
        """Get statistics about created and rejected pairs"""
        return {
            "total_pairs": len(self.preference_pairs),
            "rejected_pairs": len(self.rejected_pairs),
            "acceptance_rate": (
                len(self.preference_pairs) / (len(self.preference_pairs) + len(self.rejected_pairs))
                if (len(self.preference_pairs) + len(self.rejected_pairs)) > 0
                else 0.0
            ),
            "rejection_reasons": self._get_rejection_breakdown()
        }

    def _get_rejection_breakdown(self):
        """Get breakdown of rejection reasons"""
        reasons = {}
        for rejected in self.rejected_pairs:
            reason = rejected["reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
        return reasons

    def clear_pairs(self):
        """Clear all stored pairs (useful for batch processing)"""
        cleared_count = len(self.preference_pairs)
        self.preference_pairs = []
        self.rejected_pairs = []
        return cleared_count

    def export_statistics(self, output_path="dpo_statistics.json"):
        """Export detailed statistics to JSON file"""
        stats = self.get_statistics()
        stats["pairs_sample"] = self.preference_pairs[:5]  # Include 5 samples
        stats["rejected_sample"] = self.rejected_pairs[:5]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved DPO statistics to {output_path}")
        return output_path


# ============================================================================
# DOCUMENTATION
# ============================================================================
"""
DPO BUILDER - UPDATED

PURPOSE:
--------
Creates high-quality training data for Direct Preference Optimization (DPO).
Only accepts pairs that pass strict quality and confidence checks.

QUALITY GATES:
-------------
1. Quality check must pass (correction contains verified fact)
2. Verification confidence ≥ 0.70 (LOWERED from 0.80)
3. No verification errors
4. Valid non-empty responses
5. Responses must be different
6. Correction must contain fact content (40% overlap)

THRESHOLDS:
----------
- Minimum confidence: 0.70 (was 0.80)
- Fact content overlap: 40%
- Acceptance rate target: 60-80%

OUTPUT FORMAT:
-------------
JSONL file with format:
{
    "prompt": "user query",
    "chosen": "corrected response",
    "rejected": "hallucinated response",
    "metadata": {
        "verified_fact": "...",
        "error_type": "...",
        "verification_confidence": 0.75,
        "created_at": "2024-...",
        "quality_score": 1.0
    }
}

USAGE:
-----
builder = DPOBuilder()
builder.create_preference_pair(...)
builder.save_to_file("training_data.jsonl")
stats = builder.get_statistics()
"""