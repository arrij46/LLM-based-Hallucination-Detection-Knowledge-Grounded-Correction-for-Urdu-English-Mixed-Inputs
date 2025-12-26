"""
DPO (Direct Preference Optimization) Training Data Builder
Creates preference pairs for model fine-tuning
Owner: Rimsha Azam
"""

import json
from datetime import datetime

class DPOBuilder:
    """Builds preference pairs for DPO training"""
    
    def __init__(self, output_file="dpo_training_data.jsonl"):
        self.output_file = output_file
        self.preference_pairs = []
    
    def create_preference_pair(self, query, hallucinated_response, corrected_response, 
                               verified_fact, error_type):
        """
        Create a preference pair for DPO training
        
        Args:
            query: Original user query
            hallucinated_response: Wrong response (rejected)
            corrected_response: Correct response (chosen)
            verified_fact: Ground truth fact
            error_type: Type of hallucination
            
        Returns:
            dict: DPO training pair
        """
        pair = {
            "prompt": query,
            "chosen": corrected_response,  # Better response
            "rejected": hallucinated_response,  # Worse response
            "metadata": {
                "verified_fact": verified_fact,
                "error_type": error_type,
                "created_at": datetime.now().isoformat()
            }
        }
        
        self.preference_pairs.append(pair)
        return pair
    
    def save_to_file(self, output_path=None):
        """Save all preference pairs to JSONL file"""
        path = output_path or self.output_file
        
        with open(path, 'w', encoding='utf-8') as f:
            for pair in self.preference_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved {len(self.preference_pairs)} preference pairs to {path}")
        return path
    
    def get_statistics(self):
        """Get statistics about collected pairs"""
        if not self.preference_pairs:
            return {"total_pairs": 0}
        
        error_types = {}
        for pair in self.preference_pairs:
            error_type = pair["metadata"]["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_pairs": len(self.preference_pairs),
            "error_type_distribution": error_types
        }


if __name__ == "__main__":
    # Test DPO builder
    builder = DPOBuilder("test_dpo_data.jsonl")
    
    # Example pair
    builder.create_preference_pair(
        query="Pakistan ki capital kya hai?",
        hallucinated_response="Pakistan ki capital Karachi hai.",
        corrected_response="Pakistan ki capital Islamabad hai.",
        verified_fact="Islamabad is the capital of Pakistan",
        error_type="factual_error"
    )
    
    stats = builder.get_statistics()
    print("=== DPO Builder Stats ===")
    print(json.dumps(stats, indent=2))