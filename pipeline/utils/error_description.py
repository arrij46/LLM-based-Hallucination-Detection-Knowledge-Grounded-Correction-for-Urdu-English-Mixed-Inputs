"""
Error Classification Module
Classifies hallucination types and generates descriptions
Owner: Rimsha Azam
"""

class ErrorClassifier:
    """Classifies different types of hallucination errors"""
    
    ERROR_TYPES = {
        "factual_error": "Incorrect factual information",
        "date_error": "Wrong date or time information",
        "numerical_error": "Incorrect numerical values",
        "extrinsic_addition": "Added information not present in source",
        "intrinsic_contradiction": "Self-contradictory statements",
        "entity_error": "Wrong entity or name"
    }
    
    def __init__(self):
        self.error_patterns = self._load_error_patterns()
    
    def _load_error_patterns(self):
        """Load patterns for each error type"""
        return {
            "date_error": ["january", "february", "march", "april", "may", "june",
                          "july", "august", "september", "october", "november", "december",
                          "2020", "2021", "2022", "2023", "2024"],
            "numerical_error": ["million", "billion", "thousand", "percent", "%"],
        }
    
    def classify_error(self, error_text, error_type=None):
        """
        Classify the type of error
        
        Args:
            error_text: The hallucinated text span
            error_type: Pre-labeled error type (if available)
            
        Returns:
            dict: Classification results
        """
        if error_type and error_type in self.ERROR_TYPES:
            return {
                "type": error_type,
                "description": self.ERROR_TYPES[error_type],
                "confidence": 1.0
            }
        
        # Auto-detect error type
        error_lower = error_text.lower()
        
        # Check for dates
        for pattern in self.error_patterns["date_error"]:
            if pattern in error_lower:
                return {
                    "type": "date_error",
                    "description": self.ERROR_TYPES["date_error"],
                    "confidence": 0.9
                }
        
        # Check for numbers
        for pattern in self.error_patterns["numerical_error"]:
            if pattern in error_lower:
                return {
                    "type": "numerical_error",
                    "description": self.ERROR_TYPES["numerical_error"],
                    "confidence": 0.85
                }
        
        # Default to factual error
        return {
            "type": "factual_error",
            "description": self.ERROR_TYPES["factual_error"],
            "confidence": 0.7
        }
    
    def generate_explanation(self, hallucinated_text, corrected_text, error_type):
        """
        Generate human-readable explanation of the correction
        
        Args:
            hallucinated_text: Original wrong text
            corrected_text: Corrected version
            error_type: Type of error
            
        Returns:
            str: Explanation
        """
        explanations = {
            "factual_error": f"Corrected factual error: '{hallucinated_text}' → '{corrected_text}'",
            "date_error": f"Fixed incorrect date: '{hallucinated_text}' → '{corrected_text}'",
            "numerical_error": f"Corrected numerical value: '{hallucinated_text}' → '{corrected_text}'",
            "extrinsic_addition": f"Removed extraneous information: '{hallucinated_text}'",
            "entity_error": f"Fixed entity name: '{hallucinated_text}' → '{corrected_text}'"
        }
        
        return explanations.get(error_type, f"Corrected: '{hallucinated_text}' → '{corrected_text}'")


if __name__ == "__main__":
    # Test the classifier
    classifier = ErrorClassifier()
    
    # Test cases
    test_cases = [
        ("March 15, 2022", "date_error"),
        ("15 million", None),
        ("Karachi", "factual_error")
    ]
    
    print("=== Error Classification Tests ===\n")
    for text, error_type in test_cases:
        result = classifier.classify_error(text, error_type)
        print(f"Text: '{text}'")
        print(f"Classification: {result}")
        print()