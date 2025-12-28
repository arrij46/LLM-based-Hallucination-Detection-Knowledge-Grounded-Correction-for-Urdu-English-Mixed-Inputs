"""
Error Classification Module (UPDATED)
Owner: Rimsha Azam

UPDATES:
- Better error pattern matching
- More error types supported
- Improved confidence scoring
"""

import re


class ErrorClassifier:
    """Classifies hallucination types and generates explanations"""

    ERROR_TYPES = {
        "factual_error": "Incorrect factual information",
        "date_error": "Wrong date or time information",
        "numerical_error": "Incorrect numerical values",
        "extrinsic_addition": "Added information not present in source",
        "intrinsic_contradiction": "Self-contradictory statements",
        "entity_error": "Wrong entity or name",
        "verification_error": "Correction blocked due to unreliable verification",
        "location_error": "Wrong location or place information",
        "identity_error": "Wrong person or organization identity"
    }

    def __init__(self):
        self.error_patterns = self._load_error_patterns()

    def _load_error_patterns(self):
        """Load patterns for automatic error type detection"""
        return {
            "date_error": [
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
                r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
                r'\b(19|20)\d{2}\b',  # Years
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
            ],
            "numerical_error": [
                r'\b\d+\s*(million|billion|thousand|crore|lakh)\b',
                r'\b\d+\s*percent\b',
                r'\b\d+%\b',
                r'\b\d+\.\d+\b',  # Decimals
            ],
            "location_error": [
                r'\b(in|at|near|from)\s+[A-Z][a-z]+\b',
                r'\b(city|country|province|state|capital)\b',
            ],
            "identity_error": [
                r'\b(president|minister|governor|founder|leader|poet)\b',
                r'\b(is|was|are|were)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Names
            ]
        }

    def classify_error(self, error_text, error_type=None):
        """
        Classify the type of error in hallucinated text.
        
        Args:
            error_text: The hallucinated text
            error_type: Optional pre-assigned error type
        
        Returns:
            Dictionary with type, description, and confidence
        """
        # If type is already provided and valid, use it
        if error_type and error_type in self.ERROR_TYPES:
            return {
                "type": error_type,
                "description": self.ERROR_TYPES[error_type],
                "confidence": 1.0
            }
        
        # Auto-detect error type
        text_lower = error_text.lower()
        
        # Check each pattern type
        for pattern_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return {
                        "type": pattern_type,
                        "description": self.ERROR_TYPES[pattern_type],
                        "confidence": 0.85
                    }
        
        # Default to factual_error
        return {
            "type": "factual_error",
            "description": self.ERROR_TYPES["factual_error"],
            "confidence": 0.70
        }

    def generate_explanation(self, hallucinated, corrected, error_type):
        """
        Generate human-readable explanation of the correction.
        
        Args:
            hallucinated: Original incorrect text
            corrected: Corrected text
            error_type: Type of error
        
        Returns:
            Explanation string
        """
        if error_type == "verification_error":
            return "Correction abstained due to unreliable or conflicting retrieved facts."
        
        # Truncate long texts for readability
        hall_short = hallucinated[:50] + "..." if len(hallucinated) > 50 else hallucinated
        corr_short = corrected[:50] + "..." if len(corrected) > 50 else corrected
        
        error_desc = self.ERROR_TYPES.get(error_type, "error")
        
        return f"Corrected {error_desc}: '{hall_short}' → '{corr_short}'"

    def get_all_error_types(self):
        """Return list of all supported error types"""
        return list(self.ERROR_TYPES.keys())

    def get_error_description(self, error_type):
        """Get description for a specific error type"""
        return self.ERROR_TYPES.get(error_type, "Unknown error type")