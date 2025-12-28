# stage5_correction.py - IMPROVED

"""
Stage 5: Hallucination Correction Pipeline
IMPROVED: Better Groq prompting for concise, natural responses
"""

import os
import re

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from pipeline.utils.error_description import ErrorClassifier
from pipeline.utils.dpo_builder import DPOBuilder


def validate_groq_key(client):
    """Validate Groq API key"""
    try:
        client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="llama-3.3-70b-versatile",
            max_tokens=5
        )
        return True
    except Exception as e:
        return False


class HallucinationCorrector:
    """Corrects hallucinated responses using verified facts"""

    def __init__(self, correction_method="template"):
        self.method = correction_method
        self.error_classifier = ErrorClassifier()
        self.dpo_builder = DPOBuilder()
        self.corrections = []
        self.groq_client = None

        if self.method == "model" and GROQ_AVAILABLE:
            self._initialize_groq()
        else:
            self.method = "template"

    def _initialize_groq(self):
        """Initialize Groq API client"""
        api_key = os.environ.get("GROQ_API_KEY")
        
        if not api_key:
            self.method = "template"
            return
        
        try:
            self.groq_client = Groq(api_key=api_key)
            if validate_groq_key(self.groq_client):
                print("✅ Groq API initialized successfully.")
            else:
                self.method = "template"
                self.groq_client = None
        except Exception as e:
            self.method = "template"
            self.groq_client = None

    def _abstain(self, input_data, reason):
        """Return abstention response"""
        verified_fact = input_data.get("verified_fact")
        
        if verified_fact and verified_fact != "No verified fact available":
            corrected_text = verified_fact
        else:
            corrected_text = input_data.get("hallucinated_response", "Unable to verify answer")
        
        return {
            "query_id": input_data.get("query_id"),
            "original_query": input_data.get("original_query"),
            "hallucinated_response": input_data.get("hallucinated_response"),
            "corrected_text": corrected_text,
            "hallucination_type": "verification_error",
            "correction_method": "abstained",
            "correction_explanation": reason,
            "reason": reason,
            "quality_checks_passed": False,
            "verified_fact": verified_fact,
            "verification_confidence": input_data.get("verification_confidence", 0.0)
        }

    def _detect_language_mix(self, query):
        """Detect if query uses Urdu-English code-mixing"""
        urdu_words = ['ka', 'ki', 'ke', 'ko', 'se', 'mein', 'par', 
                     'kon', 'kaun', 'kya', 'kab', 'kahan', 'kyun', 'kaise',
                     'hai', 'hain', 'tha', 'the', 'thy', 'kitne', 'konsa']
        
        query_lower = query.lower()
        return any(word in query_lower.split() for word in urdu_words)

    def _template_based_correction(self, query, verified_fact):
        """Generate correction using templates"""
        fact_clean = verified_fact.strip()
        
        # If fact ends with period, return as-is
        if fact_clean.endswith('.'):
            return fact_clean
        
        # Otherwise add period
        return f"{fact_clean}."

    def _quality_check(self, corrected, verified_fact):
        """Check if correction contains verified fact content"""
        if not corrected or not verified_fact:
            return False
        
        corrected_lower = corrected.lower()
        fact_words = set(verified_fact.lower().split())
        
        # Remove common words
        stop_words = {'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but',
                     'hai', 'hain', 'ka', 'ki', 'ke'}
        fact_words -= stop_words
        
        # At least 50% of fact words should be in correction
        matches = sum(1 for word in fact_words if word in corrected_lower)
        return matches >= len(fact_words) * 0.5 if fact_words else True

    def _model_output_guard(self, output, verified_fact):
        """Verify model output contains verified fact"""
        output_words = set(output.lower().split())
        fact_words = set(verified_fact.lower().split())
        
        overlap = len(output_words & fact_words)
        required = max(2, len(fact_words) // 2)
        
        return overlap >= required

    def _model_based_correction(self, query, hallucinated, verified_fact):
        """
        Use Groq API to generate correction.
        IMPROVED: Better prompt for concise, natural responses.
        """
        if not self.groq_client:
            return None
        
        has_urdu = self._detect_language_mix(query)
        
        # IMPROVED PROMPT for concise responses
        if has_urdu:
            prompt = f"""Question: {query}
Incorrect Answer: {hallucinated}
Verified Fact: {verified_fact}

Task: Provide a SHORT, natural answer in Urdu-English mixed style that corrects the error using the verified fact.
Requirements:
- Maximum 2 sentences
- Use simple, conversational language
- Mix Urdu and English naturally (like "Pakistan ke 4 provinces hain")
- DO NOT add unnecessary explanation or context
- DO NOT say things like "na ki" or "bilkul galat hai"
- Just state the correct fact naturally

Answer:"""
        else:
            prompt = f"""Question: {query}
Incorrect Answer: {hallucinated}
Verified Fact: {verified_fact}

Task: Provide a SHORT, natural answer that corrects the error.
Requirements:
- Maximum 2 sentences
- Use simple, clear language
- DO NOT add unnecessary explanation
- Just state the correct fact

Answer:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Lower temperature for more focused responses
                max_tokens=100,    # Reduced from 150 to encourage brevity
                top_p=0.9
            )

            corrected = response.choices[0].message.content.strip()
            
            # Clean up common prefixes
            prefixes_to_remove = [
                r'^(Answer:|Corrected Answer:|Response:|Here is the corrected answer:)\s*',
                r'^(Yeh|Toh|Actually|In fact)\s*',
            ]
            
            for prefix_pattern in prefixes_to_remove:
                corrected = re.sub(prefix_pattern, '', corrected, flags=re.IGNORECASE)
            
            corrected = corrected.strip()
            
            # Verify output contains the verified fact
            if not self._model_output_guard(corrected, verified_fact):
                return None

            return corrected

        except Exception as e:
            return None

    def correct_hallucination(self, input_data):
        """Main entry point for correction"""
        
        verified_fact = input_data.get("verified_fact")
        verification_confidence = input_data.get("verification_confidence", 0.0)
        hallucinated = input_data.get("hallucinated_response", "")
        original_query = input_data.get("original_query", "")
        
        # Confidence check (threshold 0.40)
        if not verified_fact or verified_fact == "No verified fact available":
            return self._abstain(input_data, "No verified fact available")
        
        if verification_confidence < 0.40:
            return self._abstain(input_data, f"Low verification confidence: {verification_confidence:.2f}")
        
        # Classify error
        error_info = self.error_classifier.classify_error(
            hallucinated,
            input_data.get("hallucination_type", "factual_error")
        )
        
        # Generate correction
        corrected = None
        
        if self.method == "model":
            corrected = self._model_based_correction(original_query, hallucinated, verified_fact)
        
        # Fallback to template
        if corrected is None:
            corrected = self._template_based_correction(original_query, verified_fact)
        
        # Quality check
        quality_passed = self._quality_check(corrected, verified_fact)
        
        # Explanation
        explanation = self.error_classifier.generate_explanation(
            hallucinated, corrected, error_info["type"]
        )
        
        # DPO pair (only if high quality)
        if quality_passed and verification_confidence >= 0.70:
            self.dpo_builder.create_preference_pair(
                query=original_query,
                hallucinated_response=hallucinated,
                corrected_response=corrected,
                verified_fact=verified_fact,
                error_type=error_info["type"],
                verification_confidence=verification_confidence,
                quality_passed=quality_passed
            )
        
        return {
            "query_id": input_data.get("query_id"),
            "original_query": original_query,
            "hallucinated_response": hallucinated,
            "corrected_text": corrected,
            "hallucination_type": error_info["type"],
            "correction_explanation": explanation,
            "verification_confidence": verification_confidence,
            "quality_checks_passed": quality_passed,
            "correction_method": self.method,
            "verified_fact": verified_fact
        }

    def save_dpo_data(self, output_path=None):
        """Save DPO training data"""
        return self.dpo_builder.save_to_file(output_path)

    def get_statistics(self):
        """Get statistics"""
        return {
            "total_corrections": len(self.corrections),
            "dpo_pairs": self.dpo_builder.get_statistics()["total_pairs"],
            "correction_method": self.method
        }