"""
Stage 5: Hallucination Correction Pipeline (FIXED - Threshold 0.40)
Owner: Rimsha Azam

CRITICAL FIX: Lowered threshold from 0.50 to 0.40 to accept confidence like 0.495
"""

import os
import re

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq library not installed. Using template mode only.")

# Relative imports
from .utils.error_description import ErrorClassifier
from .utils.dpo_builder import DPOBuilder


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
        print(f"❌ Groq API validation failed: {e}")
        return False


class HallucinationCorrector:
    """Corrects hallucinated responses using verified facts"""

    def __init__(self, correction_method="template"):
        self.method = correction_method
        self.error_classifier = ErrorClassifier()
        self.dpo_builder = DPOBuilder()
        self.corrections = []
        self.groq_client = None

        # Try to initialize Groq if method is "model"
        if self.method == "model" and GROQ_AVAILABLE:
            self._initialize_groq()
        else:
            if self.method == "model" and not GROQ_AVAILABLE:
                print("⚠️ Groq not available. Falling back to TEMPLATE mode.")
            self.method = "template"
            print(f"[Stage 5] Using correction method: {self.method}")

    def _initialize_groq(self):
        """Initialize Groq API client"""
        api_key = os.environ.get("GROQ_API_KEY")
        
        if not api_key:
            print("⚠️ GROQ_API_KEY not found. Falling back to TEMPLATE mode.")
            self.method = "template"
            return
        
        try:
            self.groq_client = Groq(api_key=api_key)
            if validate_groq_key(self.groq_client):
                print("✅ Groq API initialized successfully.")
            else:
                print("⚠️ Groq API validation failed. Falling back to TEMPLATE mode.")
                self.method = "template"
                self.groq_client = None
        except Exception as e:
            print(f"⚠️ Groq initialization failed: {e}. Falling back to TEMPLATE mode.")
            self.method = "template"
            self.groq_client = None

    def _abstain(self, input_data, reason):
        """
        Return abstention response
        FIXED: Return verified_fact if available, not hallucinated response
        """
        # Try to return verified fact even when abstaining
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
            "reason": reason,
            "quality_checks_passed": False,
            "verified_fact": verified_fact
        }

    def _detect_language_mix(self, query):
        """Detect if query uses Urdu-English code-mixing"""
        urdu_words = ['ka', 'ki', 'ke', 'ko', 'se', 'mein', 'par', 
                     'kon', 'kaun', 'kya', 'kab', 'kahan', 'kyun', 'kaise',
                     'hai', 'hain', 'tha', 'the', 'thy', 'kitne', 'konsa']
        
        query_lower = query.lower()
        return any(word in query_lower.split() for word in urdu_words)

    def _is_complete_sentence(self, text):
        """Check if text is already a complete sentence"""
        text = text.strip()
        
        # Has proper ending punctuation
        if text.endswith(('.', '!', '?', '।')):
            return True
        
        # Has a verb (is, was, are, were, has, have)
        if any(verb in text.lower().split() for verb in ['is', 'was', 'are', 'were', 'has', 'have', 'had']):
            return True
        
        return False

    def _template_based_correction(self, query, verified_fact):
        """
        Generate correction using templates.
        FIXED: Smart detection of when to add "hai"
        """
        has_urdu = self._detect_language_mix(query)
        fact_clean = verified_fact.strip()
        
        # CRITICAL FIX: If fact is already a complete English sentence, return as-is
        if self._is_complete_sentence(fact_clean):
            # Check if it already has "hai" at the end
            if fact_clean.endswith(' hai.') or fact_clean.endswith(' hai'):
                return fact_clean
            
            # If query has Urdu BUT fact is a long complete English sentence, DON'T add hai
            if has_urdu and len(fact_clean.split()) > 8:
                # Long English sentences look weird with "hai"
                return fact_clean if fact_clean.endswith('.') else f"{fact_clean}."
            
            # Short fact + Urdu query = can add hai
            if has_urdu and len(fact_clean.split()) <= 8:
                # Check if fact is pure English (no Urdu words)
                fact_lower = fact_clean.lower()
                has_english_verb = any(verb in fact_lower.split() for verb in ['is', 'was', 'are', 'were'])
                
                if has_english_verb:
                    # "Muhammad Iqbal is the national poet" → keep as-is (already has verb)
                    return fact_clean if fact_clean.endswith('.') else f"{fact_clean}."
            
            # Default: return with period
            return fact_clean if fact_clean.endswith('.') else f"{fact_clean}."
        
        # Incomplete sentence - add appropriate ending
        if has_urdu:
            return f"{fact_clean} hai."
        else:
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
        """Use Groq API to generate correction"""
        if not self.groq_client:
            return None
        
        has_urdu = self._detect_language_mix(query)
        
        prompt = f"""Fix the incorrect answer using the verified fact.

Question: {query}
Incorrect Answer: {hallucinated}
Verified Fact: {verified_fact}

Instructions:
1. Use the verified fact to correct the answer
2. Keep the response natural and conversational
{"3. Maintain Urdu-English code-mixed style" if has_urdu else "3. Use clear English"}
4. Output ONLY the corrected answer

Corrected Answer:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )

            corrected = response.choices[0].message.content.strip()
            corrected = re.sub(r'^(Answer:|Corrected Answer:|Response:)\s*', '', corrected, flags=re.IGNORECASE)
            corrected = corrected.strip()
            
            if not self._model_output_guard(corrected, verified_fact):
                print("⚠️ Model output failed verification, using template.")
                return None

            return corrected

        except Exception as e:
            print(f"⚠️ Groq API error: {e}")
            return None

    def correct_hallucination(self, input_data):
        """Main entry point for correction"""
        
        verified_fact = input_data.get("verified_fact")
        verification_confidence = input_data.get("verification_confidence", 0.0)
        hallucinated = input_data.get("hallucinated_response", "")
        original_query = input_data.get("original_query", "")
        
        # Confidence check (LOWERED to 0.40 from 0.50)
        # This accepts confidence like 0.495 which was being rejected
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