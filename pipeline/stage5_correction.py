"""
Stage 5: Hallucination Correction Pipeline - INTEGRATED VERSION
Connects with Stage 4 (Arrij) output
Owner: Rimsha Azam (22i-1129)
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Stage 4 verification
try:
    from pipeline.stage4_verification import verify_fact
    STAGE4_AVAILABLE = True
except ImportError:
    print("⚠️  Stage 4 not found. Running in standalone mode.")
    STAGE4_AVAILABLE = False

# Import utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from error_description import ErrorClassifier
    from dpo_builder import DPOBuilder
except ImportError:
    print("Error: Cannot import utilities. Make sure error_description.py and dpo_builder.py are in utils/ folder")
    sys.exit(1)

# Import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    print("⚠️  Groq not installed. Model-based correction will be disabled.")
    print("   Install with: pip install groq")
    GROQ_AVAILABLE = False


class HallucinationCorrector:
    """Main correction pipeline integrated with Stage 4"""
    
    def __init__(self, correction_method="template", groq_api_key=None):
        """
        Initialize corrector
        
        Args:
            correction_method: 'template' (rule-based) or 'model' (LLM-based)
            groq_api_key: Groq API key for model-based correction
        """
        self.method = correction_method
        self.error_classifier = ErrorClassifier()
        self.dpo_builder = DPOBuilder()
        self.corrections = []
        
        # Initialize Groq client if using model-based correction
        self.groq_client = None
        if self.method == "model" and GROQ_AVAILABLE and groq_api_key:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
                print("✅ Groq API initialized successfully!")
            except Exception as e:
                print(f"⚠️  Groq initialization failed: {e}")
                print("   Falling back to template-based correction")
                self.method = "template"
        elif self.method == "model" and not groq_api_key:
            print("⚠️  No API key provided. Using template-based correction.")
            self.method = "template"
    
    def process_stage2_output(self, stage2_data: Dict, retrieved_docs: List[Dict] = None) -> Dict:
        """
        Process Stage 2 output through Stage 4 and Stage 5
        
        Args:
            stage2_data: Output from Stage 2 (detection)
            retrieved_docs: Retrieved documents from Stage 3 (optional)
            
        Returns:
            Complete correction output
        """
        original_text = stage2_data.get("original_text")
        normalized_text = stage2_data.get("normalized_text")
        hallucination_detected = stage2_data.get("hallucination_detected", False)
        entropy_score = stage2_data.get("entropy_score", 0)
        responses = stage2_data.get("responses", [])
        
        # If no hallucination detected, return original
        if not hallucination_detected:
            return {
                "original_query": original_text,
                "hallucination_detected": False,
                "corrected_text": responses[0] if responses else original_text,
                "correction_method": "none",
                "needs_correction": False
            }
        
        # Get hallucinated response (first response from Stage 2)
        hallucinated_response = responses[0] if responses else ""
        
        # Stage 4: Verify facts
        if STAGE4_AVAILABLE and retrieved_docs:
            verification_result = verify_fact(original_text, retrieved_docs)
        else:
            # Fallback if Stage 4 not available
            verification_result = {
                "verified_fact": "No verification available",
                "confidence": 0.5,
                "sources": [],
                "correction_candidates": []
            }
        
        # Prepare input for Stage 5
        stage5_input = {
            "query_id": f"q_{hash(original_text) % 10000}",
            "original_query": original_text,
            "normalized_query": normalized_text,
            "hallucinated_response": hallucinated_response,
            "hallucination_detected": hallucination_detected,
            "entropy_score": entropy_score,
            "verified_fact": verification_result["verified_fact"],
            "verification_confidence": verification_result["confidence"],
            "correction_candidates": verification_result["correction_candidates"],
            "sources": verification_result["sources"],
            "hallucination_type": self._detect_error_type(hallucinated_response),
            "error_span": self._identify_error_span(hallucinated_response)
        }
        
        # Stage 5: Correct hallucination
        corrected_output = self.correct_hallucination(stage5_input)
        
        return corrected_output
    
    def _detect_error_type(self, text: str) -> str:
        """Auto-detect error type from hallucinated text"""
        error_info = self.error_classifier.classify_error(text, None)
        return error_info["type"]
    
    def _identify_error_span(self, text: str) -> Dict:
        """Identify error span in text (simplified)"""
        return {
            "start": 0,
            "end": len(text),
            "text": text
        }
    
    def correct_hallucination(self, input_data: Dict) -> Dict:
        """
        Main correction function
        
        Args:
            input_data: Combined data from Stage 2, 3, 4
            
        Returns:
            dict: Corrected output
        """
        query_id = input_data.get("query_id")
        original_query = input_data.get("original_query")
        hallucinated_response = input_data.get("hallucinated_response")
        verified_fact = input_data.get("verified_fact")
        error_span = input_data.get("error_span", {})
        error_type = input_data.get("hallucination_type")
        correction_candidates = input_data.get("correction_candidates", [])
        
        # Step 1: Classify error
        error_info = self.error_classifier.classify_error(
            error_span.get("text", ""), 
            error_type
        )
        
        # Step 2: Generate correction
        if self.method == "template":
            corrected_text = self._template_based_correction(
                hallucinated_response,
                error_span,
                correction_candidates,
                verified_fact
            )
        else:
            corrected_text = self._model_based_correction(
                original_query,
                hallucinated_response,
                verified_fact,
                error_info["type"]
            )
        
        # Step 3: Generate explanation
        explanation = self.error_classifier.generate_explanation(
            hallucinated_response,
            corrected_text,
            error_info["type"]
        )
        
        # Step 4: Quality check
        quality_passed = self._quality_check(corrected_text, verified_fact)
        
        # Step 5: Create DPO pair
        self.dpo_builder.create_preference_pair(
            query=original_query,
            hallucinated_response=hallucinated_response,
            corrected_response=corrected_text,
            verified_fact=verified_fact,
            error_type=error_info["type"]
        )
        
        # Prepare output
        output = {
            "query_id": query_id,
            "original_query": original_query,
            "normalized_query": input_data.get("normalized_query", ""),
            "hallucinated_response": hallucinated_response,
            "corrected_text": corrected_text,
            "hallucination_type": error_info["type"],
            "correction_explanation": explanation,
            "confidence_score": error_info["confidence"],
            "verification_confidence": input_data.get("verification_confidence", 0),
            "correction_method": self.method,
            "quality_checks_passed": quality_passed,
            "verified_fact": verified_fact,
            "sources": input_data.get("sources", []),
            "entropy_score": input_data.get("entropy_score", 0)
        }
        
        self.corrections.append(output)
        return output
    
    def _template_based_correction(self, hallucinated_text, error_span, 
                                   candidates, verified_fact):
        """Template-based correction using verified facts"""
        # Use first correction candidate if available
        if candidates and len(candidates) > 0:
            return candidates[0]
        
        # Fallback: Use verified fact directly
        return verified_fact
    
    def _model_based_correction(self, query, hallucinated_text, verified_fact, error_type):
        """LLM-based correction using Groq API"""
        if not self.groq_client:
            return verified_fact
        
        # Build prompt for Groq
        prompt = self._build_correction_prompt(
            query, hallucinated_text, verified_fact, error_type
        )
        
        try:
            # Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a hallucination correction system for Urdu-English code-mixed text. Fix factual errors while preserving the language mixing style and naturalness."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=150,
                top_p=0.9
            )
            
            corrected = chat_completion.choices[0].message.content.strip()
            corrected = corrected.replace('"', '').replace("'", "").strip()
            
            return corrected
            
        except Exception as e:
            print(f"⚠️  Groq API error: {e}")
            return verified_fact
    
    def _build_correction_prompt(self, query, hallucinated_text, verified_fact, error_type):
        """Build effective prompt for Groq API"""
        
        prompt = f"""Fix the hallucination in this response while keeping the same language style (Urdu-English code-mixed).

Original Question: {query}

Hallucinated Response (WRONG): {hallucinated_text}

Verified Fact: {verified_fact}

Error Type: {error_type}

Instructions:
1. Rewrite the response to be factually correct using the verified fact
2. Keep the same language mixing style (Urdu-English code-mixed)
3. Keep it natural and conversational
4. Only output the corrected response, nothing else

Corrected Response:"""
        
        return prompt
    
    def _quality_check(self, corrected_text, verified_fact):
        """Basic quality validation"""
        if not corrected_text or not verified_fact:
            return False
        
        corrected_words = set(corrected_text.lower().split())
        fact_words = set(verified_fact.lower().split())
        
        overlap = len(corrected_words & fact_words)
        return overlap > 0
    
    def process_batch_from_stage2(self, stage2_file, output_file, retrieved_docs_map=None):
        """
        Process Stage 2 output file through complete pipeline
        
        Args:
            stage2_file: Path to Stage 2 output JSON
            output_file: Path to save Stage 5 output
            retrieved_docs_map: Optional dict mapping queries to retrieved docs
        """
        print(f"📖 Reading Stage 2 output from: {stage2_file}")
        
        with open(stage2_file, 'r', encoding='utf-8') as f:
            stage2_data = json.load(f)
        
        print(f"🔧 Processing {len(stage2_data)} queries...")
        print(f"   Method: {self.method.upper()}")
        print()
        
        results = []
        for idx, item in enumerate(stage2_data):
            try:
                # Get retrieved docs if available
                query = item.get("original_text")
                retrieved_docs = retrieved_docs_map.get(query) if retrieved_docs_map else None
                
                # Process through pipeline
                result = self.process_stage2_output(item, retrieved_docs)
                results.append(result)
                
                status = "🔄" if not item.get("hallucination_detected") else "✅"
                print(f"{status} Processed: {idx+1}/{len(stage2_data)} - {query[:50]}...")
                
            except Exception as e:
                print(f"❌ Error processing query {idx+1}: {e}")
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved {len(results)} results to: {output_file}")
        
        # Save DPO training data
        dpo_file = output_file.replace(".json", "_dpo.jsonl")
        self.dpo_builder.save_to_file(dpo_file)
        
        # Print statistics
        self._print_statistics(results)
        
        return results
    
    def _print_statistics(self, results):
        """Print correction statistics"""
        print("\n" + "="*50)
        print("📊 CORRECTION STATISTICS")
        print("="*50)
        
        total = len(results)
        
        if total == 0:
            print("No corrections processed")
            print("="*50)
            return
        
        hallucinations = sum(1 for r in results if r.get("hallucination_detected", True))
        passed_quality = sum(1 for r in results if r.get("quality_checks_passed", False))
        
        error_types = {}
        for r in results:
            if r.get("hallucination_type"):
                et = r["hallucination_type"]
                error_types[et] = error_types.get(et, 0) + 1
        
        print(f"Total queries: {total}")
        print(f"Hallucinations detected: {hallucinations}")
        print(f"Corrections made: {hallucinations}")
        print(f"Quality checks passed: {passed_quality}/{hallucinations} ({passed_quality/hallucinations*100 if hallucinations > 0 else 0:.1f}%)")
        print(f"Correction method: {self.method.upper()}")
        
        if error_types:
            print(f"\nError type distribution:")
            for error_type, count in error_types.items():
                print(f"  - {error_type}: {count}")
        
        dpo_stats = self.dpo_builder.get_statistics()
        print(f"\nDPO training pairs created: {dpo_stats['total_pairs']}")
        print("="*50)


def main():
    """Main entry point"""
    print("="*60)
    print("🔧 STAGE 5: HALLUCINATION CORRECTION (INTEGRATED)")
    print("="*60)
    print()
    
    # Check for API key
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    if not groq_api_key:
        print("⚠️  GROQ_API_KEY not found in environment")
        print("   Set it with: set GROQ_API_KEY=your_key_here  (Windows)")
        print()
        print("🔧 Running in TEMPLATE mode")
        correction_method = "template"
    else:
        print("✅ GROQ_API_KEY found!")
        print()
        choice = input("Choose correction method:\n  1) Template-based\n  2) Model-based (Groq)\n\nEnter (1 or 2): ").strip()
        
        if choice == "2":
            correction_method = "model"
            print("\n🤖 Using MODEL-based correction")
        else:
            correction_method = "template"
            print("\n📋 Using TEMPLATE-based correction")
    
    print()
    
    # File paths
    stage2_file = "data/stage2_output.json"
    output_file = "data/stage5_output.json"
    
    if not os.path.exists(stage2_file):
        print(f"❌ Error: {stage2_file} not found!")
        print("   Make sure Stage 2 has run successfully")
        return
    
    # Initialize corrector
    corrector = HallucinationCorrector(
        correction_method=correction_method,
        groq_api_key=groq_api_key
    )
    
    # Process corrections
    results = corrector.process_batch_from_stage2(stage2_file, output_file)
    
    print(f"\n✅ Stage 5 Complete! Check {output_file} for results.")


if __name__ == "__main__":
    main()