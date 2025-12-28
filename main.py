# main.py

"""
Main pipeline runner for the hallucination detection & correction system.
FINAL VERSION: Clean, professional console output
"""

from typing import Dict, Any
import os
import sys

# ---- Stage imports ----
from pipeline.stage1_preprocessing import preprocess_text
from pipeline.stage2_detection import detect_hallucination
from pipeline.stage3_retrieval import retrieve_facts
from pipeline.stage4_verification import verify_fact
from pipeline.stage5_correction import HallucinationCorrector
from pipeline.stage6_validation import validate_stage5_output


# ============================================================================
# Utility Functions for Clean Output
# ============================================================================

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "═" * 70)
    print(f" {title}")
    print("═" * 70)


def print_stage(stage_num: int, stage_name: str):
    """Print stage header"""
    print(f"\n{'─' * 70}")
    print(f"  Stage {stage_num}: {stage_name}")
    print(f"{'─' * 70}")


def print_metric(label: str, value: Any, indent: int = 2):
    """Print a metric with proper formatting"""
    spaces = " " * indent
    if isinstance(value, float):
        print(f"{spaces}✓ {label}: {value:.3f}")
    elif isinstance(value, bool):
        status = "YES" if value else "NO"
        print(f"{spaces}✓ {label}: {status}")
    else:
        # Truncate long strings
        value_str = str(value)
        if len(value_str) > 70:
            value_str = value_str[:67] + "..."
        print(f"{spaces}✓ {label}: {value_str}")


def print_success(message: str):
    """Print success message"""
    print(f"  ✅ {message}")


def print_warning(message: str):
    """Print warning message"""
    print(f"  ⚠️  {message}")


def print_info(message: str):
    """Print info message"""
    print(f"  ℹ️  {message}")


# ============================================================================
# Data Adapter Functions
# ============================================================================

def adapt_stage3_to_stage4(stage3_output: Dict) -> tuple:
    """
    Convert Stage 3 retrieved_facts to Stage 4 document format.
    Extracts 'sentence' field from each triple.
    """
    retrieved_docs = []
    entity = None
    
    if isinstance(stage3_output, dict):
        entity = stage3_output.get("entity")
        retrieved_facts = stage3_output.get("retrieved_facts", [])
        
        for fact in retrieved_facts:
            if isinstance(fact, dict):
                text = fact.get("sentence", "")
                if text and text.strip():
                    retrieved_docs.append({
                        "text": text,
                        "source": fact.get("category", "Knowledge Base"),
                        "date": "N/A"
                    })
    
    return retrieved_docs, entity


# ============================================================================
# Pipeline Runner
# ============================================================================

def run_pipeline(user_query: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Run the complete 6-stage pipeline with clean output.
    """
    pipeline_trace = {}

    if verbose:
        print_header("PIPELINE EXECUTION")

    # ============================================================================
    # STAGE 1: Preprocessing
    # ============================================================================
    if verbose:
        print_stage(1, "Text Preprocessing")
    
    stage1_output = preprocess_text(user_query)
    pipeline_trace["stage1_preprocessing"] = stage1_output
    
    if verbose:
        print_metric("Normalized", stage1_output.get('normalized_text', '')[:60])
        print_metric("Tokens", len(stage1_output.get('tokens', [])))
        print_metric("Code-switches", len(stage1_output.get('code_switch_points', [])))

    # ============================================================================
    # STAGE 2: Hallucination Detection
    # ============================================================================
    if verbose:
        print_stage(2, "Hallucination Detection")
    
    stage2_results = detect_hallucination([stage1_output])
    detection_result = stage2_results[0]
    pipeline_trace["stage2_detection"] = detection_result

    hallucination_detected = detection_result.get("hallucination_detected", False)
    entropy_score = detection_result.get("entropy_score", 0)
    
    if verbose:
        print_metric("Responses Generated", len(detection_result.get('responses', [])))
        print_metric("Entropy Score", entropy_score)
        print_metric("Hallucination Detected", hallucination_detected)

    # Early exit if no hallucination
    if not hallucination_detected:
        if verbose:
            print_success("No hallucination detected - using model response directly")
        
        final_answer = {
            "text": detection_result["responses"][0],
            "confidence": max(0.0, 1.0 - entropy_score),
            "sources": [],
            "hallucination_detected": False,
            "status": "No correction needed"
        }
        return {
            "final_answer": final_answer,
            "pipeline_trace": pipeline_trace
        }

    # ============================================================================
    # STAGE 3: Knowledge Retrieval
    # ============================================================================
    if verbose:
        print_stage(3, "Knowledge Base Retrieval")
    
    stage3_output = retrieve_facts(query=user_query, top_k=5)
    pipeline_trace["stage3_retrieval"] = stage3_output
    
    if verbose and isinstance(stage3_output, dict):
        print_metric("Entity Detected", stage3_output.get('entity', 'N/A'))
        print_metric("Facts Retrieved", len(stage3_output.get('retrieved_facts', [])))
        print_metric("Retrieval Confidence", stage3_output.get('retrieval_confidence', 0))

    # ============================================================================
    # STAGE 4: Fact Verification
    # ============================================================================
    if verbose:
        print_stage(4, "Fact Verification (CRAG)")
    
    # Adapter: Convert Stage 3 format to Stage 4 format
    retrieved_docs, entity = adapt_stage3_to_stage4(stage3_output)
    
    if verbose:
        print_metric("Documents Prepared", len(retrieved_docs))
    
    verification_result = verify_fact(
        query=user_query,
        retrieved_docs=retrieved_docs,
        entity=entity
    )
    pipeline_trace["stage4_verification"] = verification_result
    
    verified_fact = verification_result.get("verified_fact")
    verification_confidence = verification_result.get("confidence", 0.0)
    
    if verbose:
        if verified_fact:
            print_success(f"Fact verified: {verified_fact[:60]}...")
            print_metric("Verification Confidence", verification_confidence)
            print_metric("Sources", len(verification_result.get("sources", [])))
        else:
            print_warning("No fact could be verified")

    # ============================================================================
    # STAGE 5: Correction Generation
    # ============================================================================
    if verbose:
        print_stage(5, "Hallucination Correction")
    
    # Auto-select correction method
    correction_method = "model" if os.environ.get("GROQ_API_KEY") else "template"
    
    if verbose:
        method_name = "Groq LLM" if correction_method == "model" else "Template-based"
        print_info(f"Using {method_name} correction")
    
    corrector = HallucinationCorrector(correction_method=correction_method)

    stage5_input = {
        "query_id": f"q_{hash(user_query) % 10000}",
        "original_query": user_query,
        "hallucinated_response": detection_result["responses"][0],
        "hallucination_type": detection_result.get("hallucination_type", "factual_error"),
        "verified_fact": verified_fact,
        "verification_confidence": verification_confidence,
        "sources": verification_result.get("sources", []),
        "entropy_score": entropy_score
    }

    stage5_output = corrector.correct_hallucination(stage5_input)
    pipeline_trace["stage5_correction"] = stage5_output
    
    if verbose:
        corrected_text = stage5_output.get("corrected_text", "")
        if corrected_text:
            print_metric("Corrected Text", corrected_text[:60])
        print_metric("Quality Checks", stage5_output.get("quality_checks_passed", False))
        print_metric("Method Used", stage5_output.get("correction_method", "unknown"))

    # ============================================================================
    # STAGE 6: Final Validation
    # ============================================================================
    if verbose:
        print_stage(6, "Language Quality Validation")
    
    # Prepare Stage 6 input with proper field names
    stage6_input = {
        "corrected_text": stage5_output.get("corrected_text", ""),
        "original_text": user_query,
        "hallucination_type": stage5_output.get("hallucination_type", "unknown"),
        "explanation": stage5_output.get("correction_explanation", "")
    }
    
    stage6_results = validate_stage5_output([stage6_input])
    pipeline_trace["stage6_validation"] = stage6_results

    # Extract final results
    final_text = None
    confidence = verification_confidence
    refinement_applied = False
    
    if stage6_results and len(stage6_results) > 0:
        result = stage6_results[0]
        final_text = result.get("final_text")
        
        validation = result.get("validation", {})
        confidence = validation.get("overall_score", verification_confidence)
        
        metadata = result.get("metadata", {})
        refinement_applied = metadata.get("refinement_applied", False)
        
        if verbose:
            print_metric("Validation Score", confidence)
            
            # Show individual validation scores
            if "naturalness" in validation:
                print_metric("  - Naturalness", validation["naturalness"].get("score", 0))
            if "code_mixing_consistency" in validation:
                print_metric("  - Code-mixing", validation["code_mixing_consistency"].get("score", 0))
            if "cultural_appropriateness" in validation:
                print_metric("  - Cultural", validation["cultural_appropriateness"].get("score", 0))
            
            if refinement_applied:
                print_success("Text refined by validator")

    # Fallback chain
    if not final_text or not final_text.strip():
        final_text = (
            stage5_output.get("corrected_text") or
            verified_fact or
            detection_result["responses"][0]
        )

    # ============================================================================
    # Assemble Final Answer
    # ============================================================================
    final_answer = {
        "text": final_text,
        "confidence": confidence,
        "sources": verification_result.get("sources", []),
        "hallucination_detected": True,
        "refinement_applied": refinement_applied,
        "status": "Corrected and validated"
    }

    return {
        "final_answer": final_answer,
        "pipeline_trace": pipeline_trace
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main CLI interface"""
    
    # Print welcome banner
    print("\n" + "═" * 70)
    print("  HALLUCINATION DETECTION & CORRECTION PIPELINE")
    print("  Knowledge-Grounded Correction for Urdu-English Mixed Text")
    print("═" * 70)
    print("\n  Commands:")
    print("    • Type your question to process it")
    print("    • 'quit' or 'exit' to exit")
    print("    • 'help' for more information")
    print("\n" + "═" * 70)

    while True:
        try:
            # Get user input
            user_query = input("\n📝 Enter your query: ").strip()

            # Handle commands
            if user_query.lower() in ["quit", "exit"]:
                print("\n👋 Thank you for using the pipeline. \n<< END >>\n")
                break

            if user_query.lower() == "help":
                print("\n" + "─" * 70)
                print("  HELP")
                print("─" * 70)
                print("\n  This pipeline detects and corrects hallucinations in")
                print("  Urdu-English code-mixed text using a 6-stage process:")
                print("\n  1. Text Preprocessing - Normalize and tokenize")
                print("  2. Hallucination Detection - Entropy-based detection")
                print("  3. Knowledge Retrieval - Search knowledge base")
                print("  4. Fact Verification - Cross-verify facts (CRAG)")
                print("  5. Correction Generation - Generate corrections")
                print("  6. Language Validation - Validate final output")
                print("\n  Example queries:")
                print("    • Pakistan ka capital kya hai?")
                print("    • Pakistan ke kitne provinces hain?")
                print("    • Pakistan ka pehla Governor General kon tha?")
                print("\n" + "─" * 70)
                continue

            if not user_query:
                print("  ⚠️  Please enter a query.")
                continue

            # Run pipeline
            result = run_pipeline(user_query, verbose=True)

            # Print final answer
            print_header("FINAL ANSWER")
            
            if result.get("error"):
                print(f"\n  ❌ Error: {result['error']}\n")
            else:
                final = result["final_answer"]
                
                # Answer text
                print(f"\n  📄 Answer:")
                answer_text = final['text']
                # Word wrap for long answers
                if len(answer_text) > 65:
                    words = answer_text.split()
                    line = "     "
                    for word in words:
                        if len(line) + len(word) + 1 > 70:
                            print(line)
                            line = "     " + word
                        else:
                            line += (" " if line.strip() else "") + word
                    if line.strip():
                        print(line)
                else:
                    print(f"     {answer_text}")
                
                # Metrics
                print(f"\n  📊 Confidence: {final['confidence']:.2%}")
                print(f"  🔍 Hallucination: {'Detected' if final['hallucination_detected'] else 'Not detected'}")
                print(f"  ✅ Status: {final.get('status', 'Processed')}")
                
                if final.get("refinement_applied"):
                    print(f"  🔧 Refinement: Applied")

                # Sources
                if final.get("sources"):
                    print(f"\n  📚 Sources:")
                    for idx, source in enumerate(final["sources"], 1):
                        print(f"     {idx}. {source}")
            
            print("\n" + "═" * 70)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. \n<< END >>\n")
            break
            
        except Exception as e:
            print(f"\n❌ Pipeline Error: {e}")
            import traceback
            print("\n" + "─" * 70)
            print("Error Details:")
            print("─" * 70)
            traceback.print_exc()
            print("─" * 70)
            print("\nYou can continue with another query or type 'quit' to exit.")


if __name__ == "__main__":
    main()