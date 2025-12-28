# run_pipeline.py

"""
Main pipeline runner for the hallucination detection & correction system.

Pipeline Stages:
Stage 1: Input preprocessing
Stage 2: Hallucination detection
Stage 3: Knowledge retrieval (RHO)
Stage 4: Fact verification (CRAG)
Stage 5: Correction generation
Stage 6: Final response formatting
"""

from typing import Dict, Any

# ---- Stage imports (owned by different teammates) ----
from pipeline.stage1_preprocessing import preprocess_text
from pipeline.stage2_detection import detect_hallucination
from pipeline.stage3_retrieval import retrieve_facts
from pipeline.stage4_verification import verify_fact
from pipeline.stage5_correction import HallucinationCorrector
from pipeline.stage6_validation import validate_stage5_output, generate_validation_summary

# ----------------------------------------------------
# Pipeline Runner
# ----------------------------------------------------
def run_pipeline(user_query: str) -> Dict[str, Any]:
    """
    Runs the complete hallucination detection and correction pipeline.

    Inputs:
        user_query (str): Original user question
        model_response (str): LLM-generated response

    Returns:
        Dict containing final response and intermediate metadata
    """

    pipeline_trace = {}

    # ---------------- Stage 1 ----------------
    print("Running Stage 1: Preprocessing...")
    stage1_output = preprocess_text(user_query)
    print("Stage 1 Preprocessing Output:", stage1_output)
    pipeline_trace["stage1_preprocessing"] = stage1_output

    
    # ---------------- Stage 2 ----------------
    
    # Stage 2 expects a LIST of stage1 items
    print("Running Stage 2: Hallucination Detection...")
    stage2_results = detect_hallucination([stage1_output])

    # Stage 2 always returns a list
    detection_result = stage2_results[0]

    print("Stage 2 Detection Output:", detection_result)
    pipeline_trace["stage2_detection"] = detection_result

    # If NO hallucination detected, return early
    if not detection_result.get("hallucination_detected", False):
        final_answer = {
            "text": detection_result["responses"][0],
            "confidence": 1.0,
            "sources": [],
            "hallucination_detected": False
        }

        return {
            "final_answer": final_answer,
            "pipeline_trace": pipeline_trace
        }

    # ---------------- Stage 3 ----------------
    print("Running Stage 3: Knowledge Retrieval...")
    retrieved_docs_raw = retrieve_facts(
        query=user_query, top_k=5
    )
    print("Stage 3 Retrieval Output:", retrieved_docs_raw)

    # Transform Stage 3 output to Stage 4 expected format
    # Stage 4 expects a list of dicts: [{"text": "...", "source": "...", "date": "YYYY"}, ...]
    retrieved_docs = []
    detected_entity = None
    
    if isinstance(retrieved_docs_raw, dict):
        # Single doc returned
        retrieved_docs.append({
            "text": retrieved_docs_raw.get("extracted_answer", ""),
            "source": retrieved_docs_raw.get("linked_kb_id", "unknown"),
            "date": retrieved_docs_raw.get("date", "unknown")
        })
        detected_entity = retrieved_docs_raw.get("entity")
    elif isinstance(retrieved_docs_raw, list):
        # Multiple docs returned
        for doc in retrieved_docs_raw:
            retrieved_docs.append({
                "text": doc.get("extracted_answer", ""),
                "source": doc.get("linked_kb_id", "unknown"),
                "date": doc.get("date", "unknown")
            })
        if retrieved_docs_raw and isinstance(retrieved_docs_raw[0], dict):
            detected_entity = retrieved_docs_raw[0].get("entity")

    pipeline_trace["stage3_retrieval"] = retrieved_docs

    # ---------------- Stage 4 ----------------
    print("Running Stage 4: Fact Verification...")
    verification_result = verify_fact(
        query=user_query,
        retrieved_docs=retrieved_docs,
        entity=detected_entity
    )
    print("Stage 4 Verification Output:", verification_result)
    pipeline_trace["stage4_verification"] = verification_result


    # ---------------- Stage 5 ----------------
    print("Running Stage 5: Correction Generation...")
    corrector = HallucinationCorrector(correction_method="template")  # or "model" if API key available
    stage5_output = corrector.process_stage2_output(
        detection_result,
        retrieved_docs=retrieved_docs
    )
    print("Stage 5 Correction Output:", stage5_output)
    pipeline_trace["stage5_correction"] = stage5_output

    # ---------------- Stage 6 ----------------
    print("Running Stage 6: Final Validation...")
    stage6_results = validate_stage5_output([stage5_output])

    print("Stage 6 Validation Output:", stage6_results)
    pipeline_trace["stage6_validation"] = stage6_results

    # Final answer preparation
    # Use verified fact if final_text is empty or placeholder
    final_text = stage6_results[0]["final_text"]
    if not final_text or final_text == "No verified fact available" or final_text.strip() == "":
        # Fallback to verified fact from Stage 4
        verified_fact = verification_result.get("verified_fact")
        if verified_fact and verified_fact != "No verified fact available":
            final_text = verified_fact
        elif stage5_output.get("corrected_text"):
            final_text = stage5_output.get("corrected_text")
        else:
            # Last resort: use original response
            final_text = detection_result.get("responses", [""])[0] if detection_result.get("responses") else "No answer available"
    
    final_answer = {
        "text": final_text,
        "confidence": stage6_results[0]["validation"]["overall_score"] if stage6_results else verification_result.get("confidence", 0.0),
        "sources": verification_result.get("sources", stage5_output.get("sources", [])),
        "hallucination_detected": detection_result.get("hallucination_detected", False),
        "refinement_applied": stage6_results[0]["metadata"]["refinement_applied"] if stage6_results else False
    }

    return {
        "final_answer": final_answer,
        "pipeline_trace": pipeline_trace
    }


# ----------------------------------------------------
# CLI / Debug Entry Point
# ----------------------------------------------------
if __name__ == "__main__":

    print("=" * 70)
    print(" Hallucination Detection & Knowledge-Grounded Correction Pipeline")
    print("Type 'quit' to exit at any time.")
    print("=" * 70)

    while True:
        print("\n" + "-" * 50)
        user_query = input("Enter your query: ").strip()

        if user_query.lower() == "quit":
            print("\n Exiting pipeline. Thank you!")
            break

        if not user_query:
            print("  Query cannot be empty. Please try again.")
            continue

        print("\n Running pipeline...\n")

        try:
            result = run_pipeline(user_query)

            # ---------------- Final Answer ----------------
            print(" FINAL ANSWER")
            print("-" * 30)
            final = result.get("final_answer", {})
            print(f" Text       : {final.get('text', 'N/A')}")
            print(f" Confidence : {final.get('confidence', 0):.2f}")
            print(f" Hallucinated: {final.get('hallucination_detected', False)}")

            if final.get("sources"):
                print(" Sources:")
                for src in final["sources"]:
                    print(f"   • {src}")

            # ---------------- Debug Trace ----------------
            print("\n PIPELINE TRACE (DEBUG)")
            print("-" * 30)

            for stage, output in result.get("pipeline_trace", {}).items():
                print(f"\n[{stage.upper()}]")
                print(output)

        except Exception as e:
            print("\n Pipeline execution failed")
            print(f"Error: {e}")

        print("\n" + "=" * 70)
