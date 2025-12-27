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
def run_pipeline(user_query: str, model_response: str) -> Dict[str, Any]:
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
    retrieved_docs = retrieve_facts(
        query=user_query,
        response=model_response
    )
    print("Stage 3 Retrieval Output:", retrieved_docs)
    pipeline_trace["stage3_retrieval"] = retrieved_docs

    # ---------------- Stage 4 ----------------
    print("Running Stage 4: Fact Verification...")
    verification_result = verify_fact(
        query=user_query,
        retrieved_docs=retrieved_docs
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
    final_answer = {
        "text": stage6_results[0]["final_text"],
        "confidence": stage6_results[0]["validation"]["overall_score"],
        "sources": stage5_output.get("sources", []),
        "hallucination_detected": detection_result.get("hallucination_detected", False),
        "refinement_applied": stage6_results[0]["metadata"]["refinement_applied"]
    }

    return {
        "final_answer": final_answer,
        "pipeline_trace": pipeline_trace
    }


# ----------------------------------------------------
# CLI / Debug Entry Point
# ----------------------------------------------------
if __name__ == "__main__":

    print("=== Hallucination Mitigation Pipeline ===\n")

    user_query = input("User Query: ").strip()
    
    model_response = input("Model Response: ").strip()

    result = run_pipeline(user_query, model_response)

    print("\n--- Final Answer ---")
    print(result["final_answer"])

    print("\n--- Pipeline Trace (Debug) ---")
    for stage, output in result["pipeline_trace"].items():
        print(f"\n[{stage}]")
        print(output)
