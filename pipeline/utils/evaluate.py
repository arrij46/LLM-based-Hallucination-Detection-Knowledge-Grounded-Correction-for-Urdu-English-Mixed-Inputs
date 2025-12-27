# evaluation/evaluate.py

"""
End-to-end pipeline evaluation script
Evaluates Stage 6 output against ground truth
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation.metrics import EvaluationMetrics

# ================= CONFIG =================

STAGE6_OUTPUT = Path("data/stage6_output.json")
GROUND_TRUTH = Path("data/ground_truth.json")
RESULTS_DIR = Path("results/evaluation")

# =========================================


class PipelineEvaluator:
    """
    Comprehensive evaluation of the 6-stage pipeline
    """
    
    def __init__(self):
        self.metrics = EvaluationMetrics()
        self.results = {}
    
    def load_data(
        self,
        stage6_path: Path,
        ground_truth_path: Path
    ) -> tuple:
        """Load Stage 6 output and ground truth"""
        
        with open(stage6_path, "r", encoding="utf-8") as f:
            stage6_data = json.load(f)
        
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
        
        # Extract results from Stage 6 output
        if "results" in stage6_data:
            stage6_results = stage6_data["results"]
        else:
            stage6_results = stage6_data
        
        return stage6_results, ground_truth
    
    def prepare_predictions(self, stage6_results: List[Dict]) -> List[Dict]:
        """
        Convert Stage 6 output to evaluation format
        """
        predictions = []
        
        for item in stage6_results:
            pred = {
                "original_text": item.get("original_text", ""),
                "corrected_text": item.get("final_text", ""),
                "hallucination_detected": not item.get("validation", {}).get("all_checks_passed", True),
                "entropy_score": 1.0 - item.get("validation", {}).get("overall_score", 0.5),
                "validation_scores": item.get("validation", {}),
                "retrieved_docs": item.get("metadata", {}).get("sources", [])
            }
            predictions.append(pred)
        
        return predictions
    
    def evaluate_detection(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict:
        """
        Evaluate hallucination detection performance
        """
        y_true = [int(gt.get("has_hallucination", 0)) for gt in ground_truth]
        y_pred = [int(p["hallucination_detected"]) for p in predictions]
        y_scores = [p["entropy_score"] for p in predictions]
        
        # Classification metrics
        classification = self.metrics.compute_f1_precision_recall(y_true, y_pred)
        auroc = self.metrics.compute_auroc(y_true, y_scores)
        
        return {
            **classification,
            **auroc,
            "detection_accuracy": sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        }
    
    def evaluate_correction(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict:
        """
        Evaluate correction quality
        """
        bleu_scores = []
        feqa_scores = []
        correction_metrics = []
        
        for pred, gt in zip(predictions, ground_truth):
            if not gt.get("correct_text"):
                continue
            
            # BLEU
            bleu = self.metrics.compute_bleu(
                gt["correct_text"],
                pred["corrected_text"]
            )
            bleu_scores.append(bleu["bleu_4"])
            
            # FeQA
            feqa = self.metrics.compute_feqa(
                pred["original_text"],
                gt["correct_text"],
                pred["corrected_text"],
                retrieved_docs=pred.get("retrieved_docs")
            )
            feqa_scores.append(feqa["feqa_score"])
            
            # Correction quality
            corr_quality = self.metrics.compute_correction_quality(
                pred["original_text"],
                pred["corrected_text"],
                gt["correct_text"]
            )
            correction_metrics.append(corr_quality)
        
        return {
            "bleu_4": {
                "mean": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
                "std": self._std(bleu_scores),
                "min": min(bleu_scores) if bleu_scores else 0.0,
                "max": max(bleu_scores) if bleu_scores else 0.0
            },
            "feqa": {
                "mean": sum(feqa_scores) / len(feqa_scores) if feqa_scores else 0.0,
                "std": self._std(feqa_scores),
                "min": min(feqa_scores) if feqa_scores else 0.0,
                "max": max(feqa_scores) if feqa_scores else 0.0
            },
            "edit_distance": {
                "mean": sum(c["edit_ratio"] for c in correction_metrics) / len(correction_metrics) if correction_metrics else 0.0
            },
            "cmi_preservation": {
                "mean": sum(c["cmi_preservation"] for c in correction_metrics) / len(correction_metrics) if correction_metrics else 0.0
            }
        }
    
    def evaluate_validation(
        self,
        predictions: List[Dict]
    ) -> Dict:
        """
        Evaluate Stage 6 validation scores
        """
        naturalness = []
        code_mixing = []
        cultural = []
        overall = []
        
        for pred in predictions:
            val = pred.get("validation_scores", {})
            
            if "naturalness" in val:
                naturalness.append(val["naturalness"].get("score", 0.0))
            if "code_mixing_consistency" in val:
                code_mixing.append(val["code_mixing_consistency"].get("score", 0.0))
            if "cultural_appropriateness" in val:
                cultural.append(val["cultural_appropriateness"].get("score", 0.0))
            if "overall_score" in val:
                overall.append(val["overall_score"])
        
        return {
            "naturalness": self._summarize_scores(naturalness),
            "code_mixing_consistency": self._summarize_scores(code_mixing),
            "cultural_appropriateness": self._summarize_scores(cultural),
            "overall_validation": self._summarize_scores(overall)
        }
    
    def evaluate_code_mixing_preservation(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict:
        """
        Evaluate how well code-mixing is preserved
        """
        cmi_diffs = []
        
        for pred, gt in zip(predictions, ground_truth):
            original_cmi = self.metrics.compute_code_mixing_index(pred["original_text"])
            corrected_cmi = self.metrics.compute_code_mixing_index(pred["corrected_text"])
            
            if gt.get("correct_text"):
                reference_cmi = self.metrics.compute_code_mixing_index(gt["correct_text"])
                cmi_diff = abs(corrected_cmi["cmi"] - reference_cmi["cmi"])
            else:
                cmi_diff = abs(corrected_cmi["cmi"] - original_cmi["cmi"])
            
            cmi_diffs.append(cmi_diff)
        
        return {
            "cmi_preservation_score": 1.0 - (sum(cmi_diffs) / len(cmi_diffs) if cmi_diffs else 0.0),
            "avg_cmi_difference": sum(cmi_diffs) / len(cmi_diffs) if cmi_diffs else 0.0,
            "cmi_differences": cmi_diffs
        }
    
    def run_full_evaluation(
        self,
        stage6_path: Path = STAGE6_OUTPUT,
        ground_truth_path: Path = GROUND_TRUTH
    ) -> Dict:
        """
        Run complete evaluation pipeline
        """
        print("="*60)
        print("PIPELINE EVALUATION")
        print("="*60)
        print()
        
        # Load data
        print("Loading data...")
        stage6_results, ground_truth = self.load_data(stage6_path, ground_truth_path)
        predictions = self.prepare_predictions(stage6_results)
        
        print(f"  Stage 6 results: {len(stage6_results)} items")
        print(f"  Ground truth: {len(ground_truth)} items")
        print()
        
        # Run evaluations
        print("Evaluating detection performance...")
        detection_results = self.evaluate_detection(predictions, ground_truth)
        
        print("Evaluating correction quality...")
        correction_results = self.evaluate_correction(predictions, ground_truth)
        
        print("Evaluating validation scores...")
        validation_results = self.evaluate_validation(predictions)
        
        print("Evaluating code-mixing preservation...")
        code_mixing_results = self.evaluate_code_mixing_preservation(predictions, ground_truth)
        
        # Compile all results
        self.results = {
            "detection": detection_results,
            "correction": correction_results,
            "validation": validation_results,
            "code_mixing": code_mixing_results,
            "sample_count": len(predictions)
        }
        
        return self.results
    
    def save_results(self, output_path: Path = None):
        """Save evaluation results to JSON"""
        if output_path is None:
            output_path = RESULTS_DIR / "evaluation_results.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_path}")
    
    def generate_report(self) -> str:
        """Generate human-readable evaluation report"""
        report = []
        report.append("="*60)
        report.append("EVALUATION REPORT")
        report.append("="*60)
        report.append("")
        
        # Detection performance
        report.append("HALLUCINATION DETECTION")
        report.append("-" * 40)
        det = self.results["detection"]
        report.append(f"  F1 Score:          {det['f1']:.4f}")
        report.append(f"  Precision:         {det['precision']:.4f}")
        report.append(f"  Recall:            {det['recall']:.4f}")
        report.append(f"  AUROC:             {det['auroc']:.4f}")
        report.append(f"  Accuracy:          {det['detection_accuracy']:.4f}")
        report.append("")
        
        # Correction quality
        report.append("CORRECTION QUALITY")
        report.append("-" * 40)
        corr = self.results["correction"]
        report.append(f"  BLEU-4:            {corr['bleu_4']['mean']:.4f} ± {corr['bleu_4']['std']:.4f}")
        report.append(f"  FeQA:              {corr['feqa']['mean']:.4f} ± {corr['feqa']['std']:.4f}")
        report.append(f"  Edit Distance:     {corr['edit_distance']['mean']:.4f}")
        report.append(f"  CMI Preservation:  {corr['cmi_preservation']['mean']:.4f}")
        report.append("")
        
        # Validation scores
        report.append("VALIDATION SCORES (Stage 6)")
        report.append("-" * 40)
        val = self.results["validation"]
        for metric, scores in val.items():
            report.append(f"  {metric:25s}: {scores['mean']:.4f} ± {scores['std']:.4f}")
        report.append("")
        
        # Code-mixing
        report.append("CODE-MIXING PRESERVATION")
        report.append("-" * 40)
        cm = self.results["code_mixing"]
        report.append(f"  Preservation Score: {cm['cmi_preservation_score']:.4f}")
        report.append(f"  Avg CMI Difference: {cm['avg_cmi_difference']:.4f}")
        report.append("")
        
        report.append("="*60)
        
        return "\n".join(report)
    
    def _summarize_scores(self, scores: List[float]) -> Dict:
        """Generate summary statistics for a list of scores"""
        if not scores:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "mean": sum(scores) / len(scores),
            "std": self._std(scores),
            "min": min(scores),
            "max": max(scores)
        }
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


def main():
    """Main evaluation entry point"""
    evaluator = PipelineEvaluator()
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    # Print report
    print()
    report = evaluator.generate_report()
    print(report)
    
    # Save results
    evaluator.save_results()
    
    # Save report
    report_path = RESULTS_DIR / "evaluation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {report_path}")
    
    return results


if __name__ == "__main__":
    main()