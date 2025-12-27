# evaluation/metrics.py

"""
Comprehensive evaluation metrics for hallucination detection and correction
- F1, Precision, Recall
- AUROC
- FeQA (Faithfulness QA)
- BLEU
- Custom code-mixing metrics
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
import re


class EvaluationMetrics:
    """
    Compute all evaluation metrics for the pipeline
    """
    
    def __init__(self):
        self.smoothing = SmoothingFunction()
    
    # ================= CLASSIFICATION METRICS =================
    
    def compute_f1_precision_recall(
        self, 
        y_true: List[int], 
        y_pred: List[int]
    ) -> Dict[str, float]:
        """
        Compute F1, Precision, Recall for hallucination detection
        
        Args:
            y_true: Ground truth labels (0=no hallucination, 1=hallucination)
            y_pred: Predicted labels
        """
        return {
            "f1": f1_score(y_true, y_pred, average='binary', zero_division=0),
            "precision": precision_score(y_true, y_pred, average='binary', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='binary', zero_division=0)
        }
    
    def compute_auroc(
        self, 
        y_true: List[int], 
        y_scores: List[float]
    ) -> Dict[str, float]:
        """
        Compute AUROC for hallucination detection
        
        Args:
            y_true: Ground truth binary labels
            y_scores: Predicted probability scores (e.g., entropy scores)
        """
        try:
            auroc = roc_auc_score(y_true, y_scores)
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            
            # Find optimal threshold (Youden's J statistic)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            return {
                "auroc": auroc,
                "optimal_threshold": float(optimal_threshold),
                "tpr_at_optimal": float(tpr[optimal_idx]),
                "fpr_at_optimal": float(fpr[optimal_idx])
            }
        except ValueError as e:
            return {
                "auroc": 0.0,
                "optimal_threshold": 0.5,
                "error": str(e)
            }
    
    # ================= TEXT QUALITY METRICS =================
    
    def compute_bleu(
        self, 
        reference: str, 
        hypothesis: str,
        max_n: int = 4
    ) -> Dict[str, float]:
        """
        Compute BLEU score between reference and hypothesis
        
        Args:
            reference: Ground truth text
            hypothesis: Generated/corrected text
            max_n: Maximum n-gram (default: 4 for BLEU-4)
        """
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        # Individual n-gram scores
        weights = {
            1: (1.0, 0, 0, 0),
            2: (0.5, 0.5, 0, 0),
            3: (0.33, 0.33, 0.33, 0),
            4: (0.25, 0.25, 0.25, 0.25)
        }
        
        scores = {}
        for n in range(1, max_n + 1):
            score = sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=weights[n],
                smoothing_function=self.smoothing.method1
            )
            scores[f"bleu_{n}"] = score
        
        return scores
    
    def compute_feqa(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str,
        retrieved_docs: List[str] = None
    ) -> Dict[str, float]:
        """
        Compute FeQA (Faithfulness QA) score
        Measures faithfulness of generated answer to retrieved documents
        
        Simplified version - can be enhanced with QA model
        """
        if not retrieved_docs:
            # Fallback: compare answer similarity
            return {
                "feqa_score": self._text_overlap(reference_answer, generated_answer),
                "method": "token_overlap"
            }
        
        # Compute overlap between generated answer and each document
        doc_overlaps = [
            self._text_overlap(generated_answer, doc)
            for doc in retrieved_docs
        ]
        
        # FeQA is max overlap with any document
        feqa_score = max(doc_overlaps) if doc_overlaps else 0.0
        
        return {
            "feqa_score": feqa_score,
            "max_doc_overlap": feqa_score,
            "avg_doc_overlap": np.mean(doc_overlaps) if doc_overlaps else 0.0,
            "method": "document_overlap"
        }
    
    def _text_overlap(self, text1: str, text2: str) -> float:
        """
        Compute token overlap ratio between two texts
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    # ================= CODE-MIXING METRICS =================
    
    def compute_code_mixing_index(self, text: str) -> Dict[str, float]:
        """
        Compute Code-Mixing Index (CMI) and related metrics
        """
        # Detect script boundaries
        urdu_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        latin_chars = sum(1 for c in text if c.isalpha() and (c < '\u0600' or c > '\u06FF'))
        total_chars = urdu_chars + latin_chars
        
        if total_chars == 0:
            return {
                "cmi": 0.0,
                "urdu_ratio": 0.0,
                "english_ratio": 0.0,
                "switch_points": 0
            }
        
        # CMI formula: 1 - (max(lang) / total)
        max_lang = max(urdu_chars, latin_chars)
        cmi = 1.0 - (max_lang / total_chars)
        
        # Count switch points
        switch_points = self._count_switch_points(text)
        
        return {
            "cmi": cmi,
            "urdu_ratio": urdu_chars / total_chars,
            "english_ratio": latin_chars / total_chars,
            "switch_points": switch_points,
            "switches_per_100_chars": (switch_points / len(text)) * 100 if text else 0
        }
    
    def _count_switch_points(self, text: str) -> int:
        """Count language switch points in text"""
        switches = 0
        prev_script = None
        
        for char in text:
            if '\u0600' <= char <= '\u06FF':
                curr_script = 'urdu'
            elif char.isalpha():
                curr_script = 'english'
            else:
                continue
            
            if prev_script and curr_script != prev_script:
                switches += 1
            prev_script = curr_script
        
        return switches
    
    # ================= PIPELINE-SPECIFIC METRICS =================
    
    def compute_correction_quality(
        self,
        original: str,
        corrected: str,
        reference: str = None
    ) -> Dict[str, float]:
        """
        Evaluate quality of correction
        """
        metrics = {}
        
        # Edit distance
        metrics["edit_distance"] = self._levenshtein_distance(original, corrected)
        metrics["edit_ratio"] = metrics["edit_distance"] / max(len(original), 1)
        
        # If reference available, compare
        if reference:
            metrics["bleu_vs_reference"] = self.compute_bleu(reference, corrected)["bleu_4"]
            metrics["token_overlap_reference"] = self._text_overlap(reference, corrected)
        
        # Preservation of code-mixing
        orig_cmi = self.compute_code_mixing_index(original)
        corr_cmi = self.compute_code_mixing_index(corrected)
        metrics["cmi_preservation"] = 1.0 - abs(orig_cmi["cmi"] - corr_cmi["cmi"])
        
        return metrics
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    # ================= AGGREGATE METRICS =================
    
    def compute_all_metrics(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict:
        """
        Compute all metrics given predictions and ground truth
        
        Args:
            predictions: List of prediction dicts with keys:
                - hallucination_detected (bool)
                - entropy_score (float)
                - corrected_text (str)
                - original_text (str)
            ground_truth: List of ground truth dicts with keys:
                - has_hallucination (bool)
                - correct_text (str)
        """
        # Extract labels and scores
        y_true = [int(gt.get("has_hallucination", 0)) for gt in ground_truth]
        y_pred = [int(p.get("hallucination_detected", 0)) for p in predictions]
        y_scores = [p.get("entropy_score", 0.0) for p in predictions]
        
        # Classification metrics
        classification = self.compute_f1_precision_recall(y_true, y_pred)
        auroc = self.compute_auroc(y_true, y_scores)
        
        # Text quality metrics (for corrected texts)
        bleu_scores = []
        feqa_scores = []
        correction_qualities = []
        
        for pred, gt in zip(predictions, ground_truth):
            if pred.get("corrected_text") and gt.get("correct_text"):
                bleu = self.compute_bleu(
                    gt["correct_text"],
                    pred["corrected_text"]
                )
                bleu_scores.append(bleu["bleu_4"])
                
                feqa = self.compute_feqa(
                    gt.get("original_text", ""),
                    gt["correct_text"],
                    pred["corrected_text"],
                    retrieved_docs=pred.get("retrieved_docs")
                )
                feqa_scores.append(feqa["feqa_score"])
                
                corr_quality = self.compute_correction_quality(
                    pred.get("original_text", ""),
                    pred["corrected_text"],
                    gt["correct_text"]
                )
                correction_qualities.append(corr_quality)
        
        # Aggregate all metrics
        return {
            "classification": classification,
            "auroc": auroc,
            "text_quality": {
                "avg_bleu_4": np.mean(bleu_scores) if bleu_scores else 0.0,
                "avg_feqa": np.mean(feqa_scores) if feqa_scores else 0.0,
                "bleu_scores": bleu_scores,
                "feqa_scores": feqa_scores
            },
            "correction_quality": {
                "avg_edit_ratio": np.mean([c["edit_ratio"] for c in correction_qualities]) if correction_qualities else 0.0,
                "avg_cmi_preservation": np.mean([c["cmi_preservation"] for c in correction_qualities]) if correction_qualities else 0.0
            },
            "sample_count": len(predictions)
        }


# Convenience functions
def calculate_f1(y_true: List[int], y_pred: List[int]) -> float:
    """Quick F1 calculation"""
    metrics = EvaluationMetrics()
    return metrics.compute_f1_precision_recall(y_true, y_pred)["f1"]


def calculate_bleu(reference: str, hypothesis: str) -> float:
    """Quick BLEU-4 calculation"""
    metrics = EvaluationMetrics()
    return metrics.compute_bleu(reference, hypothesis)["bleu_4"]


def calculate_auroc(y_true: List[int], y_scores: List[float]) -> float:
    """Quick AUROC calculation"""
    metrics = EvaluationMetrics()
    return metrics.compute_auroc(y_true, y_scores)["auroc"]