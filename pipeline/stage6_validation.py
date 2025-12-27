# pipeline/stage6_validation.py

"""
Stage 6: Language Validation with Alif Urdu LLM
- Naturalness scoring
- Code-mixing consistency validation
- Cultural appropriateness checks
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# ================= CONFIG =================

STAGE5_OUTPUT = Path("data/stage5_output.json")
OUTPUT_FILE = Path("data/stage6_output.json")

# Alif model from HuggingFace (Traversaal AI's Urdu-English LLM)
ALIF_MODEL = "traversaal-ai/alif-1.0-8b-instruct"
# Alternative models if Alif not available:
# "meta-llama/Llama-3.1-8B-Instruct" (good multilingual support)
# "CohereForAI/aya-expanse-8b" (multilingual including Urdu)
# "bigscience/bloomz-7b1" (has Urdu in training data)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Validation thresholds
NATURALNESS_THRESHOLD = 0.70
CULTURAL_THRESHOLD = 0.75
CODE_MIXING_THRESHOLD = 0.70

# =========================================


class AlifValidator:
    """
    Alif-based validator for Urdu-English code-mixed text
    """
    
    def __init__(self, model_name: str = ALIF_MODEL):
        print(f"[Stage 6] Loading model: {model_name}")
        print(f"[Stage 6] Device: {DEVICE}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Handle tokenizer pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if DEVICE == "cpu":
                self.model = self.model.to(DEVICE)
            
            self.model.eval()
            print(f"Model loaded successfully\n")
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("\nTrying fallback: rule-based validation...")
            self.model = None
            self.tokenizer = None
            print("Fallback mode enabled\n")
    
    def score_naturalness(self, text: str, original: str = "") -> Dict:
        """
        Score naturalness of Urdu-English code-mixed text
        Returns score between 0-1
        """
        # Fallback mode: rule-based scoring
        if self.model is None:
            return self._fallback_naturalness(text)
        
        prompt = self._build_prompt(
            task="naturalness",
            text=text,
            original=original
        )
        
        response = self._generate(prompt, max_tokens=120)
        score = self._extract_score(response)
        
        return {
            "score": score,
            "passed": score >= NATURALNESS_THRESHOLD,
            "explanation": response
        }
    
    def validate_code_mixing_consistency(self, text: str) -> Dict:
        """
        Check if code-switching is consistent and follows natural patterns
        """
        # Fallback mode: rule-based scoring
        if self.model is None:
            return self._fallback_code_mixing(text)
        
        prompt = self._build_prompt(
            task="code_mixing",
            text=text
        )
        
        response = self._generate(prompt, max_tokens=120)
        score = self._extract_score(response)
        
        # Additional heuristics
        switch_count = self._count_switches(text)
        has_unnatural_switches = self._detect_unnatural_switches(text)
        
        # Penalize if too many abrupt switches
        if switch_count > 10 or has_unnatural_switches:
            score = max(0.0, score - 0.15)
        
        return {
            "score": score,
            "passed": score >= CODE_MIXING_THRESHOLD,
            "switch_count": switch_count,
            "explanation": response
        }
    
    def check_cultural_appropriateness(self, text: str, context: str = "") -> Dict:
        """
        Validate cultural appropriateness for Pakistani/South Asian context
        """
        # Fallback mode: rule-based scoring
        if self.model is None:
            return self._fallback_cultural(text)
        
        prompt = self._build_prompt(
            task="cultural",
            text=text,
            original=context
        )
        
        response = self._generate(prompt, max_tokens=150)
        score = self._extract_score(response)
        
        return {
            "score": score,
            "passed": score >= CULTURAL_THRESHOLD,
            "explanation": response
        }
    
    
    def suggest_refinement(self, text: str, failed_checks: List[str]) -> str:
        """
        Use Alif to refine text that failed validation
        """
        if self.model is None:
            # In fallback mode, return original text
            return text
        
        issues = ", ".join(failed_checks)
        
        prompt = f"""The following Urdu-English text has issues with {issues}.
Please provide an improved version that maintains the meaning but fixes these issues.

Original text: {text}

Improved text:"""
        
        refined = self._generate(prompt, max_tokens=200)
        
        # Clean up the response
        refined = refined.strip()
        if refined.startswith(("Improved:", "Improved text:", "Text:")):
            refined = refined.split(":", 1)[1].strip()
        
        return refined if refined else text
    
    # ========== FALLBACK METHODS (Rule-based when model unavailable) ==========
    
    def _fallback_naturalness(self, text: str) -> Dict:
        """Rule-based naturalness scoring when model unavailable"""
        score = 0.75  # Default reasonable score
        
        # Check for obvious issues
        if not text.strip():
            score = 0.0
        elif len(text.split()) < 3:
            score = 0.6  # Very short might be unnatural
        elif text.isupper():
            score = 0.5  # All caps is unnatural
        elif len(set(text)) < len(text) * 0.1:  # Too repetitive
            score = 0.4
        
        return {
            "score": score,
            "passed": score >= NATURALNESS_THRESHOLD,
            "explanation": "Rule-based evaluation (model unavailable)"
        }
    
    def _fallback_code_mixing(self, text: str) -> Dict:
        """Rule-based code-mixing scoring"""
        switch_count = self._count_switches(text)
        has_unnatural = self._detect_unnatural_switches(text)
        
        # Good mixing: 2-8 switches, no mid-word switches
        if 2 <= switch_count <= 8 and not has_unnatural:
            score = 0.85
        elif switch_count <= 1:
            score = 0.95  # Mostly one language is also fine
        elif switch_count > 10:
            score = 0.60  # Too many switches
        elif has_unnatural:
            score = 0.55  # Unnatural switching
        else:
            score = 0.75
        
        return {
            "score": score,
            "passed": score >= CODE_MIXING_THRESHOLD,
            "switch_count": switch_count,
            "explanation": "Rule-based evaluation (model unavailable)"
        }
    
    def _fallback_cultural(self, text: str) -> Dict:
        """Rule-based cultural appropriateness"""
        score = 0.80  # Default reasonable score
        
        # Check for obviously inappropriate patterns (extend as needed)
        inappropriate_patterns = []  # Add patterns if needed
        
        text_lower = text.lower()
        for pattern in inappropriate_patterns:
            if pattern in text_lower:
                score = 0.3
                break
        
        return {
            "score": score,
            "passed": score >= CULTURAL_THRESHOLD,
            "explanation": "Rule-based evaluation (model unavailable)"
        }
    
    # ========== END FALLBACK METHODS ==========
    
    def _build_prompt(self, task: str, text: str, original: str = "") -> str:
        """
        Build task-specific prompts for Alif
        """
        prompts = {
            "naturalness": f"""Evaluate how natural this Urdu-English code-mixed text sounds to a bilingual speaker.
Consider: word choice, sentence flow, mixing patterns.

Text: {text}
{f"Original context: {original}" if original else ""}

Rate naturalness from 0 (very unnatural) to 10 (perfectly natural) and explain briefly:""",
            
            "code_mixing": f"""Analyze the code-switching patterns in this Urdu-English text.
Check if switches occur at natural boundaries and follow common bilingual patterns.

Text: {text}

Rate consistency from 0 (very inconsistent) to 10 (perfectly consistent) and explain:""",
            
            "cultural": f"""Assess if this text is culturally appropriate for Pakistani/South Asian Urdu-English speakers.
Check for: respectful language, cultural norms, contextually appropriate expressions.

Text: {text}
{f"Original context: {original}" if original else ""}

Rate appropriateness from 0 (inappropriate) to 10 (completely appropriate) and explain:"""
        }
        
        return prompts.get(task, "")
    
    def _generate(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Generate response from model
        """
        if self.model is None or self.tokenizer is None:
            return "Model unavailable - using rule-based evaluation"
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only generated portion
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            return response
        
        except Exception as e:
            print(f"Generation error: {e}")
            return "Generation failed"
    
    def _extract_score(self, response: str) -> float:
        """
        Extract numerical score from Alif response
        Handles formats: "X/10", "X.X", "X out of 10", "score: X"
        """
        response_lower = response.lower()
        
        # Pattern 1: X/10 format
        match = re.search(r'(\d+\.?\d*)\s*/\s*10', response)
        if match:
            return float(match.group(1)) / 10.0
        
        # Pattern 2: X out of 10
        match = re.search(r'(\d+\.?\d*)\s+out\s+of\s+10', response_lower)
        if match:
            return float(match.group(1)) / 10.0
        
        # Pattern 3: score/rating: X
        match = re.search(r'(?:score|rating)[:\s]+(\d+\.?\d*)', response_lower)
        if match:
            val = float(match.group(1))
            return val / 10.0 if val > 1.5 else val
        
        # Pattern 4: standalone number at start
        match = re.search(r'^(\d+\.?\d*)', response.strip())
        if match:
            val = float(match.group(1))
            return val / 10.0 if val > 1.5 else val
        
        # Fallback: sentiment analysis
        return self._sentiment_score(response_lower)
    
    def _sentiment_score(self, text: str) -> float:
        """
        Fallback scoring based on sentiment keywords
        """
        positive = ['excellent', 'good', 'natural', 'appropriate', 'correct', 
                   'proper', 'consistent', 'well', 'perfectly', 'fine']
        negative = ['poor', 'unnatural', 'inappropriate', 'incorrect', 
                   'awkward', 'inconsistent', 'issue', 'problem', 'error']
        
        pos_count = sum(1 for word in positive if word in text)
        neg_count = sum(1 for word in negative if word in text)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.75  # Neutral default
        
        return pos_count / total
    
    def _count_switches(self, text: str) -> int:
        """
        Count code-switching points (simple heuristic)
        """
        # Detect transitions between Urdu script and Latin script
        switches = 0
        prev_script = None
        
        for char in text:
            if '\u0600' <= char <= '\u06FF':  # Urdu/Arabic script
                curr_script = 'urdu'
            elif char.isalpha():  # Latin script
                curr_script = 'english'
            else:
                continue
            
            if prev_script and curr_script != prev_script:
                switches += 1
            prev_script = curr_script
        
        return switches
    
    def _detect_unnatural_switches(self, text: str) -> bool:
        """
        Detect if switches occur mid-word (unnatural)
        """
        words = text.split()
        
        for word in words:
            has_urdu = any('\u0600' <= c <= '\u06FF' for c in word)
            has_latin = any(c.isalpha() and (c < '\u0600' or c > '\u06FF') for c in word)
            
            # If single word has both scripts (not counting punctuation)
            if has_urdu and has_latin and len(word) > 3:
                return True
        
        return False


def validate_stage5_output(stage5_data: List[Dict]) -> List[Dict]:
    """
    Main validation pipeline for Stage 6
    
    Input: Stage 5 output with format:
    {
        "corrected_text": "...",
        "hallucination_type": "intrinsic",
        "explanation": "..."
    }
    
    Output: Validated results with scores
    """
    validator = AlifValidator()
    results = []
    
    total = len(stage5_data)
    print(f"[Stage 6] Processing {total} items from Stage 5\n")
    
    for idx, item in enumerate(stage5_data, 1):
        print(f"Processing {idx}/{total}...", end=" ")
        
        # Extract Stage 5 data
        corrected_text = item.get("corrected_text", "")
        original_text = item.get("original_text", "")
        hallucination_type = item.get("hallucination_type", "unknown")
        stage5_explanation = item.get("explanation", "")
        
        if not corrected_text:
            print("Empty text, skipping")
            continue
        
        # Run all validation checks
        naturalness = validator.score_naturalness(corrected_text, original_text)
        code_mixing = validator.validate_code_mixing_consistency(corrected_text)
        cultural = validator.check_cultural_appropriateness(corrected_text, original_text)
        
        # Calculate overall validation score
        overall_score = (
            naturalness["score"] * 0.4 +  # Weight naturalness higher
            code_mixing["score"] * 0.3 +
            cultural["score"] * 0.3
        )
        
        all_passed = (
            naturalness["passed"] and
            code_mixing["passed"] and
            cultural["passed"]
        )
        
        # Attempt refinement if needed
        final_text = corrected_text
        refinement_applied = False
        
        if not all_passed:
            failed_checks = []
            if not naturalness["passed"]:
                failed_checks.append("naturalness")
            if not code_mixing["passed"]:
                failed_checks.append("code-mixing consistency")
            if not cultural["passed"]:
                failed_checks.append("cultural appropriateness")
            
            refined = validator.suggest_refinement(corrected_text, failed_checks)
            
            if refined and refined != corrected_text:
                final_text = refined
                refinement_applied = True
        
        # Build output
        result = {
            "original_text": original_text,
            "stage5_corrected": corrected_text,
            "final_text": final_text,
            "validation": {
                "naturalness": naturalness,
                "code_mixing_consistency": code_mixing,
                "cultural_appropriateness": cultural,
                "overall_score": round(overall_score, 4),
                "all_checks_passed": all_passed
            },
            "metadata": {
                "refinement_applied": refinement_applied,
                "hallucination_type": hallucination_type,
                "stage5_explanation": stage5_explanation
            }
        }
        
        results.append(result)
        
        status = "✓" if all_passed else ("🔧" if refinement_applied else "⚠️")
        print(f"{status} Score: {overall_score:.3f}")
    
    return results


def generate_validation_summary(results: List[Dict]) -> Dict:
    """
    Generate summary statistics for validation results
    """
    total = len(results)
    if total == 0:
        return {}
    
    passed_all = sum(1 for r in results 
                    if r["validation"]["all_checks_passed"])
    
    refined = sum(1 for r in results 
                 if r["metadata"]["refinement_applied"])
    
    avg_scores = {
        "overall": sum(r["validation"]["overall_score"] for r in results) / total,
        "naturalness": sum(r["validation"]["naturalness"]["score"] for r in results) / total,
        "code_mixing": sum(r["validation"]["code_mixing_consistency"]["score"] for r in results) / total,
        "cultural": sum(r["validation"]["cultural_appropriateness"]["score"] for r in results) / total
    }
    
    # Per hallucination type analysis
    by_type = {}
    for r in results:
        htype = r["metadata"]["hallucination_type"]
        if htype not in by_type:
            by_type[htype] = {"count": 0, "passed": 0, "avg_score": 0}
        
        by_type[htype]["count"] += 1
        if r["validation"]["all_checks_passed"]:
            by_type[htype]["passed"] += 1
        by_type[htype]["avg_score"] += r["validation"]["overall_score"]
    
    for htype in by_type:
        count = by_type[htype]["count"]
        by_type[htype]["avg_score"] /= count
        by_type[htype]["pass_rate"] = by_type[htype]["passed"] / count
    
    return {
        "total_items": total,
        "passed_all_checks": passed_all,
        "pass_rate": passed_all / total,
        "refinements_applied": refined,
        "refinement_rate": refined / total,
        "average_scores": avg_scores,
        "by_hallucination_type": by_type
    }


if __name__ == "__main__":
    print("="*60)
    print("STAGE 6: ALIF VALIDATION & LANGUAGE QUALITY CHECK")
    print("="*60)
    print()
    
    # Load Stage 5 output
    if not STAGE5_OUTPUT.exists():
        print(f"Error: {STAGE5_OUTPUT} not found")
        print("Please run Stage 5 first")
        exit(1)
    
    with open(STAGE5_OUTPUT, "r", encoding="utf-8") as f:
        stage5_data = json.load(f)
    
    # Run validation
    results = validate_stage5_output(stage5_data)
    
    # Generate summary
    summary = generate_validation_summary(results)
    
    # Save results
    output_data = {
        "results": results,
        "summary": summary
    }
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total items: {summary['total_items']}")
    print(f"Passed all checks: {summary['passed_all_checks']} ({summary['pass_rate']:.1%})")
    print(f"Refinements applied: {summary['refinements_applied']} ({summary['refinement_rate']:.1%})")
    print()
    print("Average Scores:")
    for metric, score in summary['average_scores'].items():
        print(f"  {metric:20s}: {score:.3f}")
    print()
    
    if summary.get('by_hallucination_type'):
        print("By Hallucination Type:")
        for htype, stats in summary['by_hallucination_type'].items():
            print(f"  {htype:20s}: {stats['count']:3d} items, "
                  f"{stats['pass_rate']:.1%} pass rate, "
                  f"{stats['avg_score']:.3f} avg score")
    
    print()
    print(f"Stage 6 complete! Output saved to {OUTPUT_FILE}")