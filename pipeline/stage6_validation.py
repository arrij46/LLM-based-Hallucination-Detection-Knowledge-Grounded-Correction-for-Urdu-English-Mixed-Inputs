# pipeline/stage6_validation.py

"""
Stage 6: Language Validation & Roman Urdu Conversion (3-TIER FALLBACK)
- Tier 1: Alif Urdu model (traversaal-ai/alif-1.0-8b-instruct)
- Tier 2: Groq API (llama-3.3-70b-versatile)
- Tier 3: Rule-based validation

Features:
- Naturalness scoring
- Code-mixing consistency validation
- Cultural appropriateness checks
- Roman Urdu conversion/refinement
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import re
import os

# Try importing transformers for Alif model
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Alif model disabled.")

# Try importing Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not available. Groq fallback disabled.")


# ================= CONFIG =================

STAGE5_OUTPUT = Path("data/stage5_output.json")
OUTPUT_FILE = Path("data/stage6_output.json")

# Alif model
ALIF_MODEL = "traversaal-ai/alif-1.0-8b-instruct"
DEVICE = "cuda" if (TRANSFORMERS_AVAILABLE and torch.cuda.is_available()) else "cpu"

# Groq model
GROQ_MODEL = "llama-3.3-70b-versatile"

# Validation thresholds
NATURALNESS_THRESHOLD = 0.70
CULTURAL_THRESHOLD = 0.75
CODE_MIXING_THRESHOLD = 0.70

# =========================================


def validate_groq_key(client):
    """Validate Groq API key"""
    try:
        client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model=GROQ_MODEL,
            max_tokens=5
        )
        return True
    except Exception as e:
        return False


class AlifValidator:
    """
    3-Tier validation system:
    1. Alif Urdu model (primary)
    2. Groq API (fallback 1)
    3. Rule-based (fallback 2)
    """
    
    def __init__(self, model_name: str = ALIF_MODEL):
        self.validation_mode = None
        self.alif_model = None
        self.alif_tokenizer = None
        self.groq_client = None
        
        print(f"[Stage 6] Initializing 3-tier validation system...")
        
        # Tier 1: Try loading Alif model
        if TRANSFORMERS_AVAILABLE:
            self._try_load_alif(model_name)
        
        # Tier 2: Try initializing Groq
        if not self.alif_model and GROQ_AVAILABLE:
            self._try_initialize_groq()
        
        # Tier 3: Fallback to rule-based
        if not self.alif_model and not self.groq_client:
            self.validation_mode = "rule-based"
            print(f"[Stage 6] ✅ Using: Rule-based validation (Tier 3)\n")
    
    def _try_load_alif(self, model_name: str):
        """Try loading Alif model (Tier 1)"""
        try:
            print(f"[Stage 6] Attempting to load Alif model: {model_name}")
            print(f"[Stage 6] Device: {DEVICE}")
            
            self.alif_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            if self.alif_tokenizer.pad_token is None:
                self.alif_tokenizer.pad_token = self.alif_tokenizer.eos_token
            
            self.alif_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if DEVICE == "cpu":
                self.alif_model = self.alif_model.to(DEVICE)
            
            self.alif_model.eval()
            self.validation_mode = "alif"
            print(f"[Stage 6] ✅ Using: Alif model (Tier 1 - Primary)\n")
            
        except Exception as e:
            print(f"[Stage 6] ❌ Alif model failed: {e}")
            self.alif_model = None
            self.alif_tokenizer = None
    
    def _try_initialize_groq(self):
        """Try initializing Groq API (Tier 2)"""
        api_key = os.environ.get("GROQ_API_KEY")
        
        if not api_key:
            print(f"[Stage 6] ⚠️ GROQ_API_KEY not found")
            return
        
        try:
            print(f"[Stage 6] Attempting to initialize Groq API...")
            self.groq_client = Groq(api_key=api_key)
            
            if validate_groq_key(self.groq_client):
                self.validation_mode = "groq"
                print(f"[Stage 6] ✅ Using: Groq API (Tier 2 - Fallback)\n")
            else:
                print(f"[Stage 6] ❌ Groq validation failed")
                self.groq_client = None
        except Exception as e:
            print(f"[Stage 6] ❌ Groq initialization failed: {e}")
            self.groq_client = None
    
    # ========================================================================
    # MAIN VALIDATION METHODS
    # ========================================================================
    
    def score_naturalness(self, text: str, original: str = "") -> Dict:
        """Score naturalness using best available method"""
        if self.validation_mode == "alif":
            return self._alif_naturalness(text, original)
        elif self.validation_mode == "groq":
            return self._groq_naturalness(text, original)
        else:
            return self._fallback_naturalness(text)
    
    def validate_code_mixing_consistency(self, text: str) -> Dict:
        """Validate code-mixing using best available method"""
        if self.validation_mode == "alif":
            return self._alif_code_mixing(text)
        elif self.validation_mode == "groq":
            return self._groq_code_mixing(text)
        else:
            return self._fallback_code_mixing(text)
    
    def check_cultural_appropriateness(self, text: str, context: str = "") -> Dict:
        """Check cultural appropriateness using best available method"""
        if self.validation_mode == "alif":
            return self._alif_cultural(text, context)
        elif self.validation_mode == "groq":
            return self._groq_cultural(text, context)
        else:
            return self._fallback_cultural(text)
    
    def suggest_refinement(self, text: str, failed_checks: List[str]) -> str:
        """Suggest refinement using best available method"""
        if self.validation_mode == "alif":
            return self._alif_refine(text, failed_checks)
        elif self.validation_mode == "groq":
            return self._groq_refine(text, failed_checks)
        else:
            return text  # No refinement in rule-based mode
    
    # ========================================================================
    # ALIF MODEL METHODS (Tier 1)
    # ========================================================================
    
    def _alif_generate(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate response using Alif model"""
        if not self.alif_model or not self.alif_tokenizer:
            return ""
        
        try:
            inputs = self.alif_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = self.alif_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.alif_tokenizer.pad_token_id
                )
            
            response = self.alif_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            print(f"[Alif Error] {e}")
            return ""
    
    def _alif_naturalness(self, text: str, original: str = "") -> Dict:
        """Score naturalness using Alif"""
        prompt = f"""Evaluate how natural this Urdu-English code-mixed text sounds.
Consider: word choice, sentence flow, mixing patterns.

Text: {text}
{f"Original context: {original}" if original else ""}

Rate naturalness from 0 (very unnatural) to 10 (perfectly natural).
Provide only the number:"""
        
        response = self._alif_generate(prompt, max_tokens=50)
        score = self._extract_score(response)
        
        return {
            "score": score,
            "passed": score >= NATURALNESS_THRESHOLD,
            "explanation": f"Alif model score: {score:.2f}"
        }
    
    def _alif_code_mixing(self, text: str) -> Dict:
        """Validate code-mixing using Alif"""
        prompt = f"""Analyze the code-switching patterns in this Urdu-English text.
Check if switches occur at natural boundaries.

Text: {text}

Rate consistency from 0 (very inconsistent) to 10 (perfectly consistent).
Provide only the number:"""
        
        response = self._alif_generate(prompt, max_tokens=50)
        score = self._extract_score(response)
        
        switch_count = self._count_switches(text)
        
        return {
            "score": score,
            "passed": score >= CODE_MIXING_THRESHOLD,
            "switch_count": switch_count,
            "explanation": f"Alif model score: {score:.2f}"
        }
    
    def _alif_cultural(self, text: str, context: str = "") -> Dict:
        """Check cultural appropriateness using Alif"""
        prompt = f"""Assess if this text is culturally appropriate for Pakistani/South Asian Urdu-English speakers.

Text: {text}
{f"Context: {context}" if context else ""}

Rate appropriateness from 0 (inappropriate) to 10 (completely appropriate).
Provide only the number:"""
        
        response = self._alif_generate(prompt, max_tokens=50)
        score = self._extract_score(response)
        
        return {
            "score": score,
            "passed": score >= CULTURAL_THRESHOLD,
            "explanation": f"Alif model score: {score:.2f}"
        }
    
    def _alif_refine(self, text: str, failed_checks: List[str]) -> str:
        """Refine text using Alif"""
        issues = ", ".join(failed_checks)
        
        prompt = f"""Improve this Urdu-English text to fix issues with {issues}.
Keep the meaning but make it more natural for Pakistani bilingual speakers.
Use Roman Urdu where appropriate.

Original: {text}

Improved version (output ONLY the improved text):"""
        
        refined = self._alif_generate(prompt, max_tokens=200)
        
        # Clean up response
        refined = refined.strip()
        for prefix in ["Improved:", "Improved version:", "Text:", "Output:"]:
            if refined.lower().startswith(prefix.lower()):
                refined = refined[len(prefix):].strip()
        
        return refined if refined and len(refined) > 10 else text
    
    # ========================================================================
    # GROQ API METHODS (Tier 2)
    # ========================================================================
    
    def _groq_generate(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate response using Groq API"""
        if not self.groq_client:
            return ""
        
        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Groq Error] {e}")
            return ""
    
    def _groq_naturalness(self, text: str, original: str = "") -> Dict:
        """Score naturalness using Groq"""
        prompt = f"""Evaluate how natural this Urdu-English code-mixed text sounds to a Pakistani bilingual speaker.

Text: {text}
{f"Original: {original}" if original else ""}

Rate from 0.0 (very unnatural) to 1.0 (perfectly natural).
Respond with ONLY the number (e.g., 0.85):"""
        
        response = self._groq_generate(prompt, max_tokens=50)
        score = self._extract_score(response)
        
        return {
            "score": score,
            "passed": score >= NATURALNESS_THRESHOLD,
            "explanation": f"Groq API score: {score:.2f}"
        }
    
    def _groq_code_mixing(self, text: str) -> Dict:
        """Validate code-mixing using Groq"""
        prompt = f"""Analyze code-switching patterns in this Urdu-English text.

Text: {text}

Rate consistency from 0.0 (very inconsistent) to 1.0 (perfectly consistent).
Respond with ONLY the number:"""
        
        response = self._groq_generate(prompt, max_tokens=50)
        score = self._extract_score(response)
        
        switch_count = self._count_switches(text)
        
        return {
            "score": score,
            "passed": score >= CODE_MIXING_THRESHOLD,
            "switch_count": switch_count,
            "explanation": f"Groq API score: {score:.2f}"
        }
    
    def _groq_cultural(self, text: str, context: str = "") -> Dict:
        """Check cultural appropriateness using Groq"""
        prompt = f"""Is this text culturally appropriate for Pakistani/South Asian Urdu-English speakers?

Text: {text}
{f"Context: {context}" if context else ""}

Rate from 0.0 (inappropriate) to 1.0 (completely appropriate).
Respond with ONLY the number:"""
        
        response = self._groq_generate(prompt, max_tokens=50)
        score = self._extract_score(response)
        
        return {
            "score": score,
            "passed": score >= CULTURAL_THRESHOLD,
            "explanation": f"Groq API score: {score:.2f}"
        }
    
    def _groq_refine(self, text: str, failed_checks: List[str]) -> str:
        """Refine text using Groq"""
        issues = ", ".join(failed_checks)
        
        prompt = f"""Improve this Urdu-English text to fix issues with {issues}.
Keep the meaning but make it more natural for Pakistani bilingual speakers.
Use Roman Urdu where appropriate.

Original: {text}

Instructions:
- Maintain factual content
- Improve {issues}
- Use natural code-mixing
- Output ONLY the improved text

Improved text:"""
        
        refined = self._groq_generate(prompt, max_tokens=200)
        
        # Clean up response
        refined = refined.strip()
        for prefix in ["Improved text:", "Improved:", "Text:", "Output:"]:
            if refined.lower().startswith(prefix.lower()):
                refined = refined[len(prefix):].strip()
        
        return refined if refined and len(refined) > 10 else text
    
    # ========================================================================
    # RULE-BASED METHODS (Tier 3 - Final Fallback)
    # ========================================================================
    
    def _fallback_naturalness(self, text: str) -> Dict:
        """Rule-based naturalness scoring"""
        score = 0.75
        
        if not text.strip():
            score = 0.0
        elif len(text.split()) < 3:
            score = 0.6
        elif text.isupper():
            score = 0.5
        elif len(set(text)) < len(text) * 0.1:
            score = 0.4
        else:
            if text.strip().endswith(('.', '!', '?', '।')):
                score += 0.05
            if 10 <= len(text.split()) <= 30:
                score += 0.05
        
        score = min(score, 1.0)
        
        return {
            "score": score,
            "passed": score >= NATURALNESS_THRESHOLD,
            "explanation": "Rule-based evaluation"
        }
    
    def _fallback_code_mixing(self, text: str) -> Dict:
        """Rule-based code-mixing scoring"""
        switch_count = self._count_switches(text)
        has_unnatural = self._detect_unnatural_switches(text)
        
        if 2 <= switch_count <= 8 and not has_unnatural:
            score = 0.85
        elif switch_count <= 1:
            score = 0.95
        elif switch_count > 10:
            score = 0.60
        elif has_unnatural:
            score = 0.55
        else:
            score = 0.75
        
        return {
            "score": score,
            "passed": score >= CODE_MIXING_THRESHOLD,
            "switch_count": switch_count,
            "explanation": "Rule-based evaluation"
        }
    
    def _fallback_cultural(self, text: str) -> Dict:
        """Rule-based cultural appropriateness"""
        score = 0.80
        
        text_lower = text.lower()
        if any(word in text_lower for word in ['pakistan', 'quaid', 'jinnah', 'iqbal']):
            score += 0.05
        
        score = min(score, 1.0)
        
        return {
            "score": score,
            "passed": score >= CULTURAL_THRESHOLD,
            "explanation": "Rule-based evaluation"
        }
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _extract_score(self, response: str) -> float:
        """Extract numerical score from response"""
        if not response:
            return 0.70  # Default reasonable score
        
        # Try to find number in response
        numbers = re.findall(r'\d+\.?\d*', response)
        
        if numbers:
            score = float(numbers[0])
            
            # Normalize to 0-1 range
            if score > 1.0:
                score = score / 10.0  # Assume 0-10 scale
            
            return max(0.0, min(1.0, score))
        
        return 0.70  # Default
    
    def _count_switches(self, text: str) -> int:
        """Count language switches"""
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
    
    def _detect_unnatural_switches(self, text: str) -> bool:
        """Detect unnatural mid-word switches"""
        words = text.split()
        
        for word in words:
            has_urdu = any('\u0600' <= c <= '\u06FF' for c in word)
            has_latin = any(c.isalpha() and (c < '\u0600' or c > '\u06FF') for c in word)
            
            if has_urdu and has_latin and len(word) > 3:
                return True
        
        return False


# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def validate_stage5_output(stage5_data: List[Dict]) -> List[Dict]:
    """
    Main validation pipeline for Stage 6
    Uses 3-tier system: Alif → Groq → Rule-based
    """
    validator = AlifValidator()
    results = []
    
    total = len(stage5_data)
    print(f"[Stage 6] Processing {total} items from Stage 5\n")
    
    for idx, item in enumerate(stage5_data, 1):
        print(f"Processing {idx}/{total}...", end=" ")
        
        corrected_text = item.get("corrected_text", "")
        original_text = item.get("original_text", "")
        hallucination_type = item.get("hallucination_type", "unknown")
        stage5_explanation = item.get("explanation", "")
        
        if not corrected_text:
            print("Empty text, skipping")
            continue
        
        # Run validation checks
        naturalness = validator.score_naturalness(corrected_text, original_text)
        code_mixing = validator.validate_code_mixing_consistency(corrected_text)
        cultural = validator.check_cultural_appropriateness(corrected_text, original_text)
        
        # Calculate overall score
        overall_score = (
            naturalness["score"] * 0.4 +
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
        
        if not all_passed and validator.validation_mode in ["alif", "groq"]:
            failed_checks = []
            if not naturalness["passed"]:
                failed_checks.append("naturalness")
            if not code_mixing["passed"]:
                failed_checks.append("code-mixing consistency")
            if not cultural["passed"]:
                failed_checks.append("cultural appropriateness")
            
            refined = validator.suggest_refinement(corrected_text, failed_checks)
            
            if refined and refined != corrected_text and len(refined) > 10:
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
                "validation_mode": validator.validation_mode,
                "hallucination_type": hallucination_type,
                "stage5_explanation": stage5_explanation
            }
        }
        
        results.append(result)
        
        status = "✓" if all_passed else ("🔧" if refinement_applied else "⚠️")
        print(f"{status} Score: {overall_score:.3f} ({validator.validation_mode})")
    
    return results


def generate_validation_summary(results: List[Dict]) -> Dict:
    """Generate summary statistics"""
    total = len(results)
    if total == 0:
        return {}
    
    passed_all = sum(1 for r in results if r["validation"]["all_checks_passed"])
    refined = sum(1 for r in results if r["metadata"]["refinement_applied"])
    
    avg_scores = {
        "overall": sum(r["validation"]["overall_score"] for r in results) / total,
        "naturalness": sum(r["validation"]["naturalness"]["score"] for r in results) / total,
        "code_mixing": sum(r["validation"]["code_mixing_consistency"]["score"] for r in results) / total,
        "cultural": sum(r["validation"]["cultural_appropriateness"]["score"] for r in results) / total
    }
    
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
    print("STAGE 6: VALIDATION & ROMAN URDU CONVERSION")
    print("="*60)
    print()
    
    if not STAGE5_OUTPUT.exists():
        print(f"Error: {STAGE5_OUTPUT} not found")
        print("Please run Stage 5 first")
        exit(1)
    
    with open(STAGE5_OUTPUT, "r", encoding="utf-8") as f:
        stage5_data = json.load(f)
    
    results = validate_stage5_output(stage5_data)
    summary = generate_validation_summary(results)
    
    output_data = {
        "results": results,
        "summary": summary
    }
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
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
    print(f"Stage 6 complete! Output saved to {OUTPUT_FILE}")