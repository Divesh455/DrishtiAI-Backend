import json
import os
from pathlib import Path
from dotenv import load_dotenv

from collections import OrderedDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / "backend" / ".env")


class GeminiReportService:
    """
    Service for generating rich clinical narratives and patient summaries
    using the Gemini Flash API (gemini-1.5-flash / gemini-2.5-flash)
    with high-performance LRU response caching.
    """

    def __init__(self, max_cache_size: int = 500):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
        self.max_cache_size = max_cache_size
        self._cache = OrderedDict()
        self._redis_client = None

        # Optional Redis Cache Setup
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis
                self._redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
                print("[GeminiReportService] Redis caching enabled.")
            except Exception as exc:
                print(f"[GeminiReportService] Could not connect to Redis: {exc}. Using in-memory LRU cache.")

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def _get_cache(self, key: str) -> dict | None:
        if not key:
            return None

        # 1. Try Redis
        if self._redis_client:
            try:
                cached_val = self._redis_client.get(f"gemini:{key}")
                if cached_val:
                    data = json.loads(cached_val)
                    data["cached"] = True
                    data["cache_source"] = "redis"
                    return data
            except Exception as exc:
                print(f"[GeminiReportService] Redis read error: {exc}")

        # 2. Try In-Memory LRU
        if key in self._cache:
            self._cache.move_to_end(key)
            data = dict(self._cache[key])
            data["cached"] = True
            data["cache_source"] = "memory_lru"
            return data

        return None

    def _set_cache(self, key: str, value: dict):
        if not key or not value.get("generated"):
            return

        # 1. Save to In-Memory LRU
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)

        # 2. Save to Redis if available (TTL: 24 hours)
        if self._redis_client:
            try:
                self._redis_client.setex(
                    name=f"gemini:{key}",
                    time=86400,
                    value=json.dumps(value),
                )
            except Exception as exc:
                print(f"[GeminiReportService] Redis write error: {exc}")

    def generate_report_narrative(
        self,
        screening_data: dict,
        patient_info: dict = None,
        language: str = "English",
    ) -> dict:
        """
        Calls Gemini Flash model to produce clinical narrative and patient-friendly explanations.
        Uses cached result if available for the given screening_id and language.
        """

        # Check Cache first
        screening_id = screening_data.get("screening_id")
        cache_key = f"{screening_id}:{language}" if screening_id else None
        cached_result = self._get_cache(cache_key)
        if cached_result:
            return cached_result

        if not self.is_available():
            return {
                "generated": False,
                "model": self.model_name,
                "language": language,
                "cached": False,
                "message": (
                    "GEMINI_API_KEY is not set in backend/.env. "
                    "Please add your Gemini API key to enable AI narrative generation."
                ),
                "clinical_narrative": None,
                "patient_summary": None,
            }

        # Extract facts for prompt
        p_info = patient_info or {}
        p_name = p_info.get("name", "N/A")
        p_age = p_info.get("age", "N/A")
        p_gender = p_info.get("gender", "N/A")

        dr_grade = screening_data.get("dr_grade")
        dr_label = screening_data.get("dr_label", "N/A")
        dr_conf = screening_data.get("dr_confidence")
        dr_conf_str = f"{dr_conf * 100:.1f}%" if dr_conf is not None else "N/A"

        q_status = screening_data.get("quality_status", "N/A")
        lesion_count = screening_data.get("lesion_count", 0)

        evidence = screening_data.get("evidence") or {}
        decision_info = evidence.get("screening_decision") or {}
        decision = decision_info.get("decision", "N/A")
        reason = decision_info.get("reason", "N/A")

        lesion_evidence = evidence.get("lesion_evidence") or {}
        ma_count = lesion_evidence.get("microaneurysms", 0)
        he_count = lesion_evidence.get("hemorrhages", 0)
        ex_count = lesion_evidence.get("exudates", lesion_evidence.get("hard_exudates", 0))
        cws_count = lesion_evidence.get("cotton_wool_spots", lesion_evidence.get("soft_exudates", 0))

        prompt = f"""
You are an expert AI clinical assistant for DrishtiAI, an Explainable Diabetic Retinopathy (DR) Screening System.

Analyze the following validated screening data for a patient and generate two distinct report sections in the requested TARGET LANGUAGE ({language}):

--- TARGET LANGUAGE ---
{language}

--- PATIENT DEMOGRAPHICS ---
Name: {p_name}
Age/Gender: {p_age} / {p_gender}

--- SCREENING FINDINGS ---
Primary DR Grade: Grade {dr_grade} ({dr_label})
Primary Model Confidence: {dr_conf_str}
Image Technical Quality: {q_status}
Total Lesions Detected: {lesion_count}
Lesion Breakdown:
  - Microaneurysms: {ma_count}
  - Hemorrhages: {he_count}
  - Exudates: {ex_count}
  - Cotton Wool Spots: {cws_count}

--- REFERRAL DECISION ---
Final Referral Decision: {decision}
Decision Reason: {reason}

--- INSTRUCTIONS ---
Please respond strictly with a valid JSON object containing exactly two keys:
1. "clinical_narrative": A concise, formal paragraph (3-4 sentences) written in {language} tailored for an ophthalmologist describing the clinical severity, lesion distribution, and referral urgency.
2. "patient_summary": A compassionate, clear paragraph (3-4 sentences) written in {language} in simple plain language for the patient explaining what the DR grade means, reassuring them, and advising on next steps.

Example JSON output format:
{{
  "clinical_narrative": "...",
  "patient_summary": "..."
}}
"""

        try:
            # Try google.genai Client SDK first
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                text = response.text
            except Exception:
                # Fallback to google.generativeai legacy SDK
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                model = legacy_genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                text = response.text

            # Clean JSON code blocks if present
            cleaned_text = text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]

            parsed = json.loads(cleaned_text.strip())

            result = {
                "generated": True,
                "model": self.model_name,
                "language": language,
                "cached": False,
                "message": f"Gemini Flash AI narrative successfully generated in {language}.",
                "clinical_narrative": parsed.get("clinical_narrative"),
                "patient_summary": parsed.get("patient_summary"),
            }

            self._set_cache(cache_key, result)
            return result

        except Exception as exc:
            print(f"[GeminiReportService] Exception generating narrative: {exc}")
            return {
                "generated": False,
                "model": self.model_name,
                "message": f"Gemini narrative generation failed: {str(exc)}",
                "clinical_narrative": None,
                "patient_summary": None,
            }
