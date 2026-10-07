import time
import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / "backend" / ".env")


class PatientServiceCircuitBreaker:
    """
    Implements a Circuit Breaker pattern with in-memory caching and prefetching
    for external patient profile API calls.

    States:
    - CLOSED: Normal operation. Requests pass through to external API.
    - OPEN: Circuit tripped due to consecutive failures. Requests immediately fail-fast to cache/default.
    - HALF_OPEN: Cooldown expired. Next request tests if external API has recovered.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        request_timeout: float = 2.5,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.request_timeout = request_timeout

        self._state = "CLOSED"
        self._failure_count = 0
        self._last_state_change = 0.0
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get_state(self) -> str:
        now = time.time()
        if self._state == "OPEN" and (now - self._last_state_change) >= self.recovery_timeout:
            self._state = "HALF_OPEN"
            print("[PatientService] Circuit Breaker entering HALF_OPEN probe state.")
        return self._state

    def _record_success(self):
        self._failure_count = 0
        if self._state != "CLOSED":
            print("[PatientService] External API recovered. Circuit Breaker reset to CLOSED.")
            self._state = "CLOSED"
            self._last_state_change = time.time()

    def _record_failure(self, reason: str):
        self._failure_count += 1
        now = time.time()
        print(f"[PatientService] Failure #{self._failure_count} ({reason})")
        if self._failure_count >= self.failure_threshold:
            if self._state != "OPEN":
                print(f"[PatientService] Circuit Breaker TRIPPED to OPEN state for {self.recovery_timeout}s.")
                self._state = "OPEN"
                self._last_state_change = now

    def fetch_patient_profile(self, user_id: str) -> Dict[str, Any]:
        default_info = {
            "user_id": user_id,
            "name": "N/A",
            "age": "N/A",
            "gender": "N/A",
            "email": "N/A",
            "phone": "N/A",
            "mrn": "N/A",
            "external_fetch_status": "not_configured",
        }

        if not user_id:
            return default_info

        # Return cached profile if available
        if user_id in self._cache:
            cached_data = dict(self._cache[user_id])
            cached_data["external_fetch_status"] = "cached"
            return cached_data

        api_url_template = os.getenv("EXTERNAL_PATIENT_API_URL")
        if not api_url_template:
            return default_info

        state = self.get_state()
        if state == "OPEN":
            default_info["external_fetch_status"] = "circuit_breaker_open"
            return default_info

        # Format URL
        if "{user_id}" in api_url_template:
            target_url = api_url_template.format(user_id=user_id)
        else:
            target_url = f"{api_url_template.rstrip('/')}/{user_id}"

        headers = {
            "User-Agent": "DrishtiAI-Backend/1.0",
            "Accept": "application/json",
        }
        api_key = os.getenv("EXTERNAL_PATIENT_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-API-Key"] = api_key

        try:
            req = urllib.request.Request(target_url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                if response.status == 200:
                    body = response.read().decode("utf-8")
                    data = json.loads(body)

                    if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
                        data = data["data"]

                    if isinstance(data, dict):
                        result = {
                            "user_id": user_id,
                            "name": str(data.get("name") or data.get("patient_name") or data.get("full_name") or "N/A"),
                            "age": str(data.get("age") or data.get("patient_age") or "N/A"),
                            "gender": str(data.get("gender") or data.get("sex") or "N/A").title(),
                            "email": str(data.get("email") or data.get("gmail") or data.get("user_email") or "N/A"),
                            "phone": str(data.get("phone") or data.get("contact") or "N/A"),
                            "mrn": str(data.get("mrn") or data.get("hospital_mrn") or data.get("patient_id") or "N/A"),
                            "external_fetch_status": "success",
                        }
                        self._cache[user_id] = result
                        self._record_success()
                        return result

        except urllib.error.HTTPError as exc:
            self._record_failure(f"HTTP {exc.code}")
            default_info["external_fetch_status"] = f"http_error_{exc.code}"
        except urllib.error.URLError as exc:
            self._record_failure(f"URLError: {exc.reason}")
            default_info["external_fetch_status"] = "connection_error"
        except Exception as exc:
            self._record_failure(f"Unexpected: {exc}")
            default_info["external_fetch_status"] = "failed"

        return default_info


# Global Singleton Service Instance
patient_service_instance = PatientServiceCircuitBreaker()


def fetch_patient_profile(user_id: str) -> dict:
    """
    Public entry point for fetching patient profile details.
    Uses circuit breaker and in-memory cache.
    """
    return patient_service_instance.fetch_patient_profile(user_id)


def prefetch_patient_profile(user_id: str) -> None:
    """
    Prefetches patient profile data in the background to warm up cache.
    """
    try:
        patient_service_instance.fetch_patient_profile(user_id)
    except Exception as exc:
        print(f"[PatientService] Background prefetch error for user_id {user_id}: {exc}")
