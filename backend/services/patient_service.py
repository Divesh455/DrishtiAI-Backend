import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / "backend" / ".env")


def fetch_patient_profile(user_id: str) -> dict:
    """
    Fetches patient profile details (Name, Age, Gender, MRN, etc.)
    from an external patient database/API if configured.

    Environment Variable:
        EXTERNAL_PATIENT_API_URL: e.g. "https://api.example.com/patients/{user_id}"
                                 or "https://api.example.com/patients/"

    Returns:
        dict: Patient demographic information dictionary.
    """

    default_info = {
        "user_id": user_id,
        "name": "N/A",
        "age": "N/A",
        "gender": "N/A",
        "phone": "N/A",
        "mrn": "N/A",
        "external_fetch_status": "not_configured",
    }

    if not user_id:
        return default_info

    api_url_template = os.getenv("EXTERNAL_PATIENT_API_URL")

    if not api_url_template:
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

    # Add optional API key if configured
    api_key = os.getenv("EXTERNAL_PATIENT_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key

    try:
        req = urllib.request.Request(target_url, headers=headers, method="GET")

        with urllib.request.urlopen(req, timeout=3.0) as response:
            if response.status == 200:
                body = response.read().decode("utf-8")
                data = json.loads(body)

                # Support nested data key if present (e.g. { data: { name: ... } })
                if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
                    data = data["data"]

                if isinstance(data, dict):
                    return {
                        "user_id": user_id,
                        "name": str(data.get("name") or data.get("patient_name") or data.get("full_name") or "N/A"),
                        "age": str(data.get("age") or data.get("patient_age") or "N/A"),
                        "gender": str(data.get("gender") or data.get("sex") or "N/A").title(),
                        "phone": str(data.get("phone") or data.get("contact") or "N/A"),
                        "mrn": str(data.get("mrn") or data.get("hospital_mrn") or data.get("patient_id") or "N/A"),
                        "external_fetch_status": "success",
                    }

    except urllib.error.HTTPError as exc:
        print(f"[PatientService] External API returned HTTP {exc.code} for user_id {user_id}")
        default_info["external_fetch_status"] = f"http_error_{exc.code}"
    except urllib.error.URLError as exc:
        print(f"[PatientService] URL error connecting to external Patient API: {exc.reason}")
        default_info["external_fetch_status"] = "connection_error"
    except Exception as exc:
        print(f"[PatientService] Unexpected error fetching patient profile: {exc}")
        default_info["external_fetch_status"] = "failed"

    return default_info
