"""
i18n_service.py
===============
Centralized Internationalization & Localization Service for DrishtiAI / e-Netra.
Supports English, Hindi (हिंदी), and Marathi (मराठी).
"""

from typing import Any, Dict, Optional


SUPPORTED_LANGUAGES = ["English", "Hindi", "Marathi"]


def normalize_language(lang_input: Optional[str]) -> str:
    """
    Normalizes any language string or code (e.g., 'hi', 'mr', 'hindi', 'marathi', 'en')
    to 'English', 'Hindi', or 'Marathi'. Defaults to 'English'.
    """
    if not lang_input:
        return "English"

    cleaned = str(lang_input).strip().lower()

    if cleaned in ["hi", "hin", "hindi", "हिंदी"]:
        return "Hindi"
    elif cleaned in ["mr", "mar", "marathi", "मराठी"]:
        return "Marathi"
    elif cleaned in ["en", "eng", "english", "इंग्रजी", "अंग्रेजी"]:
        return "English"

    # Capitalized match fallback
    for supported in SUPPORTED_LANGUAGES:
        if cleaned == supported.lower():
            return supported

    return "English"


# ---------------------------------------------------------------------------
# DR Labels & Interpretations
# ---------------------------------------------------------------------------
DR_LABELS = {
    0: {
        "English": "No DR",
        "Hindi": "कोई डी.आर. नहीं (सामान्य)",
        "Marathi": "डी.आर. नाही (सामान्य)",
    },
    1: {
        "English": "Mild DR",
        "Hindi": "हल्की डी.आर. (माइल्ड)",
        "Marathi": "सौम्य डी.आर.",
    },
    2: {
        "English": "Moderate DR",
        "Hindi": "मध्यम डी.आर. (मॉडरेट)",
        "Marathi": "मध्यम डी.आर.",
    },
    3: {
        "English": "Severe DR",
        "Hindi": "गंभीर डी.आर. (सीवियर)",
        "Marathi": "तीव्र डी.आर.",
    },
    4: {
        "English": "Proliferative DR",
        "Hindi": "प्रोलिफेरेटिव डी.आर. (अति गंभीर)",
        "Marathi": "प्रोलिफेरेटिव्ह डी.आर. (अति तीव्र)",
    },
}

DR_INTERPRETATIONS = {
    0: {
        "English": "No diabetic retinopathy was classified by the primary e-Netra model.",
        "Hindi": "प्राथमिक ई-नेत्रा मॉडल द्वारा रेटिना में किसी डायबिटिक रेटिनोपैथी के लक्षण नहीं पाए गए।",
        "Marathi": "प्राथमिक ई-नेत्रा मॉडेलद्वारे रेटिनामध्ये डायबिटिक रेटिनोपॅथीचे कोणतेही लक्षण आढळलेले नाही.",
    },
    1: {
        "English": "Mild diabetic retinopathy was classified by the primary e-Netra model.",
        "Hindi": "प्राथमिक ई-नेत्रा मॉडल द्वारा प्रारंभिक (हल्की) डायबिटिक रेटिनोपैथी का पता चला है।",
        "Marathi": "प्राथमिक ई-नेत्रा मॉडेलद्वारे सौम्य डायबिटिक रेटिनोपॅथीचे निदान झाले आहे.",
    },
    2: {
        "English": "Moderate diabetic retinopathy was classified by the primary e-Netra model.",
        "Hindi": "प्राथमिक ई-नेत्रा मॉडल द्वारा मध्यम स्तर की डायबिटिक रेटिनोपैथी का पता चला है।",
        "Marathi": "प्राथमिक ई-नेत्रा मॉडेलद्वारे मध्यम स्वरूपाच्या डायबिटिक रेटिनोपॅथीचे निदान झाले आहे.",
    },
    3: {
        "English": "Severe diabetic retinopathy was classified by the primary e-Netra model.",
        "Hindi": "प्राथमिक ई-नेत्रा मॉडल द्वारा गंभीर डायबिटिक रेटिनोपैथी का पता चला है। तत्काल विशेषज्ञ परामर्श आवश्यक है।",
        "Marathi": "प्राथमिक ई-नेत्रा मॉडेलद्वारे तीव्र डायबिटिक रेटिनोपॅथीचे निदान झाले आहे. त्वरित तज्ञ सल्ला आवश्यक आहे.",
    },
    4: {
        "English": "Proliferative diabetic retinopathy was classified by the primary e-Netra model.",
        "Hindi": "प्राथमिक ई-नेत्रा मॉडल द्वारा अति गंभीर प्रोलिफेरेटिव डायबिटिक रेटिनोपैथी का पता चला है। तत्काल आपातकालीन चिकित्सा आवश्यक है।",
        "Marathi": "प्राथमिक ई-नेत्रा मॉडेलद्वारे अति तीव्र प्रोलिफेरेटिव्ह डायबिटिक रेटिनोपॅथी आढळली आहे. तातडीने वैद्यकीय उपचार आवश्यक आहेत.",
    },
}

# ---------------------------------------------------------------------------
# Referral Decisions & Advice
# ---------------------------------------------------------------------------
REFERRAL_TRANSLATIONS = {
    "REFER": {
        "label": {
            "English": "REFER TO SPECIALIST",
            "Hindi": "नेत्र विशेषज्ञ के पास भेजें (REFER)",
            "Marathi": "नेत्ररोग तज्ञांकडे पाठवा (REFER)",
        },
        "reason": {
            "English": "e-Netra primary grade is >= 2, which requires formal ophthalmologist examination and management.",
            "Hindi": "डायबिटिक रेटिनोपैथी ग्रेड 2 या उससे अधिक पाई गई है, जिसके लिए नेत्र विशेषज्ञ द्वारा विस्तृत जांच और उपचार अनिवार्य है।",
            "Marathi": "डायबिटिक रेटिनोपॅथी ग्रेड 2 किंवा त्याहून अधिक आढळली आहे, ज्यामुळे नेत्ररोग तज्ञांकडून सखोल तपासणी आणि उपचार आवश्यक आहेत.",
        },
    },
    "NON-REFERABLE": {
        "label": {
            "English": "NON-REFERABLE (ROUTINE MONITORING)",
            "Hindi": "रेफरल आवश्यक नहीं (नियमित निगरानी)",
            "Marathi": "रेफरलची आवश्यकता नाही (नियमित देखरेख)",
        },
        "reason": {
            "English": "No referable diabetic retinopathy detected. Routine annual eye checkup is advised.",
            "Hindi": "किसी गंभीर डायबिटिक रेटिनोपैथी का पता नहीं चला। नियमित वार्षिक नेत्र परीक्षण जारी रखें।",
            "Marathi": "कोणत्याही गंभीर डायबिटिक रेटिनोपॅथीचा शोध लागलेला नाही. नियमित वार्षिक डोळ्यांची तपासणी करण्याचा सल्ला दिला जातो.",
        },
    },
}

# ---------------------------------------------------------------------------
# Quality Status
# ---------------------------------------------------------------------------
QUALITY_STATUS_TRANSLATIONS = {
    "good": {
        "English": "Good",
        "Hindi": "उत्कृष्ट (Good)",
        "Marathi": "उत्कृष्ट (Good)",
    },
    "adequate": {
        "English": "Adequate",
        "Hindi": "संतोषजनक (Adequate)",
        "Marathi": "समाधानकारक (Adequate)",
    },
    "poor": {
        "English": "Poor",
        "Hindi": "खराब (Poor)",
        "Marathi": "असमाधानकारक (Poor)",
    },
    "rejected": {
        "English": "Rejected",
        "Hindi": "अस्वीकृत (Rejected)",
        "Marathi": "नाकारले (Rejected)",
    },
}

# ---------------------------------------------------------------------------
# Lesion Types
# ---------------------------------------------------------------------------
LESION_TRANSLATIONS = {
    "microaneurysms": {
        "English": "Microaneurysms",
        "Hindi": "माइक्रोएन्यूरिज्म",
        "Marathi": "मायक्रोएन्युरिझम",
    },
    "hemorrhages": {
        "English": "Hemorrhages",
        "Hindi": "रक्तस्राव",
        "Marathi": "रक्तस्राव",
    },
    "exudates": {
        "English": "Hard Exudates",
        "Hindi": "हार्ड एक्सुडेट्स (वसा रिसाव)",
        "Marathi": "हार्ड एक्स्युडेट्स (चरबी स्त्राव)",
    },
    "cotton_wool_spots": {
        "English": "Cotton Wool Spots",
        "Hindi": "कॉटन वूल स्पॉट्स",
        "Marathi": "कॉटन वूल स्पॉट्स",
    },
}

# ---------------------------------------------------------------------------
# WebSocket Progress Messages
# ---------------------------------------------------------------------------
PROGRESS_MESSAGES = {
    1: {
        "English": "Checking image quality & fundus suitability...",
        "Hindi": "नेत्र छवि की गुणवत्ता और फंडस उपयुक्तता जांची जा रही है...",
        "Marathi": "प्रतिमेची गुणवत्ता आणि फंडस उपयुक्तता तपासत आहे...",
    },
    2: {
        "English": "Running EfficientNet DR Model & Senanur Safety Check...",
        "Hindi": "EfficientNet डी.आर. मॉडल और सुरक्षा परीक्षण किया जा रहा है...",
        "Marathi": "EfficientNet डी.आर. मॉडेल आणि सुरक्षा तपासणी सुरू आहे...",
    },
    3: {
        "English": "Generating Grad-CAM++ Heatmaps & Explainability Maps...",
        "Hindi": "Grad-CAM++ हीटमैप और व्याख्यात्मक दृश्य तैयार किए जा रहे हैं...",
        "Marathi": "Grad-CAM++ हीटमॅप आणि स्पष्टीकरण नकाशे तयार करत आहे...",
    },
    4: {
        "English": "Detecting Retinal Lesions (Microaneurysms, Haemorrhages, Exudates)...",
        "Hindi": "रेटिनल घावों (माइक्रोएन्यूरिज्म, रक्तस्राव, एक्सुडेट्स) की पहचान की जा रही है...",
        "Marathi": "रेटिनल जखमांचा (मायक्रोएन्युरिझम, रक्तस्राव, एक्स्युडेट्स) शोध घेत आहे...",
    },
    5: {
        "English": "Building Clinical Evidence & Referral Recommendation...",
        "Hindi": "नैदानिक साक्ष्य और रेफरल सिफारिश तैयार की जा रही है...",
        "Marathi": "क्लिनिकल पुरावे आणि रेफरल शिफारस तयार करत आहे...",
    },
    100: {
        "English": "Screening Complete. Results saved successfully.",
        "Hindi": "स्क्रीनिंग पूर्ण हुई। परिणाम सफलतापूर्वक सहेज लिए गए हैं।",
        "Marathi": "स्क्रीनिंग पूर्ण झाले. निकाल यशस्वीरित्या जतन केले गेले आहेत.",
    },
}

# ---------------------------------------------------------------------------
# PDF Document UI Strings
# ---------------------------------------------------------------------------
PDF_STRINGS = {
    "English": {
        "title": "e-Netra Screening Report",
        "subtitle": "Explainable AI-Based Diabetic Retinopathy Screening Report",
        "patient_demographics": "Patient Demographics & Overview",
        "patient_name": "Patient Name",
        "patient_id": "Patient / User ID",
        "age_gender": "Age / Gender",
        "hospital_mrn": "Hospital MRN",
        "referral_decision": "Referral Decision",
        "screening_status": "Screening Status",
        "dr_classification": "Diabetic Retinopathy Classification",
        "primary_grade": "Primary DR Grade",
        "class_label": "Class Label",
        "confidence_score": "Confidence Score",
        "quality_assessment": "Image Quality Assessment",
        "technical_score": "Technical Score",
        "quality_status": "Quality Status",
        "fundus_suitability": "Fundus Suitability AI",
        "lesion_summary": "Lesion Detection Summary",
        "lesion_type": "Lesion Type",
        "detected_count": "Detected Count",
        "total_lesions": "Total Lesions Detected",
        "ai_narrative_title": "AI Clinical & Patient Interpretation",
        "ophthalmologist_impression": "Ophthalmologist Impression",
        "patient_guidance": "Patient Guidance",
        "visual_evidence": "Visual Evidence & Explainability Links",
        "gradcam_link": "Grad-CAM++ Attention Overlay",
        "lesion_link": "Lesion Detection Overlay",
        "disclaimer_title": "Clinical Disclaimer",
        "disclaimer_text": (
            "e-Netra is an artificial intelligence decision-support tool. "
            "Grad-CAM++ heatmaps indicate model feature attention and should be interpreted alongside clinical assessment "
            "by a certified ophthalmologist. Final diagnosis and treatment decisions remain the responsibility of the clinician."
        ),
    },
    "Hindi": {
        "title": "ई-नेत्रा स्क्रीनिंग रिपोर्ट",
        "subtitle": "व्याख्यात्मक एआई-आधारित डायबिटिक रेटिनोपैथी स्क्रीनिंग रिपोर्ट",
        "patient_demographics": "रोगी विवरण और अवलोकन",
        "patient_name": "रोगी का नाम",
        "patient_id": "रोगी / यूजर आईडी",
        "age_gender": "आयु / लिंग",
        "hospital_mrn": "अस्पताल एम.आर.एन.",
        "referral_decision": "रेफरल निर्णय",
        "screening_status": "स्क्रीनिंग स्थिति",
        "dr_classification": "डायबिटिक रेटिनोपैथी वर्गीकरण",
        "primary_grade": "प्राथमिक डी.आर. ग्रेड",
        "class_label": "श्रेणी लेबल",
        "confidence_score": "आत्मविश्वास स्कोर",
        "quality_assessment": "छवि गुणवत्ता मूल्यांकन",
        "technical_score": "तकनीकी स्कोर",
        "quality_status": "गुणवत्ता स्थिति",
        "fundus_suitability": "फंडस उपयुक्तता एआई",
        "lesion_summary": "रेटिनल घाव पहचान सारांश",
        "lesion_type": "घाव का प्रकार",
        "detected_count": "पहचाने गए घाव",
        "total_lesions": "कुल पहचाने गए घाव",
        "ai_narrative_title": "एआई नैदानिक और रोगी व्याख्या",
        "ophthalmologist_impression": "नेत्र रोग विशेषज्ञ निष्कर्ष",
        "patient_guidance": "रोगी के लिए मार्गदर्शन",
        "visual_evidence": "दृश्य साक्ष्य और व्याख्या लिंक",
        "gradcam_link": "Grad-CAM++ मॉडल ध्यान ओवरले",
        "lesion_link": "रेटिनल घाव पहचान ओवरले",
        "disclaimer_title": "चिकित्सीय अस्वीकरण",
        "disclaimer_text": (
            "ई-नेत्रा एक कृत्रिम बुद्धिमत्ता (एआई) आधारित निर्णय-सहायक प्रणाली है। "
            "Grad-CAM++ हीटमैप मॉडल के ध्यान को दर्शाते हैं और इनका मूल्यांकन प्रमाणित नेत्र विशेषज्ञ द्वारा किया जाना चाहिए। "
            "अंतिम निदान और उपचार का निर्णय चिकित्सक की जिम्मेदारी है।"
        ),
    },
    "Marathi": {
        "title": "ई-नेत्रा तपासणी अहवाल",
        "subtitle": "स्पष्टीकरणात्मक एआय-आधारित डायबिटिक रेटिनोपॅथी तपासणी अहवाल",
        "patient_demographics": "रुग्णाची माहिती आणि आढावा",
        "patient_name": "रुग्णाचे नाव",
        "patient_id": "रुग्ण / युझर आयडी",
        "age_gender": "वय / लिंग",
        "hospital_mrn": "हॉस्पिटल एम.आर.एन.",
        "referral_decision": "रेफरल निर्णय",
        "screening_status": "तपासणी स्थिती",
        "dr_classification": "डायबिटिक रेटिनोपॅथी वर्गीकरण",
        "primary_grade": "प्राथमिक डी.आर. ग्रेड",
        "class_label": "वर्ग लेबल",
        "confidence_score": "विश्वासार्हता स्कोअर",
        "quality_assessment": "प्रतिमा गुणवत्ता मूल्यमापन",
        "technical_score": "तांत्रिक स्कोअर",
        "quality_status": "गुणवत्ता स्थिती",
        "fundus_suitability": "फंडस उपयुक्तता एआय",
        "lesion_summary": "रेटिनल जखमांचा सारांश",
        "lesion_type": "जखमेचा प्रकार",
        "detected_count": "आढळलेली संख्या",
        "total_lesions": "एकूण आढळलेल्या जखमा",
        "ai_narrative_title": "एआय क्लिनिकल आणि रुग्ण मार्गदर्शन",
        "ophthalmologist_impression": "नेत्रतज्ञांचे निरीक्षण",
        "patient_guidance": "रुग्णासाठी मार्गदर्शन",
        "visual_evidence": "दृश्य पुरावे आणि लिंक्स",
        "gradcam_link": "Grad-CAM++ मॉडेल लक्ष ओव्हरले",
        "lesion_link": "जखमा शोध ओव्हरले",
        "disclaimer_title": "वैद्यकीय अस्वीकरण",
        "disclaimer_text": (
            "ई-नेत्रा हे कृत्रिम बुद्धिमत्ता (AI) आधारित निर्णय-समर्थन साधन आहे. "
            "Grad-CAM++ हीटमॅप्स मॉडेलचे लक्ष दर्शवतात आणि प्रमाणित नेत्ररोग तज्ञांच्या तपासणीसह समजून घेतले पाहिजेत. "
            "अंतिम निदान आणि उपचारांचे निर्णय संबंधित डॉक्टरांची जबाबदारी आहे."
        ),
    },
}


# ---------------------------------------------------------------------------
# Fallback AI Narratives (Deterministic, reliable clinical text)
# ---------------------------------------------------------------------------
def generate_fallback_narrative(
    screening_data: dict,
    patient_info: Optional[dict] = None,
    language: str = "English",
) -> dict:
    """
    Generates high-quality clinical and patient explanations deterministically
    when Gemini API is offline or returns an error.
    """
    lang = normalize_language(language)
    dr_grade = screening_data.get("dr_grade", 0)
    if dr_grade is None:
        dr_grade = 0
    else:
        try:
            dr_grade = int(dr_grade)
        except Exception:
            dr_grade = 0

    p_info = patient_info or {}
    p_name = p_info.get("name") or "Patient"

    dr_label = DR_LABELS.get(dr_grade, {}).get(lang, "No DR")
    lesion_count = screening_data.get("lesion_count", 0)

    if lang == "Hindi":
        if dr_grade == 0:
            clinical = (
                f"रोगी {p_name} की फंडस छवि में रेटिनोपैथी के कोई सक्रिय संकेत नहीं पाए गए (ग्रेड 0 - {dr_label})। "
                f"कुल पहचाने गए घाव: {lesion_count}। ऑप्टिक डिस्क और मैक्युला सामान्य दिखाई देते हैं। "
                f"किसी तत्काल चिकित्सीय हस्तक्षेप की आवश्यकता नहीं है।"
            )
            patient = (
                f"नमस्ते {p_name}, आपकी आंखों की जांच के अनुसार वर्तमान में डायबिटिक रेटिनोपैथी का कोई लक्षण नहीं है। "
                f"आपकी आंखें स्वस्थ हैं। कृपया अपना ब्लड शुगर नियंत्रित रखें और वर्ष में एक बार नियमित नेत्र परीक्षण कराएं।"
            )
        elif dr_grade == 1:
            clinical = (
                f"रोगी {p_name} में प्रारंभिक स्तर की डायबिटिक रेटिनोपैथी (ग्रेड 1 - {dr_label}) के सूक्ष्म लक्षण मिले हैं। "
                f"रेटिना में {lesion_count} सूक्ष्म घाव दर्ज किए गए। निकट भविष्य में दृष्टि संबंधी जोखिम न्यूनतम है।"
            )
            patient = (
                f"नमस्ते {p_name}, आपकी आंखों में शुगर के कारण प्रारंभिक बदलाव देखे गए हैं। घबराने की कोई बात नहीं है, "
                f"परंतु ब्लड शुगर और रक्तचाप को नियंत्रित रखना आवश्यक है। 6 महीने में पुनः जांच अवश्य कराएं।"
            )
        elif dr_grade == 2:
            clinical = (
                f"रोगी {p_name} में मध्यम डायबिटिक रेटिनोपैथी (ग्रेड 2 - {dr_label}) की पुष्टि हुई है। "
                f"रेटिना में {lesion_count} घाव पाए गए हैं। रेटिना में रक्तस्राव और एक्सुडेट्स की उपस्थिति के कारण नेत्र विशेषज्ञ को रेफर किया जाना चाहिए।"
            )
            patient = (
                f"नमस्ते {p_name}, आपकी आंखों में मध्यम स्तर के डायबिटिक रेटिनोपैथी के लक्षण पाए गए हैं। "
                f"दृष्टि को सुरक्षित रखने के लिए जल्द से जल्द नेत्र विशेषज्ञ (आंखों के डॉक्टर) से परामर्श लें।"
            )
        elif dr_grade >= 3:
            clinical = (
                f"रोगी {p_name} में गंभीर/प्रोलिफेरेटिव डायबिटिक रेटिनोपैथी (ग्रेड {dr_grade} - {dr_label}) दर्ज की गई है। "
                f"कुल {lesion_count} महत्वपूर्ण रेटिनल घाव उपस्थित हैं। दृष्टि हानि की रोकथाम हेतु तत्काल रेटिना विशेषज्ञ मूल्यांकन अनिवार्य है।"
            )
            patient = (
                f"नमस्ते {p_name}, आपकी आंखों में गंभीर डायबिटिक रेटिनोपैथी के लक्षण हैं। "
                f"कृपया बिना किसी देरी के तुरंत किसी वरिष्ठ नेत्र रोग विशेषज्ञ से मिलें ताकि दृष्टि को सुरक्षित रखा जा सके।"
            )

    elif lang == "Marathi":
        if dr_grade == 0:
            clinical = (
                f"रुग्ण {p_name} यांच्या फंडस प्रतिमेमध्ये रेटिनोपॅथीची कोणतीही चिन्हे आढळली नाहीत (ग्रेड 0 - {dr_label}). "
                f"एकूण आढळलेल्या जखमा: {lesion_count}. ऑप्टिक डिस्क आणि मॅक्युला सामान्य आहेत. "
                f"सध्या कोणत्याही तात्काळ उपचारांची आवश्यकता नाही."
            )
            patient = (
                f"नमस्कार {p_name}, आपल्या डोळ्यांच्या तपासणीनुसार डायबिटिक रेटिनोपॅथीचे कोणतेही लक्षण नाही. "
                f"आपले डोळे सुरक्षित आहेत. रक्तातील साखर नियंत्रणात ठेवा आणि वर्षातून एकदा नियमित डोळे तपासून घ्या."
            )
        elif dr_grade == 1:
            clinical = (
                f"रुग्ण {p_name} यांच्यामध्ये प्राथमिक सौम्य डायबिटिक रेटिनोपॅथी (ग्रेड 1 - {dr_label}) आढळली आहे. "
                f"रेटिनामध्ये {lesion_count} सूक्ष्म जखमांची नोंद झाली आहे. दृष्टीला सध्या धोका नाही."
            )
            patient = (
                f"नमस्कार {p_name}, आपल्या डोळ्यांमध्ये मधुमेहामुळे सौम्य बदल दिसून आले आहेत. "
                f"काळजीचे कारण नाही, मात्र रक्तातील साखर नियंत्रणात ठेवणे आवश्यक आहे. पुढील 6 महिन्यांत पुन्हा तपासणी करून घ्या."
            )
        elif dr_grade == 2:
            clinical = (
                f"रुग्ण {p_name} यांच्यामध्ये मध्यम डायबिटिक रेटिनोपॅथी (ग्रेड 2 - {dr_label}) चे निदान झाले आहे. "
                f"एकूण {lesion_count} जखमा आढळल्या आहेत. नेत्रतज्ञांचा तातडीने सल्ला घेणे आवश्यक आहे."
            )
            patient = (
                f"नमस्कार {p_name}, आपल्या डोळ्यांमध्ये मध्यम स्वरूपाची डायबिटिक रेटिनोपॅथी आढळली आहे. "
                f"दृष्टीचे रक्षण करण्यासाठी कृपया लवकरात लवकर डोळ्यांच्या डॉक्टरांचा (नेत्रतज्ञ) सल्ला घ्या."
            )
        elif dr_grade >= 3:
            clinical = (
                f"रुग्ण {p_name} यांच्यामध्ये तीव्र डायबिटिक रेटिनोपॅथी (ग्रेड {dr_grade} - {dr_label}) नोंदवली गेली आहे. "
                f"रेटिनामध्ये {lesion_count} गंभीर जखमा आढळल्या आहेत. दृष्टी वाचवण्यासाठी तातडीने रेटिना तज्ञांचे उपचार आवश्यक आहेत."
            )
            patient = (
                f"नमस्कार {p_name}, आपल्या डोळ्यांमध्ये गंभीर डायबिटिक रेटिनोपॅथीची लक्षणे आढळली आहेत. "
                f"कृपया कोणताही उशीर न करता त्वरित डोळ्यांच्या वरिष्ठ तज्ञांना भेटा."
            )

    else:  # English
        if dr_grade == 0:
            clinical = (
                f"Patient {p_name} exhibits no evident diabetic retinopathy on fundus evaluation (Grade 0 - {dr_label}). "
                f"Detected retinal lesions: {lesion_count}. Macula and optic disc appear healthy. "
                f"No active ophthalmic intervention required."
            )
            patient = (
                f"Hello {p_name}, your eye screening shows no signs of diabetic retinopathy. "
                f"Your retinas are healthy. Please maintain routine blood glucose control and attend an annual eye screening."
            )
        elif dr_grade == 1:
            clinical = (
                f"Patient {p_name} presents with mild non-proliferative diabetic retinopathy (Grade 1 - {dr_label}). "
                f"A minimal lesion count ({lesion_count}) was identified. Risk of immediate visual impairment is low."
            )
            patient = (
                f"Hello {p_name}, very early diabetes-related changes were detected in your eyes. "
                f"There is no immediate danger, but keeping your blood sugar in target range is essential. Schedule a re-check in 6 months."
            )
        elif dr_grade == 2:
            clinical = (
                f"Patient {p_name} is classified with moderate diabetic retinopathy (Grade 2 - {dr_label}). "
                f"{lesion_count} lesions were detected. An in-person ophthalmology consultation is advised."
            )
            patient = (
                f"Hello {p_name}, moderate signs of diabetic retinopathy were found in your retina. "
                f"Please consult an eye specialist soon to discuss preventative care and protect your vision."
            )
        elif dr_grade >= 3:
            clinical = (
                f"Patient {p_name} demonstrates severe/proliferative diabetic retinopathy (Grade {dr_grade} - {dr_label}). "
                f"{lesion_count} significant lesions are present. Prompt specialist vitreoretinal evaluation is strongly recommended."
            )
            patient = (
                f"Hello {p_name}, serious diabetic retinopathy signs were found in your eyes. "
                f"Please seek urgent examination by an ophthalmologist or retina specialist to safeguard your sight."
            )

    return {
        "generated": True,
        "model": "eNetra-Clinical-Template-Engine",
        "language": lang,
        "cached": False,
        "source": "clinical_fallback",
        "message": f"Deterministic clinical narrative generated in {lang}.",
        "clinical_narrative": clinical,
        "patient_summary": patient,
    }


def localize_screening_response(result: dict, language: str = "English") -> dict:
    """
    Enhances a screening result dictionary with localized labels,
    decision texts, interpretations, and explanations in the target language.
    """
    lang = normalize_language(language)

    # Clone and enrich
    enriched = dict(result)

    # DR classification localization
    classification = enriched.get("classification") or {}
    dr_grade = classification.get("grade")
    if dr_grade is not None:
        try:
            dr_grade_int = int(dr_grade)
            localized_label = DR_LABELS.get(dr_grade_int, {}).get(lang, classification.get("label"))
            localized_interpretation = DR_INTERPRETATIONS.get(dr_grade_int, {}).get(lang, "")
        except Exception:
            localized_label = classification.get("label")
            localized_interpretation = ""
    else:
        localized_label = classification.get("label")
        localized_interpretation = ""

    # Decision localization
    decision = enriched.get("screening_decision") or {}
    dec_val = decision.get("decision", "NON-REFERABLE")
    dec_trans = REFERRAL_TRANSLATIONS.get(dec_val, REFERRAL_TRANSLATIONS.get("NON-REFERABLE"))
    localized_decision_label = dec_trans["label"].get(lang, dec_val)
    localized_decision_reason = dec_trans["reason"].get(lang, decision.get("reason", ""))

    # Quality localization
    quality = enriched.get("quality") or {}
    q_status = str(quality.get("status", "")).lower()
    localized_quality = QUALITY_STATUS_TRANSLATIONS.get(q_status, {}).get(lang, q_status.title())

    # Explainability localization
    explainability = enriched.get("explainability") or {}
    exp_text = {
        "English": "Grad-CAM++ highlights image regions that contributed to the selected DR classification.",
        "Hindi": "Grad-CAM++ रेटिना छवि के उन प्रमुख क्षेत्रों को दर्शाता है जिन्होंने एआई मॉडल के निर्णय को प्रभावित किया।",
        "Marathi": "Grad-CAM++ प्रतिमेतील त्या भागांना ठळक करते ज्यांनी मॉडेलच्या डी.आर. वर्गीकरणात हातभार लावला.",
    }.get(lang, explainability.get("explanation", ""))

    # Attach localized bundle
    enriched["localized"] = {
        "language": lang,
        "dr_label": localized_label,
        "interpretation": localized_interpretation,
        "referral_decision": localized_decision_label,
        "referral_reason": localized_decision_reason,
        "quality_status": localized_quality,
        "explainability_note": exp_text,
    }

    return enriched
