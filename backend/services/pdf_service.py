import io
import os
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.services.i18n_service import (
    PDF_STRINGS,
    DR_LABELS,
    REFERRAL_TRANSLATIONS,
    QUALITY_STATUS_TRANSLATIONS,
    LESION_TRANSLATIONS,
    normalize_language,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FONT_DIR = BASE_DIR / "backend" / "assets" / "fonts"

DEVANAGARI_FONT_NAME = None


def _get_devanagari_font_name() -> str:
    """
    Registers and returns a TTF font that supports English, Hindi, and Marathi (Devanagari).
    """
    global DEVANAGARI_FONT_NAME
    if DEVANAGARI_FONT_NAME:
        return DEVANAGARI_FONT_NAME

    noto_path = FONT_DIR / "NotoSansDevanagari-Regular.ttf"
    if noto_path.exists():
        try:
            pdfmetrics.registerFont(TTFont("NotoDevanagari", str(noto_path)))
            DEVANAGARI_FONT_NAME = "NotoDevanagari"
            return DEVANAGARI_FONT_NAME
        except Exception as exc:
            print(f"[PDFService] Error registering NotoSansDevanagari font: {exc}")

    win_nirmala = Path("C:/Windows/Fonts/Nirmala.ttf")
    if win_nirmala.exists():
        try:
            pdfmetrics.registerFont(TTFont("NirmalaDevanagari", str(win_nirmala)))
            DEVANAGARI_FONT_NAME = "NirmalaDevanagari"
            return DEVANAGARI_FONT_NAME
        except Exception as exc:
            print(f"[PDFService] Error registering Nirmala font: {exc}")

    DEVANAGARI_FONT_NAME = "Helvetica"
    return DEVANAGARI_FONT_NAME


def generate_screening_pdf(
    screening_data: dict,
    patient_info: dict = None,
    gemini_narrative: dict = None,
    language: str = "English",
) -> bytes:
    """
    Generates a professional clinical PDF report for an e-Netra screening record
    in the requested language (English, Hindi, or Marathi).
    """
    # Determine normalized language
    g_narrative = gemini_narrative or {}
    evidence = screening_data.get("evidence") or {}
    stored_lang = evidence.get("preferred_language") or evidence.get("language")
    resolved_lang = language or g_narrative.get("language") or stored_lang or "English"
    norm_lang = normalize_language(resolved_lang)

    pdf_labels = PDF_STRINGS.get(norm_lang, PDF_STRINGS["English"])

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Font assignment
    dev_font = _get_devanagari_font_name()
    use_dev = norm_lang in ["Hindi", "Marathi"]
    base_font = dev_font if use_dev else "Helvetica"
    bold_font = dev_font if use_dev else "Helvetica-Bold"

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E293B"),
        fontName=bold_font,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#64748B"),
        fontName=base_font,
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        fontName=bold_font,
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        fontName=base_font,
    )

    bold_body_style = ParagraphStyle(
        "ReportBodyBold",
        parent=body_style,
        fontName=bold_font,
    )

    elements = []

    # ---------------------------------------------------------
    # 1. Header Section
    # ---------------------------------------------------------
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    screening_id_val = str(screening_data.get("screening_id", "N/A"))

    header_table_data = [
        [
            Paragraph(f"<b>{pdf_labels['title']}</b>", title_style),
            Paragraph(
                f"<b>Date:</b> {date_str}",
                ParagraphStyle("RightAlign", parent=body_style, alignment=2, fontName=base_font),
            ),
        ],
        [
            Paragraph(pdf_labels["subtitle"], subtitle_style),
            Paragraph(
                f"<b>ID:</b> {screening_id_val}",
                ParagraphStyle("RightAlign2", parent=body_style, alignment=2, fontName=base_font),
            ),
        ],
    ]

    header_table = Table(header_table_data, colWidths=[340, 200])
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )

    elements.append(header_table)
    elements.append(Spacer(1, 8))
    elements.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#0284C7"),
            spaceBefore=4,
            spaceAfter=12,
        )
    )

    # ---------------------------------------------------------
    # 2. Patient Demographics & Status Overview
    # ---------------------------------------------------------
    user_id = screening_data.get("user_id", "N/A")
    status = screening_data.get("screening_status", "completed").upper()

    p_info = patient_info or {}
    p_name = p_info.get("name", "N/A")
    p_age = p_info.get("age", "N/A")
    p_gender = p_info.get("gender", "N/A")
    p_mrn = p_info.get("mrn", "N/A")

    decision_info = evidence.get("screening_decision") or {}
    raw_decision = decision_info.get("decision", "PENDING/UNKNOWN")
    referable = decision_info.get("referable")

    if raw_decision == "REFER" or referable is True:
        dec_color = colors.HexColor("#DC2626")  # Red
        dec_bg = colors.HexColor("#FEF2F2")
        dec_display = REFERRAL_TRANSLATIONS["REFER"]["label"].get(norm_lang, "REFER")
    elif raw_decision == "NON-REFERABLE" or referable is False:
        dec_color = colors.HexColor("#16A34A")  # Green
        dec_bg = colors.HexColor("#F0FDF4")
        dec_display = REFERRAL_TRANSLATIONS["NON-REFERABLE"]["label"].get(norm_lang, "NON-REFERABLE")
    else:
        dec_color = colors.HexColor("#EA580C")  # Orange
        dec_bg = colors.HexColor("#FFF7ED")
        dec_display = raw_decision

    summary_box_data = [
        [
            Paragraph(f"<b>{pdf_labels['patient_name']}:</b>", body_style),
            Paragraph(str(p_name), bold_body_style),
            Paragraph(f"<b>{pdf_labels['patient_id']}:</b>", body_style),
            Paragraph(str(user_id), bold_body_style),
        ],
        [
            Paragraph(f"<b>{pdf_labels['age_gender']}:</b>", body_style),
            Paragraph(f"{p_age} / {p_gender}", body_style),
            Paragraph(f"<b>{pdf_labels['hospital_mrn']}:</b>", body_style),
            Paragraph(str(p_mrn), body_style),
        ],
        [
            Paragraph(f"<b>{pdf_labels['referral_decision']}:</b>", body_style),
            Paragraph(
                f"<font color='{dec_color.hexval()}'><b>{dec_display}</b></font>",
                ParagraphStyle("DecStyle", parent=body_style, fontSize=9.5, fontName=bold_font),
            ),
            Paragraph(f"<b>{pdf_labels['screening_status']}:</b>", body_style),
            Paragraph(f"<b>{status}</b>", bold_body_style),
        ],
    ]

    summary_table = Table(summary_box_data, colWidths=[120, 150, 120, 150])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BACKGROUND", (1, 2), (1, 2), dec_bg),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#F1F5F9")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    elements.append(summary_table)
    elements.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # 2b. AI Narrative Section (Clinical & Patient Guidance)
    # ---------------------------------------------------------
    clinical_text = g_narrative.get("clinical_narrative")
    patient_text = g_narrative.get("patient_summary")

    if clinical_text or patient_text:
        elements.append(Paragraph(pdf_labels["ai_narrative_title"], section_heading))

        ai_box_data = []

        if clinical_text:
            ai_box_data.append([
                Paragraph(f"<b>{pdf_labels['ophthalmologist_impression']}:</b>", body_style),
                Paragraph(
                    clinical_text,
                    ParagraphStyle("AiBody", parent=body_style, fontName=dev_font, fontSize=9, leading=13),
                ),
            ])

        if patient_text:
            ai_box_data.append([
                Paragraph(f"<b>{pdf_labels['patient_guidance']}:</b>", body_style),
                Paragraph(
                    patient_text,
                    ParagraphStyle("AiBody2", parent=body_style, fontName=dev_font, fontSize=9, leading=13),
                ),
            ])

        ai_table = Table(ai_box_data, colWidths=[150, 390])
        ai_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F9FF")),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#BAE6FD")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0F2FE")),
                    ("PADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )

        elements.append(ai_table)
        elements.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # 3. DR Classification Details
    # ---------------------------------------------------------
    elements.append(Paragraph(pdf_labels["dr_classification"], section_heading))

    dr_grade = screening_data.get("dr_grade")
    dr_label_raw = screening_data.get("dr_label", "N/A")
    dr_conf = screening_data.get("dr_confidence")
    dr_conf_str = f"{dr_conf * 100:.1f}%" if dr_conf is not None else "N/A"

    try:
        dr_grade_int = int(dr_grade) if dr_grade is not None else None
        localized_dr_label = DR_LABELS.get(dr_grade_int, {}).get(norm_lang, str(dr_label_raw))
    except Exception:
        localized_dr_label = str(dr_label_raw)

    class_data = [
        [
            Paragraph(f"<b>{pdf_labels['primary_grade']}</b>", bold_body_style),
            Paragraph(f"<b>{pdf_labels['class_label']}</b>", bold_body_style),
            Paragraph(f"<b>{pdf_labels['confidence_score']}</b>", bold_body_style),
        ],
        [
            Paragraph(f"Grade {dr_grade}" if dr_grade is not None else "N/A", body_style),
            Paragraph(str(localized_dr_label), body_style),
            Paragraph(dr_conf_str, body_style),
        ],
    ]

    class_table = Table(class_data, colWidths=[170, 180, 190])
    class_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )

    elements.append(class_table)
    elements.append(Spacer(1, 12))

    # ---------------------------------------------------------
    # 4. Image Quality Assessment
    # ---------------------------------------------------------
    elements.append(Paragraph(pdf_labels["quality_assessment"], section_heading))

    q_score = screening_data.get("quality_score")
    q_score_str = f"{q_score:.2f}" if q_score is not None else "N/A"
    raw_q_status = str(screening_data.get("quality_status", "N/A")).lower()
    localized_q_status = QUALITY_STATUS_TRANSLATIONS.get(raw_q_status, {}).get(norm_lang, raw_q_status.title())

    f_suit = screening_data.get("fundus_suitability")
    if f_suit is True:
        f_suit_str = "Suitable (Fundus)" if norm_lang == "English" else ("उपयुक्त (फंडस)" if norm_lang == "Hindi" else "योग्य (फंडस)")
    elif f_suit is False:
        f_suit_str = "Unsuitable" if norm_lang == "English" else ("अनुपयुक्त" if norm_lang == "Hindi" else "अयोग्य")
    else:
        f_suit_str = "N/A"

    quality_data = [
        [
            Paragraph(f"<b>{pdf_labels['technical_score']}</b>", bold_body_style),
            Paragraph(f"<b>{pdf_labels['quality_status']}</b>", bold_body_style),
            Paragraph(f"<b>{pdf_labels['fundus_suitability']}</b>", bold_body_style),
        ],
        [
            Paragraph(q_score_str, body_style),
            Paragraph(str(localized_q_status), body_style),
            Paragraph(str(f_suit_str), body_style),
        ],
    ]

    quality_table = Table(quality_data, colWidths=[170, 180, 190])
    quality_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )

    elements.append(quality_table)
    elements.append(Spacer(1, 12))

    # ---------------------------------------------------------
    # 5. Lesion Detection Breakdown
    # ---------------------------------------------------------
    elements.append(Paragraph(pdf_labels["lesion_summary"], section_heading))

    lesion_evidence = evidence.get("lesion_evidence") or {}
    total_lesions = screening_data.get("lesion_count", 0)

    ma_count = lesion_evidence.get("microaneurysms", 0)
    he_count = lesion_evidence.get("hemorrhages", 0)
    ex_count = lesion_evidence.get("exudates", lesion_evidence.get("hard_exudates", 0))
    cws_count = lesion_evidence.get("cotton_wool_spots", lesion_evidence.get("soft_exudates", 0))

    lesion_table_data = [
        [
            Paragraph(f"<b>{pdf_labels['lesion_type']}</b>", bold_body_style),
            Paragraph(f"<b>{pdf_labels['detected_count']}</b>", bold_body_style),
        ],
        [
            Paragraph(LESION_TRANSLATIONS["microaneurysms"][norm_lang], body_style),
            Paragraph(str(ma_count), body_style),
        ],
        [
            Paragraph(LESION_TRANSLATIONS["hemorrhages"][norm_lang], body_style),
            Paragraph(str(he_count), body_style),
        ],
        [
            Paragraph(LESION_TRANSLATIONS["exudates"][norm_lang], body_style),
            Paragraph(str(ex_count), body_style),
        ],
        [
            Paragraph(LESION_TRANSLATIONS["cotton_wool_spots"][norm_lang], body_style),
            Paragraph(str(cws_count), body_style),
        ],
        [
            Paragraph(f"<b>{pdf_labels['total_lesions']}</b>", bold_body_style),
            Paragraph(f"<b>{total_lesions}</b>", bold_body_style),
        ],
    ]

    lesion_table = Table(lesion_table_data, colWidths=[340, 200])
    lesion_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ]
        )
    )

    elements.append(lesion_table)
    elements.append(Spacer(1, 12))

    # ---------------------------------------------------------
    # 6. Visual Evidence Links
    # ---------------------------------------------------------
    elements.append(Paragraph(pdf_labels["visual_evidence"], section_heading))

    gradcam_url = screening_data.get("gradcam_url") or "N/A"
    lesion_url = screening_data.get("lesion_overlay_url") or "N/A"

    image_links_data = [
        [
            Paragraph(f"<b>{pdf_labels['gradcam_link']}:</b>", body_style),
            Paragraph(f"<font color='#0284C7'><u>{gradcam_url}</u></font>", body_style),
        ],
        [
            Paragraph(f"<b>{pdf_labels['lesion_link']}:</b>", body_style),
            Paragraph(f"<font color='#0284C7'><u>{lesion_url}</u></font>", body_style),
        ],
    ]

    image_table = Table(image_links_data, colWidths=[200, 340])
    image_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    elements.append(image_table)
    elements.append(Spacer(1, 14))

    # ---------------------------------------------------------
    # 7. Clinical Disclaimer
    # ---------------------------------------------------------
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=body_style,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
        fontName=base_font,
    )

    elements.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.HexColor("#CBD5E1"),
            spaceBefore=6,
            spaceAfter=8,
        )
    )
    elements.append(Paragraph(f"<b>{pdf_labels['disclaimer_title']}:</b> {pdf_labels['disclaimer_text']}", disclaimer_style))

    doc.build(elements)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes
