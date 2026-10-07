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
) -> bytes:
    """
    Generates a professional clinical PDF report for a DrishtiAI screening record.

    Parameters
    ----------
    screening_data : dict
        Serialized dictionary of a screening record.
    patient_info : dict, optional
        Demographic information (Name, Age, Gender, MRN, etc.) from external patient API.
    gemini_narrative : dict, optional
        Gemini Flash AI narrative dictionary containing clinical and patient summaries.

    Returns
    -------
    bytes
        PDF file content in bytes.
    """

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

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1E293B"),
        fontName="Helvetica-Bold",
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )

    bold_body_style = ParagraphStyle(
        "ReportBodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    elements = []

    # ---------------------------------------------------------
    # 1. Header Section
    # ---------------------------------------------------------

    header_table_data = [
        [
            Paragraph("<b>DrishtiAI</b>", title_style),
            Paragraph(
                f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                ParagraphStyle("RightAlign", parent=body_style, alignment=2),
            ),
        ],
        [
            Paragraph(
                "Explainable AI-Based Diabetic Retinopathy Screening Report",
                subtitle_style,
            ),
            Paragraph(
                f"<b>Screening ID:</b> {screening_data.get('screening_id', 'N/A')}",
                ParagraphStyle("RightAlign2", parent=body_style, alignment=2),
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
    elements.append(Spacer(1, 10))
    elements.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#0284C7"),
            spaceBefore=5,
            spaceAfter=15,
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

    evidence = screening_data.get("evidence") or {}
    decision_info = evidence.get("screening_decision") or {}

    decision = decision_info.get("decision", "PENDING/UNKNOWN")
    referable = decision_info.get("referable")

    if decision == "REFER" or referable is True:
        dec_color = colors.HexColor("#DC2626")  # Red
        dec_bg = colors.HexColor("#FEF2F2")
    elif decision == "NON-REFERABLE" or referable is False:
        dec_color = colors.HexColor("#16A34A")  # Green
        dec_bg = colors.HexColor("#F0FDF4")
    else:
        dec_color = colors.HexColor("#EA580C")  # Orange
        dec_bg = colors.HexColor("#FFF7ED")

    summary_box_data = [
        [
            Paragraph("<b>Patient Name:</b>", body_style),
            Paragraph(str(p_name), bold_body_style),
            Paragraph("<b>Patient ID / User ID:</b>", body_style),
            Paragraph(str(user_id), bold_body_style),
        ],
        [
            Paragraph("<b>Age / Gender:</b>", body_style),
            Paragraph(f"{p_age} / {p_gender}", body_style),
            Paragraph("<b>Hospital MRN:</b>", body_style),
            Paragraph(str(p_mrn), body_style),
        ],
        [
            Paragraph("<b>Referral Decision:</b>", body_style),
            Paragraph(
                f"<font color='{dec_color.hexval()}'><b>{decision}</b></font>",
                ParagraphStyle("DecStyle", parent=body_style, fontSize=11),
            ),
            Paragraph("<b>Screening Status:</b>", body_style),
            Paragraph(f"<b>{status}</b>", bold_body_style),
        ],
    ]

    summary_table = Table(summary_box_data, colWidths=[110, 150, 110, 170])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BACKGROUND", (1, 1), (1, 1), dec_bg),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#F1F5F9")),
                ("PADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    # ---------------------------------------------------------
    # 2b. Gemini AI Narrative Section (Optional)
    # ---------------------------------------------------------

    dev_font = _get_devanagari_font_name()

    g_narrative = gemini_narrative or {}
    if g_narrative.get("generated") and (g_narrative.get("clinical_narrative") or g_narrative.get("patient_summary")):
        elements.append(Paragraph("Gemini AI Clinical & Patient Interpretation", section_heading))

        ai_box_data = []

        if g_narrative.get("clinical_narrative"):
            ai_box_data.append([
                Paragraph("<b>Ophthalmologist Impression:</b>", body_style),
                Paragraph(g_narrative["clinical_narrative"], ParagraphStyle("AiBody", parent=body_style, fontName=dev_font, fontSize=9.5, leading=13)),
            ])

        if g_narrative.get("patient_summary"):
            ai_box_data.append([
                Paragraph("<b>Patient Guidance:</b>", body_style),
                Paragraph(g_narrative["patient_summary"], ParagraphStyle("AiBody2", parent=body_style, fontName=dev_font, fontSize=9.5, leading=13)),
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
        elements.append(Spacer(1, 12))

    # ---------------------------------------------------------
    # 3. DR Classification Details
    # ---------------------------------------------------------

    elements.append(Paragraph("Diabetic Retinopathy Classification", section_heading))

    dr_grade = screening_data.get("dr_grade")
    dr_label = screening_data.get("dr_label", "N/A")
    dr_conf = screening_data.get("dr_confidence")
    dr_conf_str = f"{dr_conf * 100:.1f}%" if dr_conf is not None else "N/A"

    class_data = [
        [
            Paragraph("<b>Primary DR Grade</b>", bold_body_style),
            Paragraph("<b>Class Label</b>", bold_body_style),
            Paragraph("<b>Confidence Score</b>", bold_body_style),
        ],
        [
            Paragraph(f"Grade {dr_grade}" if dr_grade is not None else "N/A", body_style),
            Paragraph(str(dr_label), body_style),
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
    elements.append(Spacer(1, 15))

    # ---------------------------------------------------------
    # 4. Image Quality Assessment
    # ---------------------------------------------------------

    elements.append(Paragraph("Image Quality Assessment", section_heading))

    q_score = screening_data.get("quality_score")
    q_score_str = f"{q_score:.2f}" if q_score is not None else "N/A"
    q_status = screening_data.get("quality_status", "N/A").title()
    f_suit = screening_data.get("fundus_suitability")
    f_suit_str = "Suitable (Fundus)" if f_suit is True else ("Unsuitable" if f_suit is False else "N/A")

    quality_data = [
        [
            Paragraph("<b>Technical Score</b>", bold_body_style),
            Paragraph("<b>Quality Status</b>", bold_body_style),
            Paragraph("<b>Fundus Suitability AI</b>", bold_body_style),
        ],
        [
            Paragraph(q_score_str, body_style),
            Paragraph(q_status, body_style),
            Paragraph(f_suit_str, body_style),
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
    elements.append(Spacer(1, 15))

    # ---------------------------------------------------------
    # 5. Lesion Detection Breakdown
    # ---------------------------------------------------------

    elements.append(Paragraph("Lesion Detection Summary", section_heading))

    lesion_evidence = evidence.get("lesion_evidence") or {}
    total_lesions = screening_data.get("lesion_count", 0)

    ma_count = lesion_evidence.get("microaneurysms", 0)
    he_count = lesion_evidence.get("hemorrhages", 0)
    ex_count = lesion_evidence.get("exudates", lesion_evidence.get("hard_exudates", 0))
    cws_count = lesion_evidence.get("cotton_wool_spots", lesion_evidence.get("soft_exudates", 0))

    lesion_table_data = [
        [
            Paragraph("<b>Lesion Type</b>", bold_body_style),
            Paragraph("<b>Detected Count</b>", bold_body_style),
        ],
        [Paragraph("Microaneurysms", body_style), Paragraph(str(ma_count), body_style)],
        [Paragraph("Hemorrhages", body_style), Paragraph(str(he_count), body_style)],
        [Paragraph("Exudates", body_style), Paragraph(str(ex_count), body_style)],
        [Paragraph("Cotton Wool Spots", body_style), Paragraph(str(cws_count), body_style)],
        [Paragraph("<b>Total Lesions Detected</b>", bold_body_style), Paragraph(f"<b>{total_lesions}</b>", bold_body_style)],
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
    elements.append(Spacer(1, 15))

    # ---------------------------------------------------------
    # 6. Visual Evidence Links
    # ---------------------------------------------------------

    elements.append(Paragraph("Visual Evidence & Explainability Links", section_heading))

    gradcam_url = screening_data.get("gradcam_url") or "N/A"
    lesion_url = screening_data.get("lesion_overlay_url") or "N/A"

    image_links_data = [
        [
            Paragraph("<b>Grad-CAM++ Attention Overlay:</b>", body_style),
            Paragraph(f"<font color='#0284C7'><u>{gradcam_url}</u></font>", body_style),
        ],
        [
            Paragraph("<b>Lesion Detection Overlay:</b>", body_style),
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
    elements.append(Spacer(1, 20))

    # ---------------------------------------------------------
    # 7. Clinical Disclaimer
    # ---------------------------------------------------------

    disclaimer_text = (
        "<b>Clinical Disclaimer:</b> DrishtiAI is an artificial intelligence decision-support tool. "
        "Grad-CAM++ heatmaps indicate model feature attention and should be interpreted alongside clinical assessment "
        "by a certified ophthalmologist. Final diagnosis and treatment decisions remain the responsibility of the clinician."
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=body_style,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
    )

    elements.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.HexColor("#CBD5E1"),
            spaceBefore=10,
            spaceAfter=10,
        )
    )
    elements.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(elements)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes
