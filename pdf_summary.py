"""
pdf_summary.py — render a protection suitability summary as a professional PDF.
"""
import io
import re
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

INK = colors.HexColor("#0E1525")
BLUE = colors.HexColor("#4F46E5")
LINE = colors.HexColor("#E6E9EF")
MUTED = colors.HexColor("#8A94A6")

_styles = getSampleStyleSheet()
_S = {
    "h1": ParagraphStyle("h1", parent=_styles["Title"], fontName="Helvetica-Bold",
                         fontSize=18, textColor=INK, spaceAfter=2, leading=22),
    "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=9.5,
                          textColor=MUTED, spaceAfter=2),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5,
                         textColor=INK, spaceBefore=14, spaceAfter=5, leading=14),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10,
                          textColor=INK, leading=15, spaceAfter=4),
    "foot": ParagraphStyle("foot", fontName="Helvetica-Oblique", fontSize=8,
                          textColor=MUTED, leading=11),
}

DISCLAIMER = (
    "This document is a draft generated to support the adviser's own suitability "
    "documentation. It reports the information captured during the fact-find and the "
    "needs identified; it does not constitute advice, a recommendation, or a quotation. "
    "Product selection, cover amounts and premiums are sourced and decided by the adviser "
    "in their certified tools. The adviser reviews, edits and is responsible for the final "
    "suitability assessment."
)

def _clean(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text

def _parse_sections(summary_text):
    sections = []
    cur_head, cur_body = None, []
    for raw in summary_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        stripped = line.strip("*# ").rstrip(":").strip()
        is_heading = False
        if (line.startswith("#") or line.startswith("**")) and len(stripped) < 60:
            is_heading = True
        elif line.endswith(":") and len(stripped) < 45 and stripped[:1].isupper():
            is_heading = True
        if is_heading:
            if cur_head:
                sections.append((cur_head, cur_body))
            cur_head, cur_body = stripped, []
        else:
            line = re.sub(r"^[-*•]\s*", "", line)
            cur_body.append(line)
    if cur_head:
        sections.append((cur_head, cur_body))
    if not sections and summary_text.strip():
        sections = [("Summary", [l.strip() for l in summary_text.splitlines() if l.strip()])]
    return sections

def build_summary_pdf(summary_text, meta):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=22*mm, rightMargin=22*mm,
                            topMargin=20*mm, bottomMargin=18*mm,
                            title="Suitability Summary")
    story = []
    story.append(Paragraph("Suitability Summary", _S["h1"]))
    story.append(Paragraph("Fact-find record &mdash; draft for adviser review", _S["sub"]))
    story.append(Spacer(1, 8))
    meta_rows = [
        ["Client", meta.get("client", "-"), "Date", meta.get("date", date.today().strftime("%d %B %Y"))],
        ["Adviser", meta.get("adviser", "-"), "Firm", meta.get("firm", "-")],
    ]
    t = Table(meta_rows, colWidths=[24*mm, 62*mm, 20*mm, 40*mm])
    t.setStyle(TableStyle([
        ("FONT",(0,0),(-1,-1),"Helvetica",9),
        ("FONT",(0,0),(0,-1),"Helvetica-Bold",9),
        ("FONT",(2,0),(2,-1),"Helvetica-Bold",9),
        ("TEXTCOLOR",(0,0),(0,-1),MUTED),
        ("TEXTCOLOR",(2,0),(2,-1),MUTED),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=LINE, spaceAfter=4))
    for heading, paras in _parse_sections(summary_text):
        story.append(Paragraph(_clean(heading).upper(), _S["h2"]))
        for p in paras:
            story.append(Paragraph(_clean(p), _S["body"]))
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=LINE, spaceAfter=8))
    story.append(Paragraph(DISCLAIMER, _S["foot"]))
    doc.build(story)
    buf.seek(0)
    return buf.read()
