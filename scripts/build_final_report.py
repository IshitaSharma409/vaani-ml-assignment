"""Render the two-page report source and generated tables into a landscape PDF."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "submissions/final_report.md"
TABLES = ROOT / "submissions/final_tables.md"


def _parse_tables(text: str) -> dict[str, list[list[str]]]:
    result: dict[str, list[list[str]]] = {}
    current = None
    rows: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("## Table "):
            if current:
                result[current] = rows
            current = line.split(".", 1)[0].removeprefix("## ").replace(" ", "")
            rows = []
            continue
        if current and line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                rows.append(cells)
    if current:
        result[current] = rows
    return result


def _paragraph_text(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    return text.replace("&lt;", "&lt;").replace("&gt;", "&gt;")


def _table_flowable(rows: list[list[str]], page_width: float) -> Table:
    col_count = len(rows[0])
    if col_count == 4:
        proportions = [0.12, 0.31, 0.40, 0.17]
    elif col_count == 7:
        proportions = [0.14, 0.07, 0.07, 0.07, 0.07, 0.29, 0.29]
    elif col_count == 8 and "Setting" in rows[0][0]:
        proportions = [0.23, 0.08, 0.09, 0.07, 0.09, 0.12, 0.16, 0.16]
    elif col_count == 8:
        proportions = [0.15, 0.07, 0.07, 0.10, 0.08, 0.07, 0.23, 0.23]
    else:
        proportions = [1 / col_count] * col_count
    data = []
    for row_index, row in enumerate(rows):
        style = ParagraphStyle(
            f"table-cell-{col_count}-{row_index}", fontName="Helvetica-Bold" if row_index == 0 else "Helvetica",
            fontSize=5.7, leading=6.5, alignment=TA_LEFT,
            textColor=colors.white if row_index == 0 else colors.HexColor("#17283b"))
        data.append([Paragraph(_paragraph_text(cell), style) for cell in row])
    table = Table(data, colWidths=[page_width * p for p in proportions], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f4f7fa")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fa")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bac7d3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
    ]))
    return table


def build_pdf(output: Path, allow_draft: bool = False) -> int:
    source = REPORT.read_text(encoding="utf-8")
    tables = _parse_tables(TABLES.read_text(encoding="utf-8"))
    required_keys = {"Table1", "Table2a", "Table2b", "Table3"}
    if not required_keys.issubset(tables):
        raise ValueError(f"Generated table file lacks: {sorted(required_keys - tables.keys())}")
    pending = ["Student feedback required", "Student reflection required", "Student confirmation required"]
    if not allow_draft and any(marker in source for marker in pending):
        raise ValueError("The report still needs the student's personal reflection/feedback. Use --draft only for previewing.")

    page_size = landscape(A4)
    left = right = 10 * mm
    top = 9 * mm
    bottom = 9 * mm
    usable_width = page_size[0] - left - right
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("RelayTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=15, leading=17, alignment=TA_LEFT, textColor=colors.HexColor("#17324d"),
                              spaceAfter=2))
    styles.add(ParagraphStyle("RelayMeta", parent=styles["Normal"], fontName="Helvetica",
                              fontSize=6.5, leading=7, textColor=colors.HexColor("#53677b"), spaceAfter=4))
    styles.add(ParagraphStyle("RelayH2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                              fontSize=8.5, leading=9.2, textColor=colors.HexColor("#17324d"),
                              spaceBefore=4, spaceAfter=2, keepWithNext=True))
    styles.add(ParagraphStyle("RelayBody", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=7, leading=8, textColor=colors.HexColor("#1d2a36"),
                              alignment=TA_LEFT, spaceAfter=3))
    styles.add(ParagraphStyle("RelayBullet", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=6.8, leading=7.6, leftIndent=7, firstLineIndent=-5,
                              textColor=colors.HexColor("#1d2a36"), spaceAfter=1.5))

    story = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            story.append(Paragraph(_paragraph_text(" ".join(paragraph_lines)), styles["RelayBody"]))
            paragraph_lines.clear()

    for raw_line in source.splitlines():
        line = raw_line.strip()
        if line == "<!-- PAGEBREAK -->":
            flush_paragraph()
            story.append(PageBreak())
        elif line.startswith("{{TABLE"):
            flush_paragraph()
            placeholder = line.strip("{}")
            key = {"TABLE1": "Table1", "TABLE2A": "Table2a", "TABLE2B": "Table2b",
                   "TABLE3": "Table3"}.get(placeholder, placeholder)
            story.append(_table_flowable(tables[key], usable_width))
            story.append(Spacer(1, 3))
        elif not line:
            flush_paragraph()
        elif line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(_paragraph_text(line[2:]), styles["RelayTitle"]))
        elif line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(_paragraph_text(line[3:]), styles["RelayH2"]))
        elif line.startswith("**Engineering experiment report"):
            flush_paragraph()
            story.append(Paragraph(_paragraph_text(line.replace("**", "")), styles["RelayMeta"]))
        elif line.startswith("- "):
            flush_paragraph()
            story.append(Paragraph("• " + _paragraph_text(line[2:]), styles["RelayBullet"]))
        else:
            paragraph_lines.append(line)
    flush_paragraph()

    def decorate(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cad4de"))
        canvas.line(left, page_size[1] - 6 * mm, page_size[0] - right, page_size[1] - 6 * mm)
        canvas.setFont("Helvetica", 6)
        canvas.setFillColor(colors.HexColor("#64788c"))
        canvas.drawString(left, 5 * mm, "The Relay | synthetic load results are labelled stand-in")
        canvas.drawRightString(page_size[0] - right, 5 * mm, f"{doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output), pagesize=page_size, leftMargin=left, rightMargin=right,
                            topMargin=top, bottomMargin=bottom,
                            title="The Relay engineering experiment report",
                            author="The Relay project")
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    reader = PdfReader(str(output))
    pages = len(reader.pages)
    if pages > 2:
        raise ValueError(f"Report rendered to {pages} pages; the assignment limit is two.")
    rendered_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if "{{TABLE" in rendered_text or "<!-- PAGEBREAK -->" in rendered_text:
        raise ValueError("Unresolved report template markers were found in the PDF.")
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true", help="Allow clearly marked student-input sections in a draft PDF.")
    args = parser.parse_args()
    output = ROOT / "submissions" / ("final_report_draft.pdf" if args.draft else "final_report.pdf")
    pages = build_pdf(output, allow_draft=args.draft)
    print(f"Rendered {pages} pages to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
