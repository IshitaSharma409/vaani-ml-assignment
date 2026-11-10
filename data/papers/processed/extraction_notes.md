# Phase 1 PDF inspection notes

Inspection date: 2026-11-10. PDFs were checked with `pypdf` and rendered with Poppler for representative visual review. Page references below use 1-based PDF page indices; printed page labels can differ.

| PDF | Pages | Representative inspection | Extraction/layout notes |
|---|---:|---|---|
| `1706.03762v7.pdf` | 15 | Title/abstract page 1; technical page 5; table page 6; final page 15. | Title, abstract, table, and main text extract legibly. The last page is a figure with word-to-word attention lines, so linear extraction is not semantically ordered even though the rendered page is clear. Keep page citations and treat figure-heavy content cautiously. |
| `2410.00037v1.pdf` | 66 | Title/abstract page 1; model section/figure page 6; final table page 66. | Main prose, figure caption, and final table extract legibly. Accented author names show decomposed/spacing artifacts in text extraction. Repeated running header is present. Visual pages remain readable. |
| `2507.07955v2.pdf` | 39 | Title/abstract page 1; technical/figure page 3; final page 39. | Main prose and captions are readable. The figure page contains complex diagrams and labels whose extracted order is not reliable. Page 39 includes a dense visualization whose small glyphs and numeric labels extract as noisy characters; nearby prose is legible in the rendered PDF. Avoid treating extracted figure glyphs as evidence without checking the PDF page. |

The extraction process must retain page-level metadata. Do not infer printed page numbers from PDF indices. OCR, figure-table reconstruction, and section-heading detection have not been implemented; they remain ingestion concerns for Phase 2. The original PDFs remain authoritative.
