"""Render the document previews shown in the talk: source pages cropped around highlighted evidence, and a contact sheet."""

import sys
from pathlib import Path

import pymupdf

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PDF = REPOSITORY_ROOT / "data" / "talk2_corpus_pages_holdout" / "pdf"
ULTRAGENYX = "cuad/ULTRAGENYXPHARMACEUTICALINC_12_23_2013-EX-10.9-SUPPLY AGREEMENT"
PREVIEWS = {
    "wage_chart": ("vidore-hr/labour_market_and_wage_developments_in_europe-KEBN24001ENN", 4, ["Graph 2.16"], 230),
    "exclusive_supply": (ULTRAGENYX, None, ["exclusively"], 200),
    "funding_amount": ("cuad/ADAPTIMMUNETHERAPEUTICSPLC_04_06_2017-EX-10.11-STRATEGIC ALLIANCE AGREEMENT", None, ["19,644,000"], 170),
    "french_gas": ("vidore-energy/programmation_pluriannuelle_de_lenergie_20200422", None, ["gaz B", "Pays-Bas"], 170),
    "fatf_contents": ("xl-docbench/doc_000219", 0, ["Table of Acronyms"], 200),
    "fatf_expansion": ("xl-docbench/doc_000219", None, ["virtual asset service provider"], 170),
    "israel_law": ("cuad/NICELTD_06_26_2003-EX-4.5-OUTSOURCING AGREEMENT", 1, ["LAW AND JURISDICTION", "laws of the state of Israel"], 150),
    "dior_share": ("vidore-finance_fr/christian_dior_2023", None, ["97,50"], 150),
    "culvert_chart": ("xl-docbench/doc_000143", None, ["Equation 2"], 190),
}
HIGHLIGHT = (0.96, 0.62, 0.05)
CONTACT_SHEET = ["vidore-hr/labour_market_and_wage_developments_in_europe-KEBN24001ENN", ULTRAGENYX, "xl-docbench/doc_000143",
                 "vidore-energy/rapport_onre_mars2022", "vidore-finance_fr/christian_dior_2023", "xl-docbench/doc_000219",
                 "vidore-industrial/TO_00-25-234", "xl-docbench/doc_000274", "cuad/NICELTD_06_26_2003-EX-4.5-OUTSOURCING AGREEMENT",
                 "xl-docbench/doc_000158", "vidore-energy/programmation_pluriannuelle_de_lenergie_20200422", "xl-docbench/doc_000067"]


def render_preview(name: str, document: str, page_index: int | None, terms: list[str], half_height: int | None,
                   output: Path) -> str:
    with pymupdf.open(CORPUS_PDF / f"{document}.pdf") as pdf:
        pages = [page_index] if page_index is not None else range(len(pdf))
        for index in pages:
            page = pdf[index]
            hits = [rect for term in terms for rect in page.search_for(term)]
            if not hits:
                continue
            for rect in hits[:6]:
                middle = (rect.y0 + rect.y1) / 2  # some fonts report glyph boxes several lines tall
                height = min(rect.height, 16) / 2
                page.draw_rect(pymupdf.Rect(rect.x0 - 1, middle - height, rect.x1 + 1, middle + height), color=None,
                               fill=HIGHLIGHT, fill_opacity=0.4, width=0, overlay=True)
            clip = page.rect
            if half_height:
                middle = hits[0].y0
                clip = pymupdf.Rect(page.rect.x0, max(page.rect.y0, middle - half_height), page.rect.x1,
                                    min(page.rect.y1, middle + half_height))
            page.get_pixmap(dpi=170, clip=clip).save(output / f"{name}.png")
            return f"{name}: page {index}, {len(hits)} hits, clip {clip.width:.0f}x{clip.height:.0f}"
    return f"{name}: NO HITS"


def render_contact_sheet(output: Path) -> None:
    columns, rows, cell_width, cell_height, gap = 6, 2, 300, 400, 16
    width, height = columns * cell_width + (columns - 1) * gap, rows * cell_height + (rows - 1) * gap
    sheet = pymupdf.open()
    canvas = sheet.new_page(width=width, height=height)
    for position, document in enumerate(CONTACT_SHEET):
        with pymupdf.open(CORPUS_PDF / f"{document}.pdf") as pdf:
            column, row = position % columns, position // columns
            x, y = column * (cell_width + gap), row * (cell_height + gap)
            canvas.show_pdf_page(pymupdf.Rect(x, y, x + cell_width, y + cell_height), pdf, min(1, len(pdf) - 1))
    canvas.get_pixmap(dpi=96).save(output / "corpus_contact_sheet.png")


if __name__ == "__main__":
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    for name, (document, page_index, terms, half_height) in PREVIEWS.items():
        print(render_preview(name, document, page_index, terms, half_height, output))
    render_contact_sheet(output)
