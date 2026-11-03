"""Convert every corpus PDF with Docling (PDF text, no OCR, table structure on) and store one Markdown string per page.

Usage: uv run --no-project --python 3.12 --with docling python convert_with_docling.py <pdf_folder> <output_folder>
The output folder is the input of talk2.conversion_comparison.
"""
import json, sys, time
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
root, output = Path(sys.argv[1]), Path(sys.argv[2])
options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
started = time.time(); pages = 0
for pdf in sorted(root.rglob("*.pdf")):
    relative = pdf.relative_to(root).with_suffix("")
    target = output / f"{str(relative).replace('/', '__')}.json"
    if target.exists(): continue
    try:
        document = converter.convert(pdf).document
        result = {str(i): document.export_to_markdown(page_no=i + 1) for i in range(document.num_pages())}
    except Exception as error:
        result = {"error": f"{type(error).__name__}: {error}"}
    target.write_text(json.dumps(result, ensure_ascii=False)); pages += len(result)
    print(relative, len(result), round((time.time() - started) / max(pages, 1), 2), "s/page", flush=True)
print("DOCLING_TRIAL_COMPLETE", flush=True)
