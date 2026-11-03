"""Tools of the structured grep agent: output that states what was searched, where each hit sits and what remains."""

import re
import subprocess
from pathlib import Path

from talk2.harness_tools import OUTPUT_LIMIT, CorpusTools

PAGE_MARKER = re.compile(r"<!-- page (\d+) -->")
HEADING = re.compile(r"^(#{1,6} |\*\*[^*].*\*\*\s*$)")
MATCHES_SHOWN_PER_FILE = 12
DEFAULT_READ_LINES = 80
LINE_DISPLAY_LIMIT = 400
READ_LINE_LIMIT = 600
EXCERPT_RADIUS = 200
EXCERPTS_PER_LINE = 3

SEARCH = {"type": "function", "function": {
    "name": "search",
    "description": ("Search every file with ripgrep, ignoring case. Returns matching lines grouped by file, each with "
                    "its line number and page. mode='count' returns only the number of matching lines per file, which "
                    "is the way to enumerate every file that mentions a term. Short terms of one to three words find "
                    "more than long phrases."),
    "parameters": {"type": "object", "properties": {
        "pattern": {"type": "string", "description": "Regular expression. Use | for alternatives: 'gaz B|gas B'."},
        "file": {"type": "string", "description": "Search only this file path from the catalogue. Omit to search all."},
        "mode": {"type": "string", "enum": ["lines", "count"], "description": "Default 'lines'."},
    }, "required": ["pattern"]}}}

OUTLINE = {"type": "function", "function": {
    "name": "outline",
    "description": "List the headings and page starts of one file with their line numbers, to locate a named section.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}

READ = {"type": "function", "function": {
    "name": "read",
    "description": (f"Read one file by line range (default {DEFAULT_READ_LINES} lines) or one whole page. Line numbers "
                    "come from search or outline, never from a printed table of contents."),
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "start_line": {"type": "integer"}, "line_count": {"type": "integer"},
        "page": {"type": "integer", "description": "A number from a <!-- page N --> marker. Replaces start_line."},
    }, "required": ["path"]}}}

TOOL_DEFINITIONS = [SEARCH, OUTLINE, READ]


class StructuredCorpusTools(CorpusTools):
    def __init__(self, root: Path, suffix: str, show_long_lines: bool = False):
        """show_long_lines=False reproduces revision 1, whose output silently cut lines at a fixed length."""
        super().__init__(root, suffix)
        self.show_long_lines = show_long_lines
        self.languages = sorted({r["language"] for r in self.catalogue if r["language"]})

    def catalogue_text(self) -> str:
        return "\n".join(f"{r['path']} | {r['title'][:110]} | {r['language']} | {r['pages']} pages"
                         for r in self.catalogue)

    def file_lines(self, path: str) -> list[str] | None:
        resolved = self.resolve(path)
        return resolved.read_text(errors="replace").splitlines() if resolved else None

    @staticmethod
    def page_of_each_line(lines: list[str]) -> list[str]:
        pages, current = [], "?"
        for line in lines:
            marker = PAGE_MARKER.search(line)
            current = marker.group(1) if marker else current
            pages.append(current)
        return pages

    def matching_text(self, pattern: str, text: str) -> str:
        """The text of a matching line: whole if short, otherwise excerpts around each match."""
        text = text.strip()
        if not self.show_long_lines:
            return text[:LINE_DISPLAY_LIMIT]
        if len(text) <= LINE_DISPLAY_LIMIT:
            return text
        try:
            found = list(re.finditer(pattern, text, re.IGNORECASE))
        except re.error:
            found = list(re.finditer(re.escape(pattern), text, re.IGNORECASE))
        spans: list[list[int]] = []
        for match in found:
            start, end = max(0, match.start() - EXCERPT_RADIUS), min(len(text), match.end() + EXCERPT_RADIUS)
            if spans and start <= spans[-1][1]:
                spans[-1][1] = end
            elif len(spans) < EXCERPTS_PER_LINE:
                spans.append([start, end])
        if not spans:
            spans = [[0, LINE_DISPLAY_LIMIT]]
        excerpts = " ... ".join(text[start:end] for start, end in spans)
        return f"{excerpts} [excerpts of a {len(text)}-character line]"

    def search(self, pattern: str, file: str = "", mode: str = "lines") -> str:
        if file and self.resolve(file) is None:
            return f"No file '{file}'. Omit file to search everything, or use an exact path from the catalogue."
        target = file or "."
        arguments = ["rg", "--no-heading", "--color", "never", "--sort", "path", "-i"]
        counted = subprocess.run(arguments + ["-c", "--", pattern, target], cwd=self.root, capture_output=True, text=True)
        literal = []
        if counted.returncode == 2:  # an invalid regular expression is retried as literal text
            literal = ["-F"]
            counted = subprocess.run(arguments + literal + ["-c", "--", pattern, target], cwd=self.root,
                                     capture_output=True, text=True)
        rows = [line.rsplit(":", 1) for line in counted.stdout.splitlines() if ":" in line] if not file else (
            [[file, counted.stdout.strip()]] if counted.stdout.strip() else [])
        rows = [(path.removeprefix("./"), int(count)) for path, count in rows]
        if not rows:
            scope = f"file {file}" if file else f"all {len(self.catalogue)} files"
            return (f"0 matches for '{pattern}' in {scope}. This does not yet show absence. Try one distinctive word, a "
                    f"synonym, or the wording of the document's language (corpus languages: {', '.join(self.languages)}).")
        total = sum(count for _, count in rows)
        header = f"{total} matching lines in {len(rows)} files."
        if mode == "count":
            body = "\n".join(f"{path}: {count}" for path, count in rows)
            return self.limit_output(f"{header}\n{body}", "Narrow the pattern.")
        hits = subprocess.run(arguments + literal + ["-n", "--with-filename", "-m", str(MATCHES_SHOWN_PER_FILE), "--",
                                                     pattern, target],
                              cwd=self.root, capture_output=True, text=True).stdout
        grouped: dict[str, list[str]] = {}
        for hit in hits.splitlines():
            path, _, remainder = hit.removeprefix("./").partition(":")
            number, _, text = remainder.partition(":")
            grouped.setdefault(path, []).append((number, text))
        sections = []
        for path, count in rows:
            pages = self.page_of_each_line(self.file_lines(path) or [])
            shown = grouped.get(path, [])
            lines = [f"{number} [page {pages[int(number) - 1] if int(number) <= len(pages) else '?'}]: "
                     f"{self.matching_text(pattern, text)}" for number, text in shown]
            remainder = f"\n(+{count - len(shown)} more matching lines in this file)" if count > len(shown) else ""
            sections.append(f"<file path=\"{path}\" matching_lines=\"{count}\">\n" + "\n".join(lines) + remainder + "\n</file>")
        return self.limit_output(f"{header}\n" + "\n".join(sections), "Use mode='count' to see every matching file.")

    @staticmethod
    def limit_output(text: str, remainder_note: str) -> str:
        if len(text) <= OUTPUT_LIMIT:
            return text
        cut = text[:OUTPUT_LIMIT].rsplit("\n", 1)[0]
        return f"{cut}\n[output stopped at {OUTPUT_LIMIT} characters] {remainder_note}"

    def outline(self, path: str) -> str:
        lines = self.file_lines(path)
        if lines is None:
            return f"No file '{path}'. Use an exact path from the catalogue."
        entries = [f"{index}: {line.strip()[:120]}" for index, line in enumerate(lines, start=1)
                   if PAGE_MARKER.search(line) or HEADING.match(line)]
        return self.limit_output(f"{path}: {len(lines)} lines\n" + "\n".join(entries), "Search for the section name.")

    def line_range(self, path: str, start_line: int | None, line_count: int | None, page: int | None
                   ) -> tuple[int, int] | None:
        """The first and last line (1-based, inclusive) of the requested range, or None if it does not exist."""
        lines = self.file_lines(path)
        if lines is None:
            return None
        if page is not None:
            pages = self.page_of_each_line(lines)
            numbers = [index for index, value in enumerate(pages, start=1) if value == str(page)]
            return (numbers[0], numbers[-1]) if numbers else None
        start = max(1, int(start_line or 1))
        return start, min(len(lines), start + max(1, int(line_count or DEFAULT_READ_LINES)) - 1)

    def read(self, path: str, start_line: int | None = None, line_count: int | None = None,
             page: int | None = None) -> tuple[str, tuple[int, int] | None]:
        """Return the text shown and the line range actually shown, which the output limit may shorten."""
        lines = self.file_lines(path)
        if lines is None:
            return f"No file '{path}'. Use an exact path from the catalogue.", None
        requested = self.line_range(path, start_line, line_count, page)
        if requested is None or requested[0] > len(lines):
            markers = sorted({value for value in self.page_of_each_line(lines) if value != "?"}, key=int)
            return f"No such range in {path}: it has {len(lines)} lines and pages {', '.join(markers)}.", None
        shown, used = [], 0
        last = requested[0] - 1
        for number in range(requested[0], requested[1] + 1):
            line = lines[number - 1]
            if not self.show_long_lines:
                rendered = f"{number}: {line[:READ_LINE_LIMIT]}"
            elif len(line) > OUTPUT_LIMIT:
                rendered = (f"{number}: {line[:OUTPUT_LIMIT]} [line {number} continues for {len(line) - OUTPUT_LIMIT} "
                            "more characters: search this file for a specific term to see the rest]")
            else:
                rendered = f"{number}: {line}"
            if used + len(rendered) > OUTPUT_LIMIT and shown:
                break
            shown.append(rendered)
            used += len(rendered) + 1
            last = number
        pages = self.page_of_each_line(lines)
        header = f"{path} lines {requested[0]}-{last} of {len(lines)}, pages {pages[requested[0] - 1]}-{pages[last - 1]}"
        footer = f"\n[continue from line {last + 1}]" if last < len(lines) else "\n[end of file]"
        return header + "\n" + "\n".join(shown) + footer, (requested[0], last)

    def quote_is_present(self, path: str, quote: str) -> bool:
        lines = self.file_lines(path)
        if lines is None:
            return False
        squeeze = lambda text: re.sub(r"[\W_]+", " ", text).strip().lower()
        return squeeze(quote) in squeeze("\n".join(lines))
