"""Deep Seek Harness tools over a corpus folder: search (ripgrep), read, list_files and record_note."""

import csv
import subprocess
from pathlib import Path

from talk2.corpus_assembly import MANIFEST

OUTPUT_LIMIT = 6000
READ_LINE_LIMIT = 200

SEARCH = {"type": "function", "function": {
    "name": "search",
    "description": ("Search every file in the corpus with ripgrep. mode='lines' returns matching lines with line "
                    "numbers; mode='files' returns only the paths of files that match; mode='count' returns the "
                    "match count per file. Searches are exhaustive across all files, and an empty result means "
                    "the pattern does not occur."),
    "parameters": {"type": "object", "properties": {
        "pattern": {"type": "string", "description": "Regular expression (or literal text if fixed_string is true)."},
        "path_glob": {"type": "string", "description": "Restrict to files matching this glob, e.g. 'cuad/**' or "
                      "'**/christian_dior_2023.md'. Default: all files."},
        "mode": {"type": "string", "enum": ["lines", "files", "count"]},
        "fixed_string": {"type": "boolean"},
        "ignore_case": {"type": "boolean"},
        "context": {"type": "integer", "description": "Lines of context around each match in 'lines' mode (0-5)."},
    }, "required": ["pattern"]}}}

READ = {"type": "function", "function": {
    "name": "read",
    "description": f"Read up to {READ_LINE_LIMIT} lines of one corpus file, starting at a line number.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "start_line": {"type": "integer"}, "line_count": {"type": "integer"},
    }, "required": ["path"]}}}

LIST_FILES = {"type": "function", "function": {
    "name": "list_files",
    "description": "List corpus files matching a glob, with title, language and page count.",
    "parameters": {"type": "object", "properties": {"glob": {"type": "string"}}, "required": []}}}

RECORD_NOTE = {"type": "function", "function": {
    "name": "record_note",
    "description": ("Save a fact you have established, with the file and line where it is stated. Notes stay "
                    "visible for the rest of this session, including later questions."),
    "parameters": {"type": "object", "properties": {
        "fact": {"type": "string"}, "file": {"type": "string"}, "line": {"type": "integer"},
    }, "required": ["fact", "file"]}}}


def truncate(text: str, remainder_note: str = "") -> str:
    if len(text) <= OUTPUT_LIMIT:
        return text
    cut = text[:OUTPUT_LIMIT].rsplit("\n", 1)[0]
    return f"{cut}\n[output truncated at {OUTPUT_LIMIT} characters] {remainder_note}".rstrip()


class CorpusTools:
    def __init__(self, root: Path, suffix: str):
        self.root = root
        self.suffix = suffix
        self.catalogue = self.load_catalogue()

    def load_catalogue(self) -> list[dict]:
        rows = []
        with MANIFEST.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["present"] == "True":
                    rows.append({"path": str(Path(row["path"]).with_suffix(self.suffix)), "title": row["title"],
                                 "language": row["language"], "pages": row["pages"]})
        return rows

    def resolve(self, path: str) -> Path | None:
        candidate = (self.root / path.lstrip("/")).resolve()
        if self.root.resolve() in candidate.parents and candidate.is_file():
            return candidate
        return None

    def search(self, pattern: str, path_glob: str = "", mode: str = "lines", fixed_string: bool = False,
               ignore_case: bool = False, context: int = 0) -> str:
        base = ["rg", "--no-heading", "--color", "never", "--sortr", "path"]
        if fixed_string:
            base.append("-F")
        if ignore_case:
            base.append("-i")
        if path_glob:
            base += ["--glob", path_glob]
        file_count = sum(1 for _ in self.root.rglob(f"*{self.suffix}")) if not path_glob else None
        counts = subprocess.run(base + ["-c", "--", pattern, "."], cwd=self.root, capture_output=True, text=True)
        if counts.returncode == 2:
            return f"search error: {counts.stderr.strip()[:300]}"
        per_file = [line.rsplit(":", 1) for line in counts.stdout.splitlines() if ":" in line]
        total_matches = sum(int(n) for _, n in per_file)
        if total_matches == 0:
            searched = f"all {file_count} files" if file_count is not None else f"files matching '{path_glob}'"
            return f"0 matches for {pattern!r} in {searched}."
        summary = f"{total_matches} matching lines in {len(per_file)} files."
        if mode == "count":
            return truncate(summary + "\n" + "\n".join(f"{p.removeprefix('./')}: {n}" for p, n in per_file))
        if mode == "files":
            return truncate(summary + "\n" + "\n".join(p.removeprefix("./") for p, _ in per_file))
        lines = subprocess.run(base + ["-n", "-C", str(max(0, min(int(context or 0), 5))), "--", pattern, "."],
                               cwd=self.root, capture_output=True, text=True).stdout
        lines = "\n".join(line.removeprefix("./")[:400] for line in lines.splitlines())
        return truncate(summary + "\n" + lines,
                        "Narrow the pattern, restrict path_glob, or use mode='files' to see every matching file.")

    def read(self, path: str, start_line: int = 1, line_count: int = READ_LINE_LIMIT) -> str:
        resolved = self.resolve(path)
        if resolved is None:
            return f"No such file: {path}. Use list_files or search mode='files' to find paths."
        lines = resolved.read_text(errors="replace").splitlines()
        start = max(1, int(start_line or 1))
        count = max(1, min(int(line_count or READ_LINE_LIMIT), READ_LINE_LIMIT))
        chunk = lines[start - 1:start - 1 + count]
        body = "\n".join(f"{start + i}: {text[:600]}" for i, text in enumerate(chunk))
        return truncate(f"{path} lines {start}-{start + len(chunk) - 1} of {len(lines)}\n{body}")

    def list_files(self, glob: str = "**") -> str:
        from fnmatch import fnmatch

        rows = [r for r in self.catalogue if fnmatch(r["path"], glob or "**") or fnmatch(r["path"], f"**/{glob}")]
        body = "\n".join(f"{r['path']} | {r['title'][:90]} | {r['language']} | {r['pages']} pages" for r in rows)
        return truncate(f"{len(rows)} files\n{body}", "Use a narrower glob.")
