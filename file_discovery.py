import re
from dataclasses import dataclass
from pathlib import Path
from models import PrioritizedFiles
from criteria import Criterion


class FileFilter:
    DEFAULT_TEXT_EXTENSIONS = {
        ".txt", ".md", ".markdown", ".rst",
        ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
        ".java", ".py", ".pyw",
        ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".svelte", ".vue",
        ".html", ".htm", ".css", ".scss", ".sass", ".less",
        ".json", ".yaml", ".yml", ".xml", ".toml", ".ini", ".cfg", ".conf",
        ".go", ".rs", ".rb", ".php", ".sql", ".swift", ".kt", ".kts",
        ".sh", ".bash", ".zsh", ".fish", ".bat", ".cmd", ".ps1",
        ".csv", ".tsv", ".env", ".log"
    }

    DEFAULT_IGNORED_DIRECTORIES = {
        ".git", ".svn", ".hg", "node_modules", "vendor",
        "dist", "build", "out", ".next", ".nuxt", ".svelte-kit",
        "__pycache__", ".pytest_cache", ".venv", "venv", "env",
        "target", ".idea", ".vscode", "swagger-ui", "pyodide", "locales"
    }

    DEFAULT_IGNORED_FILES = {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "uv.lock",
        "poetry.lock", "cargo.lock", "composer.lock", "changelog.md",
        "license", "license.md"
    }

    def __init__(
        self,
        allowed_extensions: set[str] | None = None,
        ignored_directories: set[str] | None = None,
        ignored_files: set[str] | None = None,
        max_file_size_bytes: int = 5 * 1024 * 1024,
    ):
        self.allowed_extensions = allowed_extensions or self.DEFAULT_TEXT_EXTENSIONS
        self.ignored_directories = ignored_directories or self.DEFAULT_IGNORED_DIRECTORIES
        self.ignored_files = ignored_files or self.DEFAULT_IGNORED_FILES
        self.max_file_size_bytes = max_file_size_bytes

    def is_text_file(self, file_path: Path) -> bool:
        # Validate file extension, ignore lists, size, and non-binary content.
        if file_path.suffix.lower() not in self.allowed_extensions:
            return False

        if any(part.lower() in self.ignored_directories for part in file_path.parts):
            return False

        file_name_lower = file_path.name.lower()
        if file_name_lower in self.ignored_files:
            return False

        if file_name_lower.endswith(".min.js") or file_name_lower.endswith(".min.css") or file_name_lower.endswith("-bundle.js"):
            return False

        try:
            stat_result = file_path.stat()
            if stat_result.st_size > self.max_file_size_bytes or stat_result.st_size == 0:
                return False

            with open(file_path, "rb") as probe_file:
                first_kilobyte = probe_file.read(1024)
                if b"\x00" in first_kilobyte:
                    return False
        except (OSError, PermissionError):
            return False

        return True

    def discover_files(self, directory_path: Path) -> list[Path]:
        matched_paths: list[Path] = []
        for candidate_path in directory_path.rglob("*"):
            if candidate_path.is_file() and self.is_text_file(candidate_path):
                matched_paths.append(candidate_path)
        return matched_paths


class KeywordExtractor:
    STOP_WORDS = {
        "the", "is", "at", "which", "on", "a", "an", "and", "or", "not",
        "in", "to", "for", "with", "this", "that", "does", "text", "contains",
        "contain", "matching", "match", "criteria", "any", "all", "what", "where",
        "deal", "deals", "code", "line", "lines", "find", "looking", "look", "want"
    }

    def extract_keywords(self, criteria: list[Criterion]) -> set[str]:
        extracted_keywords: set[str] = set()
        for criterion in criteria:
            clean_instruction = re.sub(r"[^\w\s]", " ", criterion.instruction.lower())
            words = clean_instruction.split()
            for word in words:
                if len(word) >= 2 and word not in self.STOP_WORDS:
                    extracted_keywords.add(word)
        return extracted_keywords


@dataclass
class ScoredFile:
    file_path: Path
    relevance_score: int


class FilePrioritizer:
    def __init__(
        self,
        keyword_extractor: KeywordExtractor | None = None,
        max_high_priority_files: int = 25,
    ):
        self.keyword_extractor = keyword_extractor or KeywordExtractor()
        self.max_high_priority_files = max_high_priority_files

    def _calculate_file_relevance_score(self, file_path: Path, keywords: set[str]) -> int:
        # Count keyword occurrences with multi-keyword coverage weighting.
        if not keywords:
            return 0

        path_str = str(file_path).lower()
        path_matches = sum(25 for kw in keywords if kw in path_str)

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as content_file:
                file_text = content_file.read().lower()

            distinct_matches = sum(1 for kw in keywords if kw in file_text)
            if distinct_matches == 0 and path_matches == 0:
                return 0

            frequency_hits = sum(file_text.count(kw) for kw in keywords)
            diversity_bonus = (distinct_matches ** 2) * 100
            return diversity_bonus + path_matches + frequency_hits
        except OSError:
            return 0

    def prioritize_files(self, files: list[Path], criteria: list[Criterion]) -> PrioritizedFiles:
        # Partition files into high and low priority based on relevance scores.
        keywords = self.keyword_extractor.extract_keywords(criteria)

        scored_files: list[ScoredFile] = []
        for file_path in files:
            score = self._calculate_file_relevance_score(file_path, keywords)
            scored_files.append(ScoredFile(file_path=file_path, relevance_score=score))

        scored_files.sort(key=lambda item: item.relevance_score, reverse=True)

        cutoff_score = 100 if len(keywords) > 1 else 1
        qualified_high = [item.file_path for item in scored_files if item.relevance_score >= cutoff_score]

        if len(qualified_high) > self.max_high_priority_files:
            high_priority = qualified_high[:self.max_high_priority_files]
            overflow = qualified_high[self.max_high_priority_files:]
        else:
            high_priority = qualified_high
            overflow = []

        low_priority = overflow + [
            item.file_path for item in scored_files if item.relevance_score < cutoff_score
        ]

        return PrioritizedFiles(
            high_priority_files=high_priority,
            low_priority_files=low_priority,
        )
