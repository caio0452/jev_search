from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TextChunk:
    file_path: Path
    chunk_index: int
    content: str
    start_character: int
    end_character: int


@dataclass
class CriterionEvaluation:
    identifier: str
    score: float
    is_satisfied: bool


@dataclass
class ChunkMatch:
    file_path: Path
    chunk_index: int
    content: str
    aggregate_score: float
    evaluations: list[CriterionEvaluation] = field(default_factory=list)


@dataclass
class FileMatch:
    file_path: Path
    best_score: float
    total_matching_chunks: int
    matching_chunks: list[ChunkMatch] = field(default_factory=list)


@dataclass
class PrioritizedFiles:
    high_priority_files: list[Path]
    low_priority_files: list[Path]


@dataclass
class DirectoryScanReport:
    best_chunks: list[ChunkMatch]
    best_files: list[FileMatch]
    high_priority_matches: list[ChunkMatch]
    low_priority_matches: list[ChunkMatch]


@dataclass
class PhaseScanResult:
    phase_name: str
    matching_chunks: list[ChunkMatch]
    matching_files: list[FileMatch]
