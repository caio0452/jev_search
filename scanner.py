from pathlib import Path
from models import DirectoryScanReport, FileMatch, ChunkMatch, PrioritizedFiles, TextChunk, PhaseScanResult
from criteria import Condition
from file_discovery import FileFilter, FilePrioritizer
from chunking import TextChunker
from evaluator import ParallelChunkScanner
from progress import ProgressObserver, PhaseResultObserver


class DirectoryScanner:
    def __init__(
        self,
        file_filter: FileFilter,
        file_prioritizer: FilePrioritizer,
        chunker: TextChunker,
        chunk_scanner: ParallelChunkScanner,
        progress_observer: ProgressObserver | None = None,
        phase_result_observer: PhaseResultObserver | None = None,
    ):
        self.file_filter = file_filter
        self.file_prioritizer = file_prioritizer
        self.chunker = chunker
        self.chunk_scanner = chunk_scanner
        self.progress_observer = progress_observer
        self.phase_result_observer = phase_result_observer

    def _collect_chunks_from_files(self, files: list[Path]) -> list[TextChunk]:
        collected_chunks: list[TextChunk] = []
        for file_path in files:
            chunks = self.chunker.chunk_file(file_path)
            collected_chunks.extend(chunks)
        return collected_chunks

    def _build_file_matches(self, chunk_matches: list[ChunkMatch]) -> list[FileMatch]:
        # Group matched chunks by file and calculate best scores.
        chunks_by_file: dict[Path, list[ChunkMatch]] = {}
        for match in chunk_matches:
            chunks_by_file.setdefault(match.file_path, []).append(match)

        file_matches: list[FileMatch] = []
        for file_path, matches in chunks_by_file.items():
            best_score = max(match.aggregate_score for match in matches)
            sorted_chunks = sorted(matches, key=lambda item: item.aggregate_score, reverse=True)
            file_matches.append(
                FileMatch(
                    file_path=file_path,
                    best_score=best_score,
                    total_matching_chunks=len(matches),
                    matching_chunks=sorted_chunks,
                )
            )

        file_matches.sort(key=lambda item: item.best_score, reverse=True)
        return file_matches

    def scan_directory(self, directory_path: Path, condition: Condition) -> DirectoryScanReport:
        # Run prioritized two-pass directory search against condition.
        discovered_files = self.file_filter.discover_files(directory_path)
        all_criteria = condition.collect_criteria()
        prioritized_files: PrioritizedFiles = self.file_prioritizer.prioritize_files(
            discovered_files,
            all_criteria,
        )

        keywords = self.file_prioritizer.keyword_extractor.extract_keywords(all_criteria)

        high_priority_chunks = self._collect_chunks_from_files(prioritized_files.high_priority_files)
        prioritized_hp_chunks = self.chunker.prioritize_chunks(high_priority_chunks, keywords)
        high_priority_matches = self.chunk_scanner.scan_chunks(
            prioritized_hp_chunks,
            condition,
            phase_name="High Priority Files",
            progress_observer=self.progress_observer,
        )

        high_priority_files = self._build_file_matches(high_priority_matches)
        if self.phase_result_observer is not None:
            self.phase_result_observer.on_phase_results(
                PhaseScanResult(
                    phase_name="High Priority Files",
                    matching_chunks=high_priority_matches,
                    matching_files=high_priority_files,
                )
            )

        low_priority_chunks = self._collect_chunks_from_files(prioritized_files.low_priority_files)
        prioritized_lp_chunks = self.chunker.prioritize_chunks(low_priority_chunks, keywords)
        low_priority_matches = self.chunk_scanner.scan_chunks(
            prioritized_lp_chunks,
            condition,
            phase_name="Low Priority Files",
            progress_observer=self.progress_observer,
        )

        low_priority_files = self._build_file_matches(low_priority_matches)
        if self.phase_result_observer is not None:
            self.phase_result_observer.on_phase_results(
                PhaseScanResult(
                    phase_name="Low Priority Files",
                    matching_chunks=low_priority_matches,
                    matching_files=low_priority_files,
                )
            )

        all_matches = high_priority_matches + low_priority_matches
        all_matches.sort(key=lambda item: item.aggregate_score, reverse=True)

        best_files = self._build_file_matches(all_matches)

        return DirectoryScanReport(
            best_chunks=all_matches,
            best_files=best_files,
            high_priority_matches=high_priority_matches,
            low_priority_matches=low_priority_matches,
        )
