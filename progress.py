import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from models import PhaseScanResult


@dataclass
class ProgressUpdate:
    phase_name: str
    completed: int
    total: int
    matches_count: int
    errors_count: int = 0


class ProgressObserver(ABC):
    @abstractmethod
    def on_phase_start(self, phase_name: str, total: int) -> None:
        pass

    @abstractmethod
    def on_chunk_completed(self, update: ProgressUpdate) -> None:
        pass

    @abstractmethod
    def on_phase_complete(self, phase_name: str, matches_found: int) -> None:
        pass


class PhaseResultObserver(ABC):
    @abstractmethod
    def on_phase_results(self, phase_result: PhaseScanResult) -> None:
        pass


class ConsoleProgressReporter(ProgressObserver):
    def on_phase_start(self, phase_name: str, total: int) -> None:
        # Announce the start of a scanning phase.
        print(f"\n[{phase_name}] Starting scan of {total} chunks...")

    def on_chunk_completed(self, update: ProgressUpdate) -> None:
        # Display chunk processing progress percentage, matches, and error counts.
        if update.total == 0:
            return
        percentage = (update.completed / update.total) * 100
        error_suffix = f" | Errors: {update.errors_count}" if update.errors_count > 0 else ""
        progress_text = (
            f"\r[{update.phase_name}] Progress: {update.completed}/{update.total} "
            f"({percentage:.1f}%) | Matches: {update.matches_count}{error_suffix}"
        )
        sys.stdout.write(progress_text)
        sys.stdout.flush()

    def on_phase_complete(self, phase_name: str, matches_found: int) -> None:
        # Conclude phase with total matches summary.
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(f"[{phase_name}] Completed. Matches found: {matches_found}")
