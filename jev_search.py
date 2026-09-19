import argparse
import sys
from pathlib import Path
from models import DirectoryScanReport, PhaseScanResult
from criteria import CriteriaExpressionParser
from file_discovery import FileFilter, FilePrioritizer
from chunking import TextChunker
from client import OpenRouterDecisionsClient
from evaluator import ParallelChunkScanner
from scanner import DirectoryScanner
from progress import ConsoleProgressReporter, PhaseResultObserver


class FileReportWriter:
    def __init__(self, output_path: Path):
        self.output_path = output_path

    def write_header(self, target_folder: Path, criteria: str) -> None:
        # Write initial metadata to output file.
        with open(self.output_path, "w", encoding="utf-8") as file_stream:
            file_stream.write("Search Results Report\n")
            file_stream.write(f"Target Directory: {target_folder}\n")
            file_stream.write(f"Criteria: {criteria}\n")
            file_stream.write("=" * 60 + "\n\n")

    def append_phase_result(self, phase_result: PhaseScanResult) -> None:
        # Append phase results to output file.
        with open(self.output_path, "a", encoding="utf-8") as file_stream:
            file_stream.write(f"\n--- Results for {phase_result.phase_name} ---\n")
            file_stream.write(f"Matching Files Count: {len(phase_result.matching_files)}\n")
            file_stream.write(f"Matching Chunks Count: {len(phase_result.matching_chunks)}\n\n")

            if phase_result.matching_files:
                file_stream.write("Files:\n")
                for file_match in phase_result.matching_files:
                    file_stream.write(
                        f" - {file_match.file_path} (Score: {file_match.best_score:.3f}, Chunks: {file_match.total_matching_chunks})\n"
                    )

                file_stream.write("\nChunks:\n")
                for rank, chunk in enumerate(phase_result.matching_chunks, start=1):
                    file_stream.write(f"\n[Match #{rank}] File: {chunk.file_path} (Score: {chunk.aggregate_score:.3f})\n")
                    file_stream.write("-" * 50 + "\n")
                    file_stream.write(chunk.content.strip() + "\n")
                    file_stream.write("-" * 50 + "\n")


class ReportPresenter(PhaseResultObserver):
    def __init__(self, file_writer: FileReportWriter | None = None):
        self.file_writer = file_writer

    def on_phase_results(self, phase_result: PhaseScanResult) -> None:
        # Output results immediately to terminal and file upon phase completion.
        print(f"\n\n=== Immediate Results: {phase_result.phase_name} ===")
        print(f"Matching files found: {len(phase_result.matching_files)}")
        print(f"Matching chunks found: {len(phase_result.matching_chunks)}")

        if phase_result.matching_files:
            print("\nTop Matching Files:")
            for file_match in phase_result.matching_files[:10]:
                print(
                    f" - {file_match.file_path} "
                    f"(Score: {file_match.best_score:.3f}, Chunks: {file_match.total_matching_chunks})"
                )

            print("\nTop Matching Chunks:")
            for rank, chunk in enumerate(phase_result.matching_chunks[:5], start=1):
                print(f"\n[Match #{rank}] File: {chunk.file_path} (Score: {chunk.aggregate_score:.3f})")
                print("=" * 60)
                print(chunk.content.strip())
                print("=" * 60)
        else:
            print(f"No matches found in {phase_result.phase_name}.")

        if self.file_writer is not None:
            self.file_writer.append_phase_result(phase_result)

    def display_final_summary(self, report: DirectoryScanReport) -> None:
        # Display aggregated summary across all search phases.
        print("\n=== Search Complete ===")
        print(f"Total matching files: {len(report.best_files)}")
        print(f"Total matching chunks: {len(report.best_chunks)}")
        if self.file_writer is not None:
            print(f"Results written to: {self.file_writer.output_path}")


class SearchApplication:
    def __init__(
        self,
        folder_path: Path,
        criteria_expression: str,
        chunk_size: int = 2000,
        concurrency_limit: int = 30,
        threshold: float = 0.7,
        allowed_extensions: set[str] | None = None,
        api_key: str | None = None,
        top_files: int = 25,
        output_file_path: Path | None = None,
    ):
        self.folder_path = folder_path
        self.criteria_expression = criteria_expression
        self.chunk_size = chunk_size
        self.concurrency_limit = concurrency_limit
        self.threshold = threshold
        self.allowed_extensions = allowed_extensions
        self.api_key = api_key
        self.top_files = top_files
        self.output_file_path = output_file_path

    def execute(self) -> DirectoryScanReport:
        # Configure scanner and observers to report and write phase results.
        condition_parser = CriteriaExpressionParser()
        condition = condition_parser.parse(self.criteria_expression)

        file_writer: FileReportWriter | None = None
        if self.output_file_path is not None:
            file_writer = FileReportWriter(self.output_file_path)
            file_writer.write_header(self.folder_path, self.criteria_expression)

        presenter = ReportPresenter(file_writer=file_writer)
        progress_reporter = ConsoleProgressReporter()

        file_filter = FileFilter(allowed_extensions=self.allowed_extensions)
        file_prioritizer = FilePrioritizer(max_high_priority_files=self.top_files)
        text_chunker = TextChunker(chunk_size=self.chunk_size)
        api_client = OpenRouterDecisionsClient(api_key=self.api_key, pool_size=self.concurrency_limit)
        chunk_scanner = ParallelChunkScanner(
            client=api_client,
            concurrency_limit=self.concurrency_limit,
            threshold=self.threshold,
        )

        scanner = DirectoryScanner(
            file_filter=file_filter,
            file_prioritizer=file_prioritizer,
            chunker=text_chunker,
            chunk_scanner=chunk_scanner,
            progress_observer=progress_reporter,
            phase_result_observer=presenter,
        )

        report = scanner.scan_directory(self.folder_path, condition)
        presenter.display_final_summary(report)
        return report


class CliArgumentParser:
    def parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Scan a directory for text matching criteria using Jev.")
        parser.add_argument("folder", type=str, help="Folder path to scan")
        parser.add_argument("criteria", type=str, help="Criteria expression (supports AND, OR, parentheses)")
        parser.add_argument("--chunk-size", type=int, default=2000, help="Chunk size in characters")
        parser.add_argument("--workers", type=int, default=30, help="Number of concurrent requests (default: 30)")
        parser.add_argument("--threshold", type=float, default=0.7, help="Minimum matching score threshold")
        parser.add_argument("--extensions", nargs="+", default=None, help="File extensions to include")
        parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (defaults to OPENROUTER_API_KEY env var)")
        parser.add_argument("--top-files", type=int, default=25, help="Number of most promising files to scan in high priority (default: 25)")
        parser.add_argument("--output", type=str, default="search_results.txt", help="Output text file path (default: search_results.txt)")
        return parser.parse_args()


def main() -> None:
    argument_parser = CliArgumentParser()
    args = argument_parser.parse_arguments()

    folder_path = Path(args.folder)
    if not folder_path.is_dir():
        print(f"Error: Provided path '{args.folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    extensions = set(args.extensions) if args.extensions else None
    output_path = Path(args.output) if args.output else None

    app = SearchApplication(
        folder_path=folder_path,
        criteria_expression=args.criteria,
        chunk_size=args.chunk_size,
        concurrency_limit=args.workers,
        threshold=args.threshold,
        allowed_extensions=extensions,
        api_key=args.api_key,
        top_files=args.top_files,
        output_file_path=output_path,
    )

    app.execute()


if __name__ == "__main__":
    main()
