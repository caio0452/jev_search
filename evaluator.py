import concurrent.futures
from models import TextChunk, ChunkMatch, CriterionEvaluation
from criteria import Condition, Criterion
from client import OpenRouterDecisionsClient
from progress import ProgressObserver, ProgressUpdate


class ParallelChunkScanner:
    def __init__(self, client: OpenRouterDecisionsClient, concurrency_limit: int = 10, threshold: float = 0.7):
        self.client = client
        self.concurrency_limit = concurrency_limit
        self.threshold = threshold

    def _evaluate_single_chunk(
        self,
        chunk: TextChunk,
        criteria: list[Criterion],
        condition: Condition,
    ) -> ChunkMatch | None:
        # Query Jev and verify if chunk satisfies criteria.
        question_instructions = {
            criterion.identifier: criterion.instruction
            for criterion in criteria
        }

        scores = self.client.evaluate(chunk.content, question_instructions)
        is_matched = condition.evaluate(scores, self.threshold)
        if not is_matched:
            return None

        aggregate_score = condition.calculate_score(scores)

        evaluations: list[CriterionEvaluation] = [
            CriterionEvaluation(
                identifier=criterion.identifier,
                score=scores.get(criterion.identifier, 0.0),
                is_satisfied=scores.get(criterion.identifier, 0.0) >= self.threshold,
            )
            for criterion in criteria
        ]

        return ChunkMatch(
            file_path=chunk.file_path,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            aggregate_score=aggregate_score,
            evaluations=evaluations,
        )

    def scan_chunks(
        self,
        chunks: list[TextChunk],
        condition: Condition,
        phase_name: str = "Scanning",
        progress_observer: ProgressObserver | None = None,
    ) -> list[ChunkMatch]:
        # Evaluate multiple chunks in parallel using thread pool.
        if not chunks:
            return []

        total_chunks = len(chunks)
        if progress_observer is not None:
            progress_observer.on_phase_start(phase_name, total_chunks)

        criteria = condition.collect_criteria()
        matching_chunks: list[ChunkMatch] = []
        completed_count = 0
        error_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency_limit) as executor:
            future_to_chunk = {
                executor.submit(self._evaluate_single_chunk, chunk, criteria, condition): chunk
                for chunk in chunks
            }

            for future in concurrent.futures.as_completed(future_to_chunk):
                completed_count += 1
                try:
                    match_result = future.result()
                    if match_result is not None:
                        matching_chunks.append(match_result)
                except (PermissionError, ValueError):
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception:
                    error_count += 1

                if progress_observer is not None:
                    progress_observer.on_chunk_completed(
                        ProgressUpdate(
                            phase_name=phase_name,
                            completed=completed_count,
                            total=total_chunks,
                            matches_count=len(matching_chunks),
                            errors_count=error_count,
                        )
                    )

        if progress_observer is not None:
            progress_observer.on_phase_complete(phase_name, len(matching_chunks))

        matching_chunks.sort(key=lambda item: item.aggregate_score, reverse=True)
        return matching_chunks
