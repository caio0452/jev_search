import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from models import TextChunk, ChunkMatch, CriterionEvaluation, PrioritizedFiles, DirectoryScanReport
from criteria import Criterion, AndCondition, OrCondition, CriteriaExpressionParser
from file_discovery import FileFilter, KeywordExtractor, FilePrioritizer
from chunking import TextChunker
from client import OpenRouterDecisionsClient
from evaluator import ParallelChunkScanner
from scanner import DirectoryScanner
from progress import ProgressObserver, ProgressUpdate


class MockJevClient:
    def __init__(self, fixed_scores: dict[str, float] | None = None):
        self.fixed_scores = fixed_scores or {}
        self.call_history: list[str] = []

    def evaluate(self, state: str, question_instructions: dict[str, str]) -> dict[str, float]:
        self.call_history.append(state)
        return {
            identifier: self.fixed_scores.get(identifier, 0.9)
            for identifier in question_instructions
        }


class TestCriteria(unittest.TestCase):
    def test_single_criterion_evaluation(self):
        # Verify single criterion threshold check and score.
        crit = Criterion(identifier="c1", instruction="check database")
        self.assertTrue(crit.evaluate({"c1": 0.8}, threshold=0.7))
        self.assertFalse(crit.evaluate({"c1": 0.5}, threshold=0.7))
        self.assertEqual(crit.calculate_score({"c1": 0.85}), 0.85)

    def test_and_condition(self):
        # Test logical conjunction across multiple criteria.
        c1 = Criterion(identifier="c1", instruction="crit 1")
        c2 = Criterion(identifier="c2", instruction="crit 2")
        and_cond = AndCondition(c1, c2)

        self.assertTrue(and_cond.evaluate({"c1": 0.8, "c2": 0.9}, threshold=0.7))
        self.assertFalse(and_cond.evaluate({"c1": 0.8, "c2": 0.4}, threshold=0.7))
        self.assertEqual(and_cond.calculate_score({"c1": 0.8, "c2": 0.4}), 0.4)

    def test_or_condition(self):
        # Test logical disjunction across multiple criteria.
        c1 = Criterion(identifier="c1", instruction="crit 1")
        c2 = Criterion(identifier="c2", instruction="crit 2")
        or_cond = OrCondition(c1, c2)

        self.assertTrue(or_cond.evaluate({"c1": 0.8, "c2": 0.3}, threshold=0.7))
        self.assertFalse(or_cond.evaluate({"c1": 0.2, "c2": 0.4}, threshold=0.7))
        self.assertEqual(or_cond.calculate_score({"c1": 0.8, "c2": 0.3}), 0.8)

    def test_criteria_expression_parser(self):
        # Test parsing boolean expression into AST.
        parser = CriteriaExpressionParser()
        parsed_condition = parser.parse("database error AND timeout OR critical failure")
        self.assertIsInstance(parsed_condition, OrCondition)
        all_criteria = parsed_condition.collect_criteria()
        self.assertEqual(len(all_criteria), 3)

    def test_parentheses_and_quotes_parsing(self):
        # Test parsing complex boolean expressions with parentheses and quotes.
        parser = CriteriaExpressionParser()
        parsed_condition = parser.parse('("WebSocket streaming" OR "SSE handler") AND "chat UI"')
        self.assertIsInstance(parsed_condition, AndCondition)
        all_criteria = parsed_condition.collect_criteria()
        self.assertEqual(len(all_criteria), 3)
        instructions = [c.instruction for c in all_criteria]
        self.assertIn("WebSocket streaming", instructions)
        self.assertIn("SSE handler", instructions)
        self.assertIn("chat UI", instructions)


class TestChunker(unittest.TestCase):
    def test_text_chunking(self):
        # Verify text chunking with boundaries and overlap.
        chunker = TextChunker(chunk_size=50, overlap=10)
        sample_text = "Line 1.\nLine 2.\nLine 3.\nLine 4.\nLine 5.\nLine 6.\nLine 7."
        chunks = chunker.split_text(Path("dummy.txt"), sample_text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIsInstance(chunk, TextChunk)
            self.assertLessEqual(len(chunk.content), 60)


class TestFileDiscovery(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_file_filter_and_prioritization(self):
        # Test file discovery and keyword-based prioritization.
        text_file = self.test_dir / "app.py"
        text_file.write_text("import sqlite3\nconn = sqlite3.connect('test.db')")

        other_file = self.test_dir / "notes.txt"
        other_file.write_text("just some random text without keywords")

        binary_file = self.test_dir / "binary.dat"
        binary_file.write_bytes(b"\x00\x01\x02")

        file_filter = FileFilter()
        discovered = file_filter.discover_files(self.test_dir)

        self.assertIn(text_file, discovered)
        self.assertIn(other_file, discovered)
        self.assertNotIn(binary_file, discovered)

        prioritizer = FilePrioritizer()
        criterion = Criterion(identifier="c1", instruction="find sqlite3 database connection")
        prioritized = prioritizer.prioritize_files(discovered, [criterion])

        self.assertIsInstance(prioritized, PrioritizedFiles)
        self.assertIn(text_file, prioritized.high_priority_files)
        self.assertIn(other_file, prioritized.low_priority_files)


class TestDirectoryScanner(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_scanner_orchestration(self):
        # Test end-to-end directory scanning and reporting.
        file1 = self.test_dir / "match.py"
        file1.write_text("database authentication token logic")

        file2 = self.test_dir / "other.txt"
        file2.write_text("unrelated general documentation")

        mock_client = MockJevClient()
        file_filter = FileFilter()
        file_prioritizer = FilePrioritizer()
        chunker = TextChunker(chunk_size=100)
        chunk_scanner = ParallelChunkScanner(client=mock_client, concurrency_limit=2, threshold=0.5)

        scanner = DirectoryScanner(
            file_filter=file_filter,
            file_prioritizer=file_prioritizer,
            chunker=chunker,
            chunk_scanner=chunk_scanner,
        )

        criterion = Criterion(identifier="c1", instruction="authentication token")
        report = scanner.scan_directory(self.test_dir, criterion)

        self.assertIsInstance(report, DirectoryScanReport)
        self.assertGreaterEqual(len(report.best_chunks), 1)
        self.assertGreaterEqual(len(report.best_files), 1)
        self.assertEqual(report.best_files[0].file_path, file1)


class TestOpenRouterDecisionsClient(unittest.TestCase):
    def test_evaluate_successful_response(self):
        # Test OpenRouter HTTP client response parsing and score mapping.
        mock_response_body = {
            "model": "~typesafe/jev-latest",
            "answers": {
                "urgency": {
                    "type": "noul",
                    "noul": 0.95
                }
            }
        }
        client = OpenRouterDecisionsClient(api_key="test_key")
        if client._session is not None:
            with patch.object(client._session, "post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.text = json.dumps(mock_response_body)
                mock_resp.json.return_value = mock_response_body
                mock_post.return_value = mock_resp
                scores = client.evaluate(
                    state="Urgent bug reported in production!",
                    question_instructions={"urgency": "Does this message express urgency?"}
                )
        else:
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = json.dumps(mock_response_body).encode("utf-8")
                mock_response.__enter__.return_value = mock_response
                mock_urlopen.return_value = mock_response
                scores = client.evaluate(
                    state="Urgent bug reported in production!",
                    question_instructions={"urgency": "Does this message express urgency?"}
                )

        self.assertIn("urgency", scores)
        self.assertAlmostEqual(scores["urgency"], 0.95)


class TestProgressObserver(unittest.TestCase):
    def test_progress_tracking(self):
        # Verify progress observer receives events during chunk scanning.
        class SpyObserver(ProgressObserver):
            def __init__(self):
                self.started_total = 0
                self.completed_count = 0
                self.finished_matches = 0

            def on_phase_start(self, phase_name: str, total: int) -> None:
                self.started_total = total

            def on_chunk_completed(self, update: ProgressUpdate) -> None:
                self.completed_count = update.completed

            def on_phase_complete(self, phase_name: str, matches_found: int) -> None:
                self.finished_matches = matches_found

        spy = SpyObserver()
        scanner = ParallelChunkScanner(client=MockJevClient(), concurrency_limit=2)
        chunks = [TextChunk(Path("test.txt"), 0, "content", 0, 7)]
        crit = Criterion("c1", "test")
        scanner.scan_chunks(chunks, crit, phase_name="TestPhase", progress_observer=spy)

        self.assertEqual(spy.started_total, 1)
        self.assertEqual(spy.completed_count, 1)
        self.assertEqual(spy.finished_matches, 1)


if __name__ == "__main__":
    unittest.main()
