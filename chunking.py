from pathlib import Path
from models import TextChunk


class TextChunker:
    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_file(self, file_path: Path) -> list[TextChunk]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file_stream:
                content = file_stream.read()
            return self.split_text(file_path, content)
        except OSError:
            return []

    def split_text(self, file_path: Path, text: str) -> list[TextChunk]:
        # Slice text into overlapping chunks respecting boundaries.
        chunks: list[TextChunk] = []
        text_length = len(text)
        if text_length == 0:
            return chunks

        current_start = 0
        chunk_index = 0

        while current_start < text_length:
            current_end = min(current_start + self.chunk_size, text_length)

            if current_end < text_length:
                search_boundary_start = max(current_start, current_end - 200)
                boundary_snippet = text[search_boundary_start:current_end]
                newline_offset = boundary_snippet.rfind("\n")
                space_offset = boundary_snippet.rfind(" ")

                if newline_offset != -1:
                    current_end = search_boundary_start + newline_offset + 1
                elif space_offset != -1:
                    current_end = search_boundary_start + space_offset + 1

            chunk_text_slice = text[current_start:current_end]
            if chunk_text_slice.strip():
                chunks.append(
                    TextChunk(
                        file_path=file_path,
                        chunk_index=chunk_index,
                        content=chunk_text_slice,
                        start_character=current_start,
                        end_character=current_end,
                    )
                )
                chunk_index += 1

            next_start = current_end - self.overlap
            if next_start <= current_start:
                next_start = current_end
            current_start = next_start

        return chunks

    def prioritize_chunks(self, chunks: list[TextChunk], keywords: set[str]) -> list[TextChunk]:
        # Sort chunks prioritizing those containing matching keywords.
        if not keywords:
            return chunks

        def calculate_chunk_keyword_weight(chunk: TextChunk) -> int:
            chunk_text = chunk.content.lower()
            distinct_matches = sum(1 for kw in keywords if kw in chunk_text)
            frequency = sum(chunk_text.count(kw) for kw in keywords)
            return (distinct_matches * 100) + frequency

        return sorted(chunks, key=calculate_chunk_keyword_weight, reverse=True)
