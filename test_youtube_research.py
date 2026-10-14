import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from youtube_research import (
    ResearchResult,
    TranscriptLine,
    Video,
    answer_question,
    build_documents,
    extract_video_id,
    format_timestamp,
    summarize_sections,
)


class VideoUrlTests(unittest.TestCase):
    def test_extracts_video_ids_from_supported_url_shapes(self):
        video_id = "dQw4w9WgXcQ"
        urls = [
            f"https://www.youtube.com/watch?v={video_id}",
            f"https://youtu.be/{video_id}",
            f"https://www.youtube.com/shorts/{video_id}",
            f"https://www.youtube.com/embed/{video_id}",
            f"https://www.youtube.com/live/{video_id}",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(extract_video_id(url), video_id)

    def test_rejects_non_video_urls(self):
        self.assertIsNone(extract_video_id("https://youtube.com/results?search_query=python"))
        self.assertIsNone(extract_video_id("not a URL"))


class TranscriptIndexTests(unittest.TestCase):
    def test_chunks_retain_exact_start_and_end_times(self):
        lines = [
            TranscriptLine("Opening", 2.25, 1.0),
            TranscriptLine("Details", 3.5, 2.0),
            TranscriptLine("Conclusion", 90.0, 3.0),
        ]

        documents = build_documents(lines, "dQw4w9WgXcQ", max_chars=40)

        self.assertEqual(len(documents), 3)
        self.assertEqual(documents[0].metadata["start"], 2.25)
        self.assertEqual(documents[0].metadata["end"], 3.25)
        self.assertIn("[00:02] Opening", documents[0].page_content)
        self.assertEqual(documents[2].metadata["start"], 90.0)

    def test_timestamp_formats_hours_when_needed(self):
        self.assertEqual(format_timestamp(65), "01:05")
        self.assertEqual(format_timestamp(3661), "01:01:01")


class AsyncWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_summarizes_each_section_asynchronously(self):
        lines = [
            TranscriptLine("First section", 0, 1),
            TranscriptLine("Second section", 300, 1),
        ]
        fake_model = RunnableLambda(lambda _: AIMessage(content="Section summary"))
        with patch("youtube_research.ChatOpenAI", return_value=fake_model):
            summaries = await summarize_sections(lines)

        self.assertEqual(summaries, ["Section summary", "Section summary"])

    async def test_answer_replaces_source_labels_with_exact_timestamp_links(self):
        class FakeStore:
            async def asimilarity_search(self, question, k):
                return [
                    Document(
                        page_content="[00:42] A transcript fact.",
                        metadata={"start": 42.7, "end": 50},
                    )
                ]

        result = ResearchResult(
            Video("dQw4w9WgXcQ", "Test video", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            [],
            FakeStore(),
        )
        fake_model = RunnableLambda(lambda _: AIMessage(content="A fact [S1]."))
        with patch("youtube_research.ChatOpenAI", return_value=fake_model):
            answer, sources = await answer_question(result, "What happened?")

        self.assertIn(
            "[00:42](https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s)",
            answer,
        )
        self.assertEqual(sources[0]["timestamp"], "00:42")


if __name__ == "__main__":
    unittest.main()
