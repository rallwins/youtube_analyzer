from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL


@dataclass(frozen=True)
class Video:
    video_id: str
    title: str
    url: str


@dataclass(frozen=True)
class TranscriptLine:
    text: str
    start: float
    duration: float


@dataclass
class ResearchResult:
    video: Video
    summaries: list[str]
    vector_store: InMemoryVectorStore


def extract_video_id(value: str) -> str | None:
    parsed = urlparse(value.strip())
    host = parsed.netloc.lower().split(":")[0]
    video_id: str | None = None

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
                video_id = parts[1]

    if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        return video_id
    return None


def search_videos(topic: str, limit: int = 5) -> list[Video]:
    query = topic.strip()
    if not query:
        raise ValueError("Enter a topic to search for.")

    with YoutubeDL(
        {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    ) as downloader:
        search_results = downloader.extract_info(
            f"ytsearch{limit}:{query}", download=False
        )

    videos = []
    for entry in (search_results or {}).get("entries", []):
        video_id = entry.get("id")
        if video_id:
            videos.append(
                Video(
                    video_id=video_id,
                    title=entry.get("title") or "Untitled video",
                    url=f"https://www.youtube.com/watch?v={video_id}",
                )
            )
    if not videos:
        raise ValueError("No YouTube videos were found for that topic.")
    return videos


def fetch_transcript(video_id: str) -> list[TranscriptLine]:
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    lines = [
        TranscriptLine(
            text=snippet.text.strip(),
            start=float(snippet.start),
            duration=float(snippet.duration),
        )
        for snippet in transcript
        if snippet.text.strip()
    ]
    if not lines:
        raise ValueError("The video has no usable English transcript.")
    return lines


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def build_documents(
    lines: Iterable[TranscriptLine], video_id: str, max_chars: int = 1200
) -> list[Document]:
    documents: list[Document] = []
    chunk: list[TranscriptLine] = []
    chunk_chars = 0

    def save_chunk() -> None:
        if not chunk:
            return
        start = chunk[0].start
        end = chunk[-1].start + chunk[-1].duration
        content = "\n".join(
            f"[{format_timestamp(line.start)}] {line.text}" for line in chunk
        )
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "video_id": video_id,
                    "start": start,
                    "end": end,
                },
            )
        )

    for line in lines:
        line_chars = len(line.text) + 16
        if chunk and chunk_chars + line_chars > max_chars:
            save_chunk()
            chunk = []
            chunk_chars = 0
        chunk.append(line)
        chunk_chars += line_chars
    save_chunk()
    return documents


def _build_summary_sections(
    lines: list[TranscriptLine], section_seconds: float = 240
) -> list[str]:
    sections: list[str] = []
    current_lines: list[str] = []
    section_start: float | None = None

    for line in lines:
        if section_start is None:
            section_start = line.start
        elif line.start - section_start >= section_seconds and current_lines:
            sections.append("\n".join(current_lines))
            current_lines = []
            section_start = line.start
        current_lines.append(f"[{format_timestamp(line.start)}] {line.text}")

    if current_lines:
        sections.append("\n".join(current_lines))
    return sections


async def summarize_sections(
    lines: list[TranscriptLine], model: str = "gpt-4o-mini"
) -> list[str]:
    sections = _build_summary_sections(lines)
    if not sections:
        return []

    llm = ChatOpenAI(model=model, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this section of a YouTube transcript in concise bullet points. "
                "Keep important details and include the provided timestamps for key claims. "
                "Do not add facts that are not in the transcript.",
            ),
            ("human", "{transcript}"),
        ]
    )
    chain = prompt | llm
    semaphore = asyncio.Semaphore(5)

    async def summarize(section: str) -> str:
        async with semaphore:
            response = await chain.ainvoke({"transcript": section})
            return str(response.content)

    return await asyncio.gather(*(summarize(section) for section in sections))


async def analyze_video(
    video: Video,
    model: str = "gpt-4o-mini",
) -> ResearchResult:
    lines = await asyncio.to_thread(fetch_transcript, video.video_id)
    documents = build_documents(lines, video.video_id)
    if not documents:
        raise ValueError("The video transcript could not be split into searchable text.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)
    await vector_store.aadd_documents(documents)
    summaries = await summarize_sections(lines, model=model)
    return ResearchResult(video, summaries, vector_store)


async def answer_question(
    result: ResearchResult,
    question: str,
    model: str = "gpt-4o-mini",
) -> tuple[str, list[dict[str, Any]]]:
    matches = await result.vector_store.asimilarity_search(question, k=5)
    if not matches:
        raise ValueError("No transcript passages matched that question.")

    sources = []
    context_parts = []
    for index, document in enumerate(matches, start=1):
        start = float(document.metadata["start"])
        seconds = int(start)
        citation_url = f"{result.video.url}&t={seconds}s"
        source = {
            "label": f"S{index}",
            "timestamp": format_timestamp(start),
            "url": citation_url,
            "text": document.page_content,
        }
        sources.append(source)
        context_parts.append(
            f"[{source['label']}] Timestamp: {source['timestamp']}\n"
            f"URL: {citation_url}\n{document.page_content}"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's question using only the supplied transcript passages. "
                "Cite every factual claim with one or more source labels exactly as provided, "
                "for example [S1]. If the transcript does not contain the answer, say so. "
                "Never invent timestamps, quotes, or details.",
            ),
            (
                "human",
                "Transcript passages:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )
    response = await (prompt | ChatOpenAI(model=model, temperature=0)).ainvoke(
        {"context": "\n\n".join(context_parts), "question": question}
    )
    answer = str(response.content)
    for source in sources:
        answer = answer.replace(
            f"[{source['label']}]",
            f"[{source['timestamp']}]({source['url']})",
        )
    return answer, sources
