import asyncio
import os

import streamlit as st
from dotenv import load_dotenv
from openai import APIConnectionError, APIError
from youtube_transcript_api._errors import YouTubeTranscriptApiException
from yt_dlp.utils import DownloadError

from youtube_research import (
    ResearchResult,
    Video,
    analyze_video,
    answer_question,
    extract_video_id,
    search_videos,
)

load_dotenv()

st.set_page_config(
    page_title="YouTube Research Assistant",
    page_icon="▶️",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container { max-width: 1120px; padding-top: 2.2rem; }
    .hero { padding: 1.4rem 1.6rem; border-radius: 18px;
            background: linear-gradient(115deg, #151d32, #253c61); color: white;
            margin-bottom: 1.5rem; }
    .hero p { color: #d8e1f2; margin-bottom: 0; }
    [data-testid="stMetric"] { background: #f5f7fb; border-radius: 12px; padding: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "research_result" not in st.session_state:
    st.session_state.research_result = None
if "answers" not in st.session_state:
    st.session_state.answers = []

st.markdown(
    """
    <div class="hero">
      <h1>▶️ YouTube Research Assistant</h1>
      <p>Turn a video into a timestamped briefing. Search, summarize, and ask questions grounded in its transcript.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Paste a YouTube link or search a topic.\n"
        "2. Choose a result and analyze its transcript.\n"
        "3. Browse section summaries or ask a question.\n"
        "4. Open citations at the exact moment in the video."
    )
    st.divider()
    st.caption("English transcripts are currently supported.")
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Add `OPENAI_API_KEY` to your `.env` file to enable analysis.")
    else:
        st.success("OpenAI is configured.")

with st.form("video_search"):
    query = st.text_input(
        "Video URL or topic",
        placeholder="Paste a YouTube URL or search, e.g. “how transformers work”",
        help="Use a YouTube video link or enter a topic to find videos.",
    )
    find_video = st.form_submit_button("Find videos", type="primary")

if find_video:
    st.session_state.candidates = []
    st.session_state.research_result = None
    st.session_state.answers = []
    video_id = extract_video_id(query)
    if video_id:
        st.session_state.candidates = [
            Video(
                video_id,
                "Selected YouTube video",
                f"https://www.youtube.com/watch?v={video_id}",
            )
        ]
    elif query.strip():
        try:
            with st.spinner("Searching YouTube..."):
                st.session_state.candidates = search_videos(query)
        except (ValueError, DownloadError) as error:
            st.error(f"Could not search YouTube: {error}")
    else:
        st.error("Enter a video URL or a topic to search.")

candidates: list[Video] = st.session_state.candidates
if candidates:
    st.subheader("Choose a video")
    if len(candidates) > 1:
        selected_index = st.selectbox(
            "Search results",
            options=range(len(candidates)),
            format_func=lambda index: candidates[index].title,
            label_visibility="collapsed",
        )
        selected_video = candidates[selected_index]
    else:
        selected_video = candidates[0]
        st.markdown(f"**{selected_video.title}**")

    st.video(selected_video.url)
    analyze_column, detail_column = st.columns([1, 3], vertical_alignment="center")
    with analyze_column:
        analyze_clicked = st.button("Analyze this video", type="primary", use_container_width=True)
    with detail_column:
        st.caption("Fetch transcript · build searchable index · summarize sections")

    if analyze_clicked:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Set `OPENAI_API_KEY` before analyzing a video.")
        else:
            try:
                with st.spinner(
                    "Fetching transcript, indexing passages, and summarizing sections..."
                ):
                    st.session_state.research_result = asyncio.run(
                        analyze_video(selected_video)
                    )
                st.session_state.answers = []
            except APIConnectionError:
                st.error(
                    "Could not connect securely to OpenAI. Check your network "
                    "connection and the system's trusted certificates, then retry."
                )
            except (ValueError, YouTubeTranscriptApiException, APIError) as error:
                st.error(f"Could not analyze this video: {error}")

result: ResearchResult | None = st.session_state.research_result
if result:
    st.divider()
    st.subheader(result.video.title)
    summary_tab, ask_tab = st.tabs(["Section summaries", "Ask questions"])

    with summary_tab:
        if result.summaries:
            st.caption(
                f"{len(result.summaries)} transcript sections · "
                "Select a section to review its key points."
            )
            for index, summary in enumerate(result.summaries, start=1):
                with st.expander(f"Section {index}", expanded=index == 1):
                    st.markdown(summary)
        else:
            st.info("No transcript sections were available to summarize.")

    with ask_tab:
        st.write("Answers are grounded in the transcript. Citations open at their source timestamp.")
        with st.form("question_form"):
            question = st.text_input(
                "Your question",
                placeholder="What are the main takeaways?",
            )
            ask = st.form_submit_button("Ask the video", type="primary")
        if ask:
            if not question.strip():
                st.error("Enter a question about the video.")
            else:
                try:
                    with st.spinner("Searching the transcript and drafting an answer..."):
                        answer, sources = asyncio.run(
                            answer_question(result, question.strip())
                        )
                    st.session_state.answers.insert(
                        0,
                        {
                            "question": question.strip(),
                            "answer": answer,
                            "sources": sources,
                        },
                    )
                except APIConnectionError:
                    st.error(
                        "Could not connect securely to OpenAI. Check your network "
                        "connection and the system's trusted certificates, then retry."
                    )
                except (ValueError, APIError) as error:
                    st.error(f"Could not answer that question: {error}")

        for entry in st.session_state.answers:
            st.markdown(f"**You asked:** {entry['question']}")
            st.markdown(entry["answer"])
            with st.expander("Transcript passages"):
                for source in entry["sources"]:
                    excerpt = source["text"].replace("\n", " ")
                    st.markdown(f"- [{source['timestamp']}]({source['url']}) — {excerpt}")
            st.divider()
