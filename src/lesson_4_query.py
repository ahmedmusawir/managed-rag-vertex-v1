#!/usr/bin/env python3
"""
LESSON 4: Ask a Grounded Question with Vertex RAG

This script reads the saved corpus resource name from `corpus_name.txt`,
builds a retrieval tool that points at that corpus, and asks Gemini a test
question using Retrieval-Augmented Generation.

How the API call works:
- `load_dotenv()` loads the project and location used for `vertexai.init(...)`
- `corpus_name.txt` provides the full corpus resource path created in Lesson 2
- `rag.RagResource(...)` identifies which corpus the retrieval tool should use
- `rag.VertexRagStore(...)` and `rag.RagRetrievalConfig(...)` configure
  grounded retrieval from that corpus
- `Tool.from_retrieval(...)` converts the retrieval config into a Gemini tool
- `GenerativeModel(...)` attaches that tool to the model
- `generate_content(...)` asks a hardcoded question and lets the model answer
  using the retrieved corpus context

This lesson is generation, not raw search. Gemini is expected to read from the
corpus first and then produce an answer grounded in the imported documents.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import vertexai
from dotenv import load_dotenv
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool

ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_NAME_PATH = ROOT_DIR / "corpus_name.txt"
TEST_QUESTION = "What are the key points in the uploaded documents?"


def get_required_setting(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value

    joined = ", ".join(names)
    print(f"Missing required environment variable. Checked: {joined}")
    print("Create a .env file from .env.example and set one of those values before retrying.")
    sys.exit(1)


def read_corpus_name() -> str:
    if not CORPUS_NAME_PATH.exists():
        print("Missing corpus_name.txt")
        print("Run `python src/lesson_2_create_corpus.py` first.")
        sys.exit(1)

    corpus_name = CORPUS_NAME_PATH.read_text(encoding="utf-8").strip()
    if not corpus_name:
        print("corpus_name.txt is empty")
        print("Run `python src/lesson_2_create_corpus.py` again to regenerate it.")
        sys.exit(1)

    return corpus_name


def main() -> None:
    load_dotenv()

    print("LESSON 4: Ask a Grounded Question with Vertex RAG\n")

    project_id = get_required_setting("VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
    location = get_required_setting("VERTEX_LOCATION", "GOOGLE_CLOUD_REGION")
    corpus_name = read_corpus_name()

    print("Step 1: Initializing Vertex AI...")
    vertexai.init(project=project_id, location=location)
    print("Vertex AI initialized.")

    print("\nStep 2: Building the RAG retrieval tool...")
    rag_retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
                rag_retrieval_config=rag.RagRetrievalConfig(top_k=3),
            )
        )
    )
    print("RAG retrieval tool created.")

    print("\nStep 3: Creating the Gemini model...")
    rag_model = GenerativeModel(
        model_name="gemini-2.0-flash-001",
        tools=[rag_retrieval_tool],
    )
    print("Gemini model created with the retrieval tool attached.")

    print("\nStep 4: Asking the test question...")
    print(f"Question: {TEST_QUESTION}")
    response = rag_model.generate_content(TEST_QUESTION)

    print("\nAnswer:")
    print(response.text)

    print("\nWhat did we learn?")
    print("- Lesson 4 uses Retrieval-Augmented Generation, not raw retrieval.")
    print("- The retrieval tool points Gemini at the saved RagCorpus.")
    print("- Gemini answers the question using the corpus context.")


if __name__ == "__main__":
    main()
