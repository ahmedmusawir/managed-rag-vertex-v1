#!/usr/bin/env python3
"""
LESSON 2: Create a RagCorpus

This script initializes Vertex AI, creates a new RAG corpus, and saves the
returned corpus resource name to `corpus_name.txt` in the project root.

How the API call works:
- `vertexai.init(...)` sets the project and location for the Vertex SDK
- `rag.create_corpus(...)` creates the persistent corpus container
- the installed SDK supports a minimal corpus creation flow with the default
  backend configuration
- the returned `corpus.name` is the full resource name needed by later lessons

This matches the old tutorial flow where a store was created first and its name
was persisted for later scripts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import vertexai
from dotenv import load_dotenv
ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_NAME_PATH = ROOT_DIR / "corpus_name.txt"


def get_required_setting(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value

    joined = ", ".join(names)
    print(f"Missing required environment variable. Checked: {joined}")
    print("Create a .env file from .env.example and set one of those values before retrying.")
    sys.exit(1)


def main() -> None:
    load_dotenv()

    print("LESSON 2: Create a RagCorpus\n")

    project_id = get_required_setting("VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
    location = get_required_setting("VERTEX_LOCATION", "GOOGLE_CLOUD_REGION")
    display_name = os.getenv("VERTEX_CORPUS_DISPLAY_NAME", "Tony-Test-Corpus").strip()

    print("Step 1: Initializing Vertex AI...")
    vertexai.init(project=project_id, location=location)
    print("Vertex AI initialized.")

    print("\nStep 2: Creating the corpus...")
    from vertexai import rag

    corpus = rag.create_corpus(display_name=display_name)
    print("Corpus created.")

    print("\nWhat did we get back?")
    print(f"Corpus Name: {corpus.name}")
    print(f"Display Name: {corpus.display_name}")

    CORPUS_NAME_PATH.write_text(f"{corpus.name}\n", encoding="utf-8")
    print(f"\nSaved corpus name to: {CORPUS_NAME_PATH.name}")

    print("\nWhat did we learn?")
    print("- A RagCorpus is the container for your indexed documents.")
    print("- The installed SDK works with the minimal `rag.create_corpus(...)` flow.")
    print("- `corpus.name` is the full resource path that later lessons must use.")
    print("- `corpus_name.txt` is the handoff file between lessons.")


if __name__ == "__main__":
    main()
