#!/usr/bin/env python3
"""
LESSON 3: Import Files into the RagCorpus

This script reads the saved corpus resource name from `corpus_name.txt`,
scans the local `uploads/` directory for files, and imports those files into the
existing Vertex AI RAG corpus.

How the API call works:
- `load_dotenv()` loads the project and location used for `vertexai.init(...)`
- `corpus_name.txt` provides the full corpus resource path created in Lesson 2
- `Path.iterdir()` finds the files sitting in `uploads/`
- `rag.import_files(...)` uploads those files into the corpus
- `rag.TransformationConfig(...)` applies chunking during ingestion
- `rag.ChunkingConfig(chunk_size=512, chunk_overlap=100)` controls how the
  imported documents are split into retrieval chunks

This lesson does not answer questions yet. Its only job is to move the local
documents into the corpus so the next lesson can use them for generation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import vertexai
from dotenv import load_dotenv
from vertexai import rag

ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = ROOT_DIR / "uploads"
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


def get_upload_files() -> list[str]:
    if not UPLOADS_DIR.exists():
        print("Missing uploads directory")
        print("Create the directory and add files before running Lesson 3.")
        sys.exit(1)

    files = sorted(str(path) for path in UPLOADS_DIR.iterdir() if path.is_file())
    if not files:
        print("No files found in uploads/")
        print("Add at least one file to uploads/ before running Lesson 3.")
        sys.exit(1)

    return files


def main() -> None:
    load_dotenv()

    print("LESSON 3: Import Files into the RagCorpus\n")

    project_id = get_required_setting("VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
    location = get_required_setting("VERTEX_LOCATION", "GOOGLE_CLOUD_REGION")
    corpus_name = read_corpus_name()
    files = get_upload_files()

    print("Step 1: Initializing Vertex AI...")
    vertexai.init(project=project_id, location=location)
    print("Vertex AI initialized.")

    print("\nStep 2: Reading the saved corpus name...")
    print(f"Corpus Name: {corpus_name}")

    print("\nStep 3: Scanning uploads/ for files...")
    print(f"Found {len(files)} file(s):")
    for file_path in files:
        print(f"  - {Path(file_path).name}")

    print("\nStep 4: Importing files with chunking...")
    response = rag.import_files(
        corpus_name=corpus_name,
        paths=files,
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=512,
                chunk_overlap=100,
            )
        ),
    )
    print("Import request completed.")
    print(f"Import response: {response}")

    print("\nWhat did we learn?")
    print("- Lesson 3 reuses the persisted corpus name from Lesson 2.")
    print("- Vertex RAG imports local files directly from the uploads/ directory.")
    print("- Chunking is applied during import with size 512 and overlap 100.")


if __name__ == "__main__":
    main()
