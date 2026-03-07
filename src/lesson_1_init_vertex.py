#!/usr/bin/env python3
"""
LESSON 1: Initialize Vertex AI for RAG

This script checks that your `.env` file is loaded, confirms that the required
Vertex settings are present, and initializes the Vertex AI SDK with those
values.

How the API call works:
- `load_dotenv()` loads the project and location from `.env`
- `vertexai.init(...)` sets the default project and region for later RAG calls
- `google.auth.default()` confirms that Application Default Credentials are
  available for the active environment

This lesson does not create a corpus or upload any files. It only proves that
the local machine is ready for the later lessons.
"""
from __future__ import annotations

import os
import sys

import google.auth
import vertexai
from dotenv import load_dotenv


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

    print("LESSON 1: Initialize Vertex AI\n")

    project_id = get_required_setting("VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
    location = get_required_setting("VERTEX_LOCATION", "GOOGLE_CLOUD_REGION")

    print("Step 1: Checking Application Default Credentials...")
    credentials, detected_project = google.auth.default()
    print("ADC found.")
    print(f"Credential type: {type(credentials).__name__}")
    if detected_project:
        print(f"ADC project hint: {detected_project}")
    else:
        print("ADC project hint: not provided by credentials")

    print("\nStep 2: Initializing Vertex AI...")
    vertexai.init(project=project_id, location=location)
    print("Vertex AI initialized.")

    print("\nEnvironment Summary:")
    print(f"Project ID: {project_id}")
    print(f"Location: {location}")
    if detected_project:
        print(f"ADC Project Hint: {detected_project}")
    else:
        print("ADC Project Hint: not provided by credentials")
    print(f"Credential Type: {type(credentials).__name__}")

    print("\nWhat did we learn?")
    print("- `.env` controls the project and location for this tutorial.")
    print("- The lesson accepts either VERTEX_* or GOOGLE_CLOUD_* variable names.")
    print("- ADC provides the credentials used by the Vertex SDK.")
    print("- `vertexai.init(...)` sets up the SDK for later RAG lessons.")


if __name__ == "__main__":
    main()
