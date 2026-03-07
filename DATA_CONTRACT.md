# Data Contract

This repository uses flat text files in the project root to pass state from one lesson script to the next.

## `corpus_name.txt`

Purpose:
Stores exactly one Vertex AI RAG Engine corpus resource name created by `lesson_2_create_corpus.py`.

Producer:
`src/lesson_2_create_corpus.py`

Consumers:
- `src/lesson_3_import_files.py`
- `src/lesson_4_query.py`

Format:
- Plain UTF-8 text
- Exactly one non-empty line
- No JSON
- No quotes
- No extra labels or comments
- A trailing newline is allowed

Expected value:
```text
projects/PROJECT_ID/locations/LOCATION/ragCorpora/CORPUS_ID
```

Examples:
```text
projects/my-gcp-project/locations/us-central1/ragCorpora/1234567890123456789
```

Read rule:
- Scripts must call `.read().strip()` before using the value.

Write rule:
- Scripts must write only the raw `corpus.name` string returned by the Vertex RAG API.

Validation rule:
- If the file is missing or empty, the consumer script must stop and tell the user to run `lesson_2_create_corpus.py` first.

Location rule:
- The file must live at the repository root as `corpus_name.txt`.

Rationale:
- This mirrors the old tutorial pattern that used `store_name.txt`, but swaps in the Vertex RAG corpus resource name for the new lesson flow.
