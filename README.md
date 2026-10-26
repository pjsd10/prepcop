# PrepCop

**Interrogate. Educate. Elevate.**

PrepCop is a guided interview-preparation app built with Streamlit, LangGraph, Fireworks AI, Tavily, and SQLite. Two sequential agents help a candidate turn a confirmed resume and job description into a source-aware preparation pack. The candidate controls input consent, skill priorities, clarification answers, brief approval, and final report review.

The **Profile Interviewer** compares confirmed experience with job requirements, separates actual exposure from desired practice depth, and asks up to five targeted clarifications. The **Research Coach** uses Tavily search and page extraction for one to three approved topics, then drafts up to six practice questions and a timed study plan. The app shows source excerpts, uncertainty, and partial results when evidence or provider calls are insufficient. A readable source alone does not prove that every answer is supported; human review remains required.

## Run locally

Use Python 3.12. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in `FIREWORKS_API_KEY`, `FIREWORKS_MODEL`, and `TAVILY_API_KEY` in the local `.env` file. Do not commit that file or paste keys into an issue. Then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app accepts a static UTF-8 `.html` or `.htm` resume and a pasted job description. It extracts inert text for candidate review; Word and PDF upload are outside this pilot. Uploads are limited to 1 MB. The candidate chooses priorities, can answer or skip clarification questions, edits and approves a versioned brief, and reviews the sourced report before downloading its HTML version. Fictional inputs for trying the flow are under `tests/fixtures/fictional-resumes/`.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pip install "pytest>=8,<10"
.\.venv\Scripts\python.exe -m pytest -q
```

These controlled tests do not make Fireworks or Tavily calls. They cover HTML parsing, workflow approvals and limits, saved SQLite/LangGraph state, source handling, failure recovery, and Streamlit interactions. They do not establish live provider reliability or the usefulness of every generated answer.

## Data and deployment limits

Runs and checkpoints are stored in the local `runtime/` folder. A private resume code reopens a saved run. This is a **local single-user pilot**, without hosted accounts or durable cloud storage. Do not publish it as an unrestricted multi-user service without adding appropriate isolation, persistence, consent wording, and API-spending controls. Input and retrieved text are treated as untrusted data; candidate information should not be sent to web search.

This repository is intentionally limited to application code, runtime prompts, configuration, controlled tests, and fictional test inputs. Local credentials, saved resumes and runs, generated screenshots, demo material, design history, and submission drafts are excluded from Git. The `.env.example` file contains names only, not keys.
