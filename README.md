# PrepCop

**Interrogate. Educate. Elevate.**

PrepCop helps a candidate prepare for an interview using their resume, a job description, and public learning sources. It is a **working local Streamlit pilot** with two sequential AI agents and human approval before research and download. This repository contains the application and controlled tests; the app has not been deployed as a public service.

## What the app does

1. **Confirm the inputs.** Upload a static HTML resume, paste the job description, review the extracted text, and consent before it is sent to the configured AI service.
2. **Understand the profile.** The **Profile Interviewer** compares the resume with the role, shows a skill table, and asks up to five relevant clarification questions. The candidate chooses one to three priorities and a desired practice depth; prior experience and desired depth are kept separate.
3. **Approve the brief.** The candidate can correct the summary and uncertainties. Research starts only after they approve the current brief.
4. **Build the preparation pack.** The **Research Coach** searches and extracts public pages with Tavily, saves source excerpts, and drafts up to six practice questions plus a study plan fitted to the candidate's available time.
5. **Review and download.** The candidate checks the questions, sources, and gaps. Only a reviewed report can be downloaded as HTML. If sources or provider calls fail, the app preserves available work and labels a partial result.

The Profile Interviewer does **not** browse the web. The Research Coach can use only `search_sources` and `extract_page`. Candidate answers, brief approval, and report review are human decisions.

## Technology and dependencies

| What we used | Purpose |
| --- | --- |
| Python 3.12 and Streamlit | Guided web interface and HTML download |
| LangGraph and `langgraph-checkpoint-sqlite` | Two-agent workflow, pauses for human input, and saved checkpoints |
| Fireworks AI through LangChain/`langchain-openai` | Structured profile, clarifications, research decisions, and drafting |
| Tavily Search and Extract APIs | Find and read public sources for approved topics |
| SQLite (included with Python) | Save each run, operation results, budgets, and recovery state locally |
| Pydantic | Validate structured model outputs and report data |
| `httpx` and `python-dotenv` | Model-client dependency and local environment settings |
| pytest (development only) | Controlled tests without paid API calls |

The direct Python package versions are pinned in [requirements.txt](requirements.txt). Fireworks and Tavily are external services and require your own API keys; using the app can consume their credits.

## Run it locally

These steps use **Windows PowerShell** and Python 3.12:

```powershell
git clone https://github.com/pjsd10/prepcop.git
cd prepcop
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open the new `.env` file and fill in your own values:

```dotenv
FIREWORKS_API_KEY=your_key
FIREWORKS_MODEL=accounts/fireworks/models/glm-5p3-flash
TAVILY_API_KEY=your_key
```

The model ID above is the one used for local calibration; use a tool-capable model available to your Fireworks account. Keep `.env` private. It is excluded from Git and must never be put in an issue, screenshot, or commit.

Start the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address Streamlit prints. To try a fictional case, upload the [QA resume](tests/fixtures/fictional-resumes/02-qa-engineer/resume.html) and paste its [matching job description](tests/fixtures/fictional-resumes/02-qa-engineer/job-description.txt). Review the extracted text, choose your priorities, answer or skip clarifications, approve the brief, and inspect sources before marking the report reviewed. Other fictional roles are in the same `tests/fixtures/fictional-resumes/` directory.

## What we built and checked

- Implemented static HTML text extraction with input, size, and encoding checks. Uploaded scripts are not executed.
- Added versioned human approvals, bounded model/retrieval attempts, saved-result reuse, and explicit recovery when a remote outcome is uncertain.
- Kept source IDs and exact excerpts with the report so a person can inspect grounding. A link or matching quote alone does not prove an answer is correct.
- Tested the workflow, parsing, source handling, provider payloads, persistence, failure paths, and Streamlit interactions with scripted responses. The latest full local run passed **228 tests and 37 subtests** without using API credits.
- Rehearsed one fictional QA flow through the UI to a six-question pack and study plan using live providers. That pack still had source-support limits and was not marked reviewed or exported. Controlled tests and one rehearsal do not prove consistent quality across all roles.

To run the controlled suite yourself:

```powershell
.\.venv\Scripts\python.exe -m pip install "pytest>=8,<10"
.\.venv\Scripts\python.exe -m pytest -q
```

## Current limits

- Resume input is static UTF-8 `.html` or `.htm` up to 1 MB; Word, PDF, OCR, and live mock interviews are outside this pilot.
- Runs are saved in local SQLite files under `runtime/` and reopened with a private resume code. The app is designed for local single-user use; public multi-user hosting needs additional isolation, durable storage, consent wording, and API-spending controls.
- Research uses public web pages. Some pages may be blocked, empty, or only partly relevant. The app discloses gaps and requires human source review.
- Private resumes, keys, saved runs, demo recordings, and working documents are not part of this GitHub repository.

Application code is in `src/interview_prep/`, agent instructions are in `prompts/`, and controlled tests with fictional inputs are in `tests/`.
