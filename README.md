# Protection Engine

An AI co-pilot for UK protection insurance advisers. While the adviser is on a call with a client, it turns their live notes into the next questions to ask, read-aloud script lines, coaching prompts and a structured fact-find. After the call, it writes the suitability summary and a follow-up email.

It also automates the work around the call: referral emails become leads in the CRM with a pre-call brief, and new leads are qualified, sent a personalised SMS and offered appointment slots.

> The co-pilot supports the adviser; it does not advise. Its prompts never quote premiums or recommend a cover amount, and firm facts such as fees or regulatory status are only stated when they are configured. The adviser makes every recommendation.

## What it does

**Before the call**
- **Referral to lead:** a mortgage broker's referral email arrives by webhook. The client's details are extracted, a contact is created in GoHighLevel (GHL), and a call-prep brief is stored on it.
- **Prep screen:** the adviser pastes a referral email, or reads the referral out loud, and gets a brief of key facts, likely needs and a call plan.
- **New-lead qualification:** a new enquiry is scored for readiness, cover type, urgency and impaired-life risk. A personalised SMS offers three real calendar slots, and the client's reply ("1", "2" or "3") books the appointment.

**During the call**
- **Live analysis:** as the adviser types notes, `/copilot/analyze` returns the next questions and prompts for the current stage of the call, grounded in retrieved product knowledge. It runs on a fast model so it lands within a second or two.
- **Read-aloud scripts:** streamed script lines for each stage of the call, from introduction and goals to the income stress test, health questions, each cover type, and close.
- **Bridges:** short lines that move the conversation naturally from one stage to the next, built from what the client has just said.
- **"Your Picture":** a whiteboard view at the pivot from needs to recommendation. It shows the client's goals, the income and cover shortfall if they became ill or died, and what existing cover is already in place, with coaching prompts drawn from a curated prompt bank.
- **Voice navigation:** the adviser can move between stages by speaking.

**After the call**
- **Suitability summary:** a factual fact-find summary for the adviser's record, available as text or a formatted PDF.
- **Follow-up email:** a warm recap email to send to the client.
- **Feedback:** advisers can rate outputs and failures are logged, to improve the prompts over time.

## How it works

| Part | Files | Role |
| --- | --- | --- |
| Web app | `app.py` | Flask app: login with signed, expiring session tokens and rate limiting; GHL webhooks for referral emails, new leads and SMS replies |
| Co-pilot API | `copilot.py` | All `/copilot/*` routes and their system prompts: prepare, analyze, script stream, bridge, Your Picture, summary, PDF, email |
| Front end | `static/copilot.html` | Single-page call screen served at `/copilot`: prep screen, notes box, stage navigation, whiteboard, voice input |
| Language models | `llm.py` | Claude helper. Each route uses a model matched to its speed and quality needs (fast model for live analysis, strongest model for the post-call summary). Handles fenced JSON and recovers partial JSON when a response is cut off |
| Knowledge retrieval (RAG) | `rag.py`, `ingest_docs.py` | OpenAI embeddings and a Chroma vector store of the adviser's methodology and provider documents (policy summaries, underwriting guides, claims statistics), so each prompt gets only the relevant passages |
| Prompt bank | `picture_prompt_bank.py` | Curated, editable lines for the Your Picture step. The model picks and fills these with the client's real names and figures; it does not improvise |
| CRM and booking | `ghl.py`, `booking.py`, `qualify.py`, `messaging.py` | GoHighLevel contacts, notes, custom fields and SMS; calendar free slots and booking; lead qualification; SMS writing |
| PDF | `pdf_summary.py` | Renders the suitability summary with ReportLab |

## Running it locally

Requires Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (it is git-ignored):

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...          # embeddings for retrieval
CHROMA_API_KEY=...          # Chroma Cloud vector store
CHROMA_TENANT=...
CHROMA_DATABASE=...
GHL_API_KEY=...             # GoHighLevel
GHL_LOCATION_ID=...
WEBHOOK_SECRET=...          # shared secret for the GHL webhooks
COPILOT_PASSWORD=...        # login for the co-pilot screen
COPILOT_SECRET_KEY=...      # optional: signing key for session tokens
BROKER_NAME=...
BROKER_COMPANY=...
BROKER_PHONE=...
```

Build the knowledge store, then start the app:

```bash
python rag.py               # load the methodology chunks
python ingest_docs.py       # optional: add provider PDFs from ./knowledge_docs (needs poppler: brew install poppler)
flask --app app run
```

Open `http://localhost:5000/copilot` and log in with `COPILOT_PASSWORD`.

## Deployment

Deployed on Railway with gunicorn (see `Procfile`). Set the same environment variables in Railway. Logs written to disk are wiped on each deploy, so feedback and failure logs need a database or log drain for long-term use.

## Tests

```bash
python tests/test_llm_repair.py
```

Checks that a Claude response cut off mid-JSON is recovered rather than lost.
