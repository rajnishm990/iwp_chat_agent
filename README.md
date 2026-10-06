# AI Concierge (100% free stack)

Streamlit + Groq (Llama 3.3 70B free tier) + fastembed (local embeddings) + numpy search.

Features: streaming RAG chat over any number of websites/files, live lead card with scoring, smart human handoff
(summary + draft reply + WhatsApp link), omnichannel inbox (Instagram/WhatsApp/email triage with approve-and-route),
admin dashboard with leads CSV, multiple workspaces (IWP, AMK, unified), Hindi/Hinglish replies.

## Run
    pip install -r requirements.txt
    export GROQ_API_KEY=...            # free at console.groq.com
    python prebuild.py                 # crawls sites once, saves data/*.pkl (commit these for instant demos)
    streamlit run app.py

## Emails
Handoffs are only logged unless you flip "Send real emails" in the sidebar AND set SMTP_USER / SMTP_PASS (Gmail app password).
Set HANDOFF_OVERRIDE_EMAIL to your own address so demo emails never reach the real company.
Replace the `replace-me@example.com` entries in workspaces.json (or edit them live in the sidebar).

## Deploy
Push to GitHub (include data/*.pkl), create a Streamlit Community Cloud app, paste .streamlit/secrets.toml.example into Secrets.
