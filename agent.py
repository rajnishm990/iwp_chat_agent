import json
import os
import smtplib
from email.message import EmailMessage
from urllib.parse import quote

from groq import Groq

MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
DELIM = "<<<META>>>"

CHAT_PROMPT = """You are the AI concierge for @@company@@. Be warm, concise and professional; use light markdown. Reply in the language the visitor writes in (Hindi/Hinglish is welcome).

Rules:
- Answer ONLY from the context below. Never invent prices, availability, dates or policies. If the context lacks the answer, say so and offer to connect a human.
- When the visitor shows planning intent, help first, then ask for ONE missing detail at a time (never an interrogation). Capture these fields when mentioned:
@@fields@@
- Already captured: @@lead@@
- Escalate (escalate=true) when the visitor asks for a person or a call, is upset, wants a custom quote/negotiation/contract, is a job applicant, is a vendor/partner, has an urgent problem with an existing booking, or you cannot answer after trying. Do not escalate for questions you can answer. When escalating, tell the visitor you are connecting them with the team and will take their contact details.
- Departments (use the key):
@@depts@@

Output format: first the reply to the visitor (no JSON). Then a new line containing exactly <<<META>>> followed by one JSON object, nothing after it:
{"escalate": false, "department": null, "reason": "", "lead": {"<field key>": "<value>"}, "lead_score": "cold|warm|hot", "suggestions": ["<short follow-up question the visitor might ask>", "...", "..."]}
"lead" holds only fields learned or updated this turn (use the field keys above). lead_score reflects buying intent: hot = specific date/budget/ready to talk, warm = exploring with some details, cold = general curiosity.

Context:
@@ctx@@"""

TRIAGE_PROMPT = """You triage inbound messages for @@company@@ (channel: @@channel@@). Ground the draft reply in the context; never invent prices or availability.
Fields to extract (use these keys):
@@fields@@
Departments (use the key):
@@depts@@
Return JSON only:
{"intent": "wedding_enquiry|travel_enquiry|career|vendor_partnership|complaint|general|spam", "urgency": "low|medium|high", "language": "<language of the message>", "department": "<key>", "needs_human": true, "lead": {"<field key>": "<value>"}, "lead_score": "cold|warm|hot", "summary": "<one or two sentences for staff>", "draft_reply": "<ready-to-send reply in the sender's language, short, ends by asking for the single most useful missing detail>"}
Context:
@@ctx@@"""

SUMMARY_PROMPT = """Summarise this customer conversation for the @@dept@@ team at @@company@@. Known lead details: @@lead@@
Return JSON only:
{"summary": "<3-4 sentences>", "key_points": ["..."], "urgency": "low|medium|high", "suggested_reply": "<short draft reply to the customer>"}"""


class Turn:
    def __init__(self):
        self.text, self.meta, self.sources = "", {}, []


def _client():
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def _fill(tpl, **kw):
    for k, v in kw.items():
        tpl = tpl.replace(f"@@{k}@@", str(v))
    return tpl


def _fields(fields):
    return "\n".join(f"- {f['key']}: {f['label']}" for f in fields)


def _depts(depts):
    return "\n".join(f"- {d['key']}: {d['name']} — {d['handles']}" for d in depts)


def _retrieve(index, query, k=6):
    hits = index.search(query, k) if index else []
    ctx = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits) or "(no knowledge loaded)"
    return ctx, list(dict.fromkeys(h["source"] for h in hits))[:3]


def parse_json(text):
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except ValueError:
        return {}


def _json(messages, temperature=0.2):
    r = _client().chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature,
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content)


def stream_reply(index, history, ws, lead, turn):
    query = " ".join(m["content"] for m in history if m["role"] == "user")[-600:]
    ctx, turn.sources = _retrieve(index, query)
    system = _fill(CHAT_PROMPT, company=ws["company"], fields=_fields(ws["fields"]),
                   lead=json.dumps(lead, ensure_ascii=False), depts=_depts(ws["depts"]), ctx=ctx)
    stream = _client().chat.completions.create(
        model=MODEL, temperature=0.3, stream=True,
        messages=[{"role": "system", "content": system}] + history[-10:],
    )
    buf, emitted, done = "", 0, False
    for chunk in stream:
        buf += chunk.choices[0].delta.content or ""
        if done:
            continue
        i = buf.find(DELIM)
        if i >= 0:
            tail = buf[emitted:i].rstrip()
            if tail:
                yield tail
            done = True
        else:
            safe = len(buf) - (len(DELIM) - 1)
            if safe > emitted:
                yield buf[emitted:safe]
                emitted = safe
    if not done and buf[emitted:]:
        yield buf[emitted:]
    head, _, meta = buf.partition(DELIM)
    turn.text, turn.meta = head.strip(), parse_json(meta)


def triage(index, ws, channel, sender, message):
    ctx, _ = _retrieve(index, message)
    system = _fill(TRIAGE_PROMPT, company=ws["company"], channel=channel,
                   fields=_fields(ws["fields"]), depts=_depts(ws["depts"]), ctx=ctx)
    return _json([{"role": "system", "content": system},
                  {"role": "user", "content": f"From: {sender}\n\n{message}"}], 0.3)


def summarise(history, dept_name, company, lead):
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    system = _fill(SUMMARY_PROMPT, dept=dept_name, company=company, lead=json.dumps(lead, ensure_ascii=False))
    return _json([{"role": "system", "content": system}, {"role": "user", "content": transcript}])


def send_email(to, subject, body):
    user, pwd = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
    if not (user and pwd):
        return "skipped: SMTP credentials not set"
    override = os.environ.get("HANDOFF_OVERRIDE_EMAIL")
    if override:
        subject, body, to = f"{subject} (intended for {to})", body, override
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = user, to, subject
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, pwd)
        s.send_message(msg)
    return f"emailed {to}"


def wa_link(phone, text):
    digits = "".join(c for c in str(phone) if c.isdigit())
    return f"https://wa.me/{digits}?text={quote(text[:900])}" if digits else ""
