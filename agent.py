import json
import os
import smtplib
from email.message import EmailMessage
from urllib.parse import quote

from groq import Groq

def model():
    return os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")


EXTRA = {"extra_body": {"reasoning_effort": "low"}}
DELIM = "<<<META>>>"

CHAT_PROMPT = """You are the AI concierge for @@company@@. Be warm, concise and genuinely helpful; use light markdown. Reply in the language the visitor writes in (Hindi/Hinglish is welcome).

How to help:
- Solve it yourself first. Answer fully from the context, combining several pieces if needed. A partial answer beats a deflection: say what you do know, then what you could not find.
- The company's public contact details (phone, WhatsApp, email, address) are published on its website and listed below. Share them whenever asked. This is expected and is not a privacy issue; never refuse.
- Never invent prices, availability, dates or policies. If something is not in the context, say so plainly and invite the visitor to rephrase or ask something else. Do not jump to a human.
- Do not end every reply with an offer to connect a human. Mention the team at most once per conversation unless the visitor asks again.
- If the visitor is planning a wedding or trip, after answering you may ask ONE helpful question about their plans, but not on every turn and not if they are just browsing. Fields worth capturing when mentioned:
@@fields@@
- Already captured: @@lead@@

Handoff is a last resort. Set escalate=true only when "reason" is one of: explicit_request (asks for a person, call or callback), complaint (upset or reporting a problem), urgent (time-critical issue with an existing booking), ready_to_book (clearly wants to proceed with a quote, booking or application). Otherwise escalate=false and reason="". When escalate is true, tell the visitor you are connecting them with the team and will take their contact details. Set unanswered=true (without escalating) when the context did not let you answer.

Official contact details:
@@contacts@@
Departments (use the key):
@@depts@@

Output format: first the reply to the visitor (no JSON). Then a new line containing exactly <<<META>>> followed by one JSON object, nothing after it:
{"escalate": false, "reason": "", "department": null, "unanswered": false, "lead": {"<field key>": "<value>"}, "lead_score": "cold|warm|hot", "suggestions": ["<short follow-up question the visitor might ask>", "...", "..."]}
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


def _contacts(depts):
    rows = []
    for d in depts:
        bits = [x for x in (d.get("email"), d.get("phone")) if x and "example.com" not in str(x) and "replace-me" not in str(x)]
        if bits:
            rows.append(f"- {d['name']}: " + ", ".join(bits))
    return "\n".join(rows) + "\n(Also use any contact details that appear in the context.)"


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
        model=model(), messages=messages, temperature=temperature,
        response_format={"type": "json_object"}, **EXTRA,
    )
    return json.loads(r.choices[0].message.content)


def stream_reply(index, history, ws, lead, turn):
    query = " ".join(m["content"] for m in history if m["role"] == "user")[-600:]
    ctx, turn.sources = _retrieve(index, query)
    system = _fill(CHAT_PROMPT, company=ws["company"], fields=_fields(ws["fields"]),
                   lead=json.dumps(lead, ensure_ascii=False), contacts=_contacts(ws["depts"]), depts=_depts(ws["depts"]), ctx=ctx)
    stream = _client().chat.completions.create(
        model=model(), temperature=0.3, stream=True, **EXTRA,
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
