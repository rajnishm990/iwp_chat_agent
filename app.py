import json
import os
import time

import pandas as pd
import streamlit as st

import agent
from rag import Index, crawl, read_upload

st.set_page_config(page_title="IWP CHAT AGENT", page_icon="✨", layout="wide")

for k in ("GROQ_API_KEY", "LLM_MODEL", "SMTP_USER", "SMTP_PASS", "HANDOFF_OVERRIDE_EMAIL"):
    try:
        if k in st.secrets:
            os.environ[k] = st.secrets[k]
    except Exception:
        pass

HONORED = {"explicit_request", "complaint", "urgent", "ready_to_book"}
BADGE = {"hot": "Hot lead", "warm": "Warm lead", "cold": "Cold lead"}
DEFAULT_DEPTS = [{"key": "sales", "name": "Sales", "email": "replace-me@example.com", "phone": "", "handles": "new enquiries and quotes"}]
DEFAULT_FIELDS = [{"key": "contact_name", "label": "Name"}, {"key": "email", "label": "Email"}, {"key": "phone", "label": "Phone"}, {"key": "interest", "label": "Interest"}]

SAMPLES = {
    "Instagram DM — NRI wedding enquiry": ("Instagram", "@priya.and.rohan", "Hi! We're getting married in Feb 2027 and want something royal in Udaipur, around 200 guests. We're based in New Jersey. Budget maybe 4 cr? Can you share packages? 💍"),
    "WhatsApp — Kerala trip": ("WhatsApp", "+91 98xxxxxx12", "Hello, do you have a 5 night Kerala backwater package for 2 adults in December? Also need airport pickup from Kochi."),
    "Email — job application": ("Email", "anita.verma@gmail.com", "Hi team, I'm applying for an Event Executive role. 3 years of decor and vendor coordination experience at a Delhi agency. CV attached. Reach me on 98xxxxxx45."),
    "Instagram DM — vague": ("Instagram", "@travel_with_sam", "price?"),
    "WhatsApp — urgent complaint": ("WhatsApp", "+91 99xxxxxx78", "Our cab for this morning's airport transfer never showed up and our flight is in 3 hours!! Please call me immediately."),
    "Email — vendor partnership": ("Email", "hello@bloomstudio.in", "We are a luxury floral decorator in Jaipur and would love to explore a vendor partnership with your team. Can we share our portfolio?"),
}

ss = st.session_state


def recs(df):
    return df.dropna(subset=["key"]).to_dict("records")


def make_ws(p):
    return {
        "name": p["name"], "company": p["company"], "urls": p["urls"],
        "depts": pd.DataFrame(p["departments"]), "fields": pd.DataFrame(p["lead_fields"]),
        "index": Index.load(p["name"]), "messages": [], "lead": {}, "score": "", "pending": None, "suggest": [],
    }


if "ws" not in ss:
    ss.ws, ss.tickets, ss.leads, ss.inbox = {}, [], {}, []
    with open("workspaces.json") as f:
        for p in json.load(f):
            ss.ws[p["name"]] = make_ws(p)


def route(name, dept, subject, body, channel, phone="", reply=""):
    if ss.get("real_mail"):
        try:
            status = agent.send_email(dept["email"], subject, body)
        except Exception as e:
            status = f"failed: {e}"
    else:
        status = "logged (demo mode, email not sent)"
    ss.tickets.append({
        "time": time.strftime("%H:%M:%S"), "workspace": name, "channel": channel, "dept": dept["name"],
        "to": dept["email"], "status": status, "body": body,
        "wa": agent.wa_link(phone, reply) if phone and reply else "",
    })
    return status


def find_dept(depts, key):
    return next((d for d in depts if d["key"] == key), depts[0])


def apply_meta(name, W, meta):
    for k, v in (meta.get("lead") or {}).items():
        if v not in (None, "", [], {}):
            W["lead"][k] = str(v)
    W["score"] = meta.get("lead_score") or W["score"]
    W["suggest"] = (meta.get("suggestions") or [])[:3]
    depts = recs(W["depts"])
    if meta.get("unanswered"):
        W["misses"] = W.get("misses", 0) + 1
    if meta.get("escalate") and meta.get("reason") in HONORED and not W["pending"]:
        W["pending"] = {"department": meta.get("department"), "reason": meta["reason"]}
    elif W.get("misses", 0) >= 2 and not W["pending"] and not W.get("offered"):
        W["offered"] = True
        W["pending"] = {"department": meta.get("department") or depts[0]["key"], "reason": "unanswered"}
        W["messages"].append({"role": "assistant", "content": "I couldn't find that on the website. If you'd like, I can pass this to the team. Just confirm your details below."})
    if W["lead"]:
        ss.leads[f"chat:{name}"] = {"source": "Web chat", "workspace": name, "score": W["score"], **W["lead"]}


def lead_card(W):
    fields = recs(W["fields"])
    filled = sum(1 for f in fields if W["lead"].get(f["key"]))
    st.subheader("Live lead card")
    st.markdown(BADGE.get(W["score"], "No score yet"))
    st.progress(filled / max(len(fields), 1), text=f"{filled}/{len(fields)} details captured")
    for f in fields:
        st.markdown(f"**{f['label']}**  \n{W['lead'].get(f['key'], '—')}")


def chat_view(name, W):
    st.title(f"{W['company']}")
    depts = recs(W["depts"])
    left, right = st.columns([3, 1.2])
    with right:
        lead_card(W)
    with left:
        n_src, n_chunks = W["index"].stats()
        if not n_chunks:
            st.warning("No knowledge loaded yet. Open 'Knowledge & configuration' in the sidebar and click Crawl & index.")
        for m in W["messages"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("sources"):
                    st.caption("Sources: " + " · ".join(m["sources"]))
                if m.get("hits"):
                    with st.expander("🔎 Why this answer? (retrieved website text)"):
                        for src, snippet in m["hits"]:
                            st.markdown(f"**{src}**  \n{snippet}…")
        if W["suggest"] and not W["pending"]:
            cols = st.columns(len(W["suggest"]))
            for i, s in enumerate(W["suggest"]):
                if cols[i].button(s, key=f"sg{name}{len(W['messages'])}{i}"):
                    ss.queued = s
                    st.rerun()
        if W["pending"]:
            d = find_dept(depts, W["pending"]["department"])
            with st.form("handoff"):
                st.info(f"Connecting you with **{d['name']}** ({d['phone']}). Confirm your details and they'll reach out.")
                n = st.text_input("Your name", W["lead"].get("contact_name", ""))
                e = st.text_input("Email", W["lead"].get("email", ""))
                p = st.text_input("Phone / WhatsApp", W["lead"].get("phone", ""))
                if st.form_submit_button("Send to team"):
                    W["lead"].update({"contact_name": n, "email": e, "phone": p})
                    with st.spinner("Preparing summary for the team..."):
                        s = agent.summarise(W["messages"], d["name"], W["company"], W["lead"])
                    body = (
                        f"Customer: {n} | {e} | {p}\nLead score: {W['score']} | Urgency: {s.get('urgency')}\n\n"
                        f"Summary:\n{s.get('summary')}\n\nKey points:\n" + "\n".join(f"- {x}" for x in s.get("key_points", []))
                        + "\n\nCaptured details:\n" + "\n".join(f"- {k}: {v}" for k, v in W["lead"].items())
                        + f"\n\nSuggested reply:\n{s.get('suggested_reply')}"
                    )
                    route(name, d, f"[Chat handoff] {str(s.get('urgency', '')).upper()} - {n or 'Visitor'}", body, "Web chat", p, s.get("suggested_reply", ""))
                    ss.leads[f"chat:{name}"] = {"source": "Web chat", "workspace": name, "score": W["score"], **W["lead"]}
                    W["messages"].append({"role": "assistant", "content": f"Done. I've passed everything to **{d['name']}**; they'll contact you shortly."})
                    W["pending"] = None
                    st.rerun()

    prompt = st.chat_input("Ask about venues, packages, careers, anything...") or ss.pop("queued", None)
    if not prompt:
        return
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Set GROQ_API_KEY in Streamlit secrets or the environment.")
        return
    W["suggest"] = []
    with left:
        with st.chat_message("user"):
            st.markdown(prompt)
        W["messages"].append({"role": "user", "content": prompt})
        history = [{"role": m["role"], "content": m["content"]} for m in W["messages"]]
        turn = agent.Turn()
        with st.chat_message("assistant"):
            try:
                st.write_stream(agent.stream_reply(W["index"], history, {"company": W["company"], "depts": depts, "fields": recs(W["fields"])}, W["lead"], turn))
            except Exception as e:
                W["messages"].pop()
                st.error(f"LLM error: {e}")
                return
    W["messages"].append({"role": "assistant", "content": turn.text, "sources": turn.sources, "hits": turn.hits})
    apply_meta(name, W, turn.meta)
    st.rerun()


def inbox_view(name, W):
    st.title("Omnichannel inbox")
    st.caption("Instagram, WhatsApp and email land in one place. AI classifies, extracts the lead, drafts a grounded reply and routes it. A human approves. (Messages are simulated here; the same pipeline plugs into the Meta webhooks.)")
    depts, fields = recs(W["depts"]), recs(W["fields"])
    ws_cfg = {"company": W["company"], "depts": depts, "fields": fields}
    c1, c2 = st.columns([1, 2])
    with c1:
        sample = st.selectbox("Incoming message", list(SAMPLES) + ["Custom"])
        ch, snd, txt = SAMPLES.get(sample, ("Instagram", "@someone", ""))
        channel = st.selectbox("Channel", ["Instagram", "WhatsApp", "Email"], index=["Instagram", "WhatsApp", "Email"].index(ch), key=f"ch{sample}")
        sender = st.text_input("Sender", snd, key=f"sn{sample}")
        message = st.text_area("Message", txt, height=150, key=f"mg{sample}")
        run1 = st.button("Process with AI", type="primary", use_container_width=True)
        run_all = st.button("⚡ Triage all samples", use_container_width=True)

    def process(ch_, snd_, msg_):
        r = agent.triage(W["index"], ws_cfg, ch_, snd_, msg_)
        item = {"id": len(ss.inbox), "workspace": name, "channel": ch_, "sender": snd_, "message": msg_, "r": r, "done": None}
        ss.inbox.insert(0, item)
        ss.leads[f"inbox:{item['id']}"] = {"source": ch_, "workspace": name, "score": r.get("lead_score", ""), "sender": snd_, **{k: str(v) for k, v in (r.get("lead") or {}).items() if v}}

    if not os.environ.get("GROQ_API_KEY") and (run1 or run_all):
        st.error("Set GROQ_API_KEY first.")
    elif run1 and message.strip():
        with st.spinner("AI is triaging..."):
            process(channel, sender, message)
    elif run_all:
        bar = st.progress(0.0)
        for i, (ch_, snd_, msg_) in enumerate(SAMPLES.values()):
            process(ch_, snd_, msg_)
            bar.progress((i + 1) / len(SAMPLES), text=f"Triaged {i + 1}/{len(SAMPLES)}")
            time.sleep(1)
        bar.empty()

    with c2:
        items = [i for i in ss.inbox if i["workspace"] == name]
        if not items:
            st.info("Nothing processed yet. Pick a message and click Process with AI, or triage all samples.")
        for it in items:
            r = it["r"]
            dept = find_dept(depts, r.get("department"))
            icon = {"Instagram": "📸", "WhatsApp": "💬", "Email": "✉️"}.get(it["channel"], "📨")
            head = f"{icon} {it['sender']} · {r.get('intent', '?')} · {BADGE.get(r.get('lead_score'), '')} · urgency {r.get('urgency', '?')} → {dept['name']}"
            with st.expander(head, expanded=it["done"] is None and it is items[0]):
                st.markdown(f"> {it['message']}")
                st.markdown(f"**AI summary:** {r.get('summary', '')}  \n*Language: {r.get('language', '?')}*")
                lead = {k: str(v) for k, v in (r.get("lead") or {}).items() if v}
                if lead:
                    st.table(pd.DataFrame(list(lead.items()), columns=["Field", "Value"]))
                draft = st.text_area("Draft reply (editable)", r.get("draft_reply", ""), key=f"dr{it['id']}", height=120)
                if it["done"]:
                    st.success(f"Routed to {dept['name']} · {it['done']} · reply marked as sent (simulated)")
                elif st.button(f"Approve reply & route to {dept['name']}", key=f"ap{it['id']}"):
                    body = (
                        f"Channel: {it['channel']} | From: {it['sender']}\nIntent: {r.get('intent')} | Urgency: {r.get('urgency')} | Score: {r.get('lead_score')}\n\n"
                        f"Message:\n{it['message']}\n\nAI summary:\n{r.get('summary')}\n\nExtracted details:\n"
                        + "\n".join(f"- {k}: {v}" for k, v in lead.items()) + f"\n\nApproved reply:\n{draft}"
                    )
                    it["done"] = route(name, dept, f"[{it['channel']}] {r.get('intent')} - {it['sender']}", body, it["channel"], lead.get("phone", ""), draft)
                    st.rerun()


def admin_view(name, W):
    st.title("Admin: leads & handoffs")
    leads = [v for v in ss.leads.values()]
    n_hot = sum(1 for v in leads if v.get("score") == "hot")
    mins = st.number_input("Assumption: minutes a person spends triaging one message", 1, 30, 5)
    handled = len(ss.inbox) + len([k for k in ss.leads if k.startswith("chat:")])
    a, b, c, d = st.columns(4)
    a.metric("Conversations & messages handled", handled)
    b.metric("Leads captured", len(leads))
    c.metric("Hot leads", n_hot)
    d.metric("Est. hours saved", round(handled * mins / 60, 1))
    st.subheader("Leads")
    if leads:
        df = pd.DataFrame(leads)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download leads CSV", df.to_csv(index=False), "leads.csv", "text/csv")
    else:
        st.write("No leads yet.")
    st.subheader("Handoff tickets")
    for t in reversed(ss.tickets):
        with st.expander(f"{t['time']} · {t['channel']} → {t['dept']} ({t['to']}) · {t['status']}"):
            st.text(t["body"])
            if t["wa"]:
                st.link_button("Reply on WhatsApp", t["wa"])


st.sidebar.title("IWP CHAT AGENT")
view = st.sidebar.radio("View", ["Chat", "Omnichannel inbox", "Admin"])
name = st.sidebar.selectbox("Workspace", list(ss.ws))
W = ss.ws[name]
st.sidebar.toggle("Send real emails", key="real_mail", value=False, help="Off = demo mode: handoffs are logged, nothing is sent.")

with st.sidebar.expander("⚙️ Knowledge & configuration"):
    W["company"] = st.text_input("Company name", W["company"], key=f"co_{name}")
    urls = st.text_area("Website URLs (one per line)", "\n".join(W["urls"]), key=f"u_{name}")
    W["urls"] = [u.strip() for u in urls.splitlines() if u.strip().startswith("http")]
    pages = st.slider("Max pages per site", 5, 80, 40, key=f"p_{name}")
    if st.button("Crawl & index", key=f"c_{name}", use_container_width=True) and W["urls"]:
        idx, bar = Index(), st.progress(0.0)
        for u in W["urls"]:
            idx.add(crawl(u, pages, lambda n, url: bar.progress(min(n / pages, 1.0), text=url[:45])))
        idx.save(name)
        W["index"] = idx
        bar.empty()
        st.success("Indexed")
    files = st.file_uploader("Extra knowledge (PDF, DOCX, TXT)", accept_multiple_files=True, key=f"f_{name}")
    if files and st.button("Add files", key=f"af_{name}", use_container_width=True):
        W["index"].add([(f.name, read_upload(f.name, f.read())) for f in files])
        W["index"].save(name)
        st.success(f"Added {len(files)} file(s)")
    st.caption("Knowledge: %d sources, %d chunks" % W["index"].stats())
    cb = W["index"].contact_block()
    st.caption("Contacts found on site: " + (cb[:350] if cb else "none. Re-crawl, or the site may load them with JavaScript."))
    st.markdown("**Departments**")
    W["depts"] = st.data_editor(W["depts"], num_rows="dynamic", hide_index=True, key=f"d_{name}")
    st.markdown("**Lead fields**")
    W["fields"] = st.data_editor(W["fields"], num_rows="dynamic", hide_index=True, key=f"lf_{name}")

with st.sidebar.expander("➕ New workspace"):
    new = st.text_input("Name", key="newws")
    if st.button("Create") and new and new not in ss.ws:
        ss.ws[new] = make_ws({"name": new, "company": new, "urls": [], "departments": DEFAULT_DEPTS, "lead_fields": DEFAULT_FIELDS})
        st.rerun()

if st.sidebar.button("Reset this chat", use_container_width=True):
    W.update({"messages": [], "lead": {}, "score": "", "pending": None, "suggest": [], "misses": 0, "offered": False})
    ss.leads.pop(f"chat:{name}", None)
    st.rerun()

{"Chat": chat_view, "Omnichannel inbox": inbox_view, "Admin": admin_view}[view](name, W)
