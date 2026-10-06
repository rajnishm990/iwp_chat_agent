import io
import pickle
import re
import time
from collections import Counter, deque
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse, urldefrag

import numpy as np
import requests
from bs4 import BeautifulSoup


SKIP_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip", ".mp4", ".css", ".js", ".ico", ".webp")
HEADERS = {"User-Agent": "Mozilla/5.0 (RAG-Demo-Bot)"}
DATA = Path("data")
_embedder = None


def slug(name):
    return re.sub(r"\W+", "-", name.lower()).strip("-")


def embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _embedder


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


PHONE = re.compile(r"\+\d{1,3}[\s-]?\(?\d{2,5}\)?[\s-]?\d{3,5}[\s-]?\d{3,5}|\b[6-9]\d{4}[\s-]?\d{5}\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def contacts_from(soup, text):
    phones, emails = set(), set()
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h.lower().startswith("tel:"):
            phones.add(clean(unquote(h[4:])))
        elif h.lower().startswith("mailto:"):
            emails.add(unquote(h[7:]).split("?")[0].strip())
        else:
            m = re.search(r"(?:wa\.me/|phone=)\+?(\d{8,15})", h)
            if m:
                phones.add(f"+{m.group(1)} (WhatsApp)")
    phones |= {clean(m) for m in PHONE.findall(text)}
    emails |= set(EMAIL.findall(text))
    emails = {e for e in emails if not e.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"))}
    return phones, emails


def uniq_phones(counter, n=6):
    out, seen = [], set()
    for k, _ in counter.most_common():
        d = re.sub(r"\D", "", k)[-10:]
        if d not in seen:
            seen.add(d)
            out.append(k)
    return out[:n]


def crawl(start_url, max_pages=40, progress=None):
    root = urlparse(start_url).netloc
    seen, queue, docs = set(), deque([start_url]), []
    phones, emails, boiler = Counter(), Counter(), {}
    while queue and len(docs) < max_pages:
        url = urldefrag(queue.popleft())[0]
        if url in seen or url.lower().endswith(SKIP_EXT):
            continue
        seen.add(url)
        time.sleep(0.15)
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if "text/html" not in r.headers.get("content-type", ""):
                continue
        except requests.RequestException:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            link = urldefrag(urljoin(url, a["href"]))[0]
            if urlparse(link).netloc == root and link not in seen:
                queue.append(link)
        p, e = contacts_from(soup, clean(soup.get_text(" ")))
        phones.update(p)
        emails.update(e)
        for tag in soup.find_all(["header", "footer"]):
            t = clean(tag.get_text(" "))
            if 40 < len(t) < 1500:
                boiler.setdefault(t, url)
        title = clean(soup.title.string) if soup.title and soup.title.string else ""
        for t in soup(["script", "style", "noscript", "nav", "footer", "header", "svg"]):
            t.decompose()
        text = clean(soup.get_text(" "))
        if len(text) > 200:
            docs.append((url, f"{title}. {text}"))
            if progress:
                progress(len(docs), url)
    if phones or emails:
        docs.append((start_url + "#contact", "Contact us. Phone and WhatsApp numbers: " + ", ".join(uniq_phones(phones))
                     + ". Email addresses: " + ", ".join(k for k, _ in emails.most_common(6)) + "."))
    for t, u in list(boiler.items())[:6]:
        docs.append((u + "#header-footer", "Site header and footer information: " + t))
    return docs


def read_upload(name, data):
    name = name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        return clean(" ".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages))
    if name.endswith(".docx"):
        from docx import Document
        return clean(" ".join(p.text for p in Document(io.BytesIO(data)).paragraphs))
    return clean(data.decode("utf-8", errors="ignore"))


def chunk(text, size=900, overlap=150):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


class Index:
    def __init__(self):
        self.chunks, self.sources, self.vecs = [], [], None

    def add(self, docs):
        new_c, new_s = [], []
        for src, text in docs:
            for c in chunk(text):
                new_c.append(c)
                new_s.append(src)
        if not new_c:
            return
        v = np.array(list(embedder().embed(new_c)), dtype="float32")
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        self.vecs = v if self.vecs is None else np.vstack([self.vecs, v])
        self.chunks += new_c
        self.sources += new_s

    def search(self, query, k=6):
        if self.vecs is None:
            return []
        q = np.array(list(embedder().query_embed(query)), dtype="float32")[0]
        q /= np.linalg.norm(q)
        idx = np.argsort(-(self.vecs @ q))[:k]
        return [{"text": self.chunks[i], "source": self.sources[i]} for i in idx]

    def stats(self):
        return len(set(self.sources)), len(self.chunks)

    def save(self, name):
        DATA.mkdir(exist_ok=True)
        with open(DATA / f"{slug(name)}.pkl", "wb") as f:
            pickle.dump((self.chunks, self.sources, self.vecs), f)

    @classmethod
    def load(cls, name):
        idx, p = cls(), DATA / f"{slug(name)}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                idx.chunks, idx.sources, idx.vecs = pickle.load(f)
        return idx
