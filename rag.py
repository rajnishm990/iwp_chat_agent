import io
import pickle
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

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


def crawl(start_url, max_pages=40, progress=None):
    root = urlparse(start_url).netloc
    seen, queue, docs = set(), deque([start_url]), []
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
        title = clean(soup.title.string) if soup.title and soup.title.string else ""
        for t in soup(["script", "style", "noscript", "nav", "footer", "header", "svg"]):
            t.decompose()
        text = clean(soup.get_text(" "))
        if len(text) > 200:
            docs.append((url, f"{title}. {text}"))
            if progress:
                progress(len(docs), url)
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
