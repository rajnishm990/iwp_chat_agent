import json

from rag import Index, crawl

pages = 60
cache = {}
for w in json.load(open("workspaces.json")):
    idx = Index()
    for u in w["urls"]:
        if u not in cache:
            print("crawling", u)
            cache[u] = crawl(u, pages, lambda n, url: print(f"  {n}: {url}"))
        idx.add(cache[u])
    idx.save(w["name"])
    print(w["name"], idx.stats())
