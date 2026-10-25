"""Keep the manuscript references in Zotero.

usage:
    python -m tools.zotero_refs push   # create the Zotero collection and add every reference cited in main.tex
    python -m tools.zotero_refs bib    # write 05_manuscript_JESTIE/references.bib from that collection

Zotero desktop must be running (local API at http://127.0.0.1:23119). Reading needs no key; 'push' writes
and needs a local API key (Zotero asks for approval after POST /api/local/authorize), given in the
environment variable ZOTERO_LOCAL_KEY. Crossref lookups put CROSSREF_MAILTO, if set, in the User-Agent.

Conventions in Zotero:
  * The LaTeX cite key is the item's Citation Key field (Zotero 10; 'push' writes "Citation Key: <key>" to Extra,
    which Zotero moves into that field). Extra may hold "BibTeX: and others" for a truncated author list.
  * Words whose capitalization must survive the IEEE style are wrapped in <span class="nocase">...</span>
    in the title; they are exported as {...}.
  * The journal is exported as its abbreviation: Journal Abbr if it differs from the title, else ISO_ABBR below.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:23119/api/users/0"
COLLECTION = "JESTIE SS 2026 - Trustworthy Vibration FD"
MANUSCRIPT = Path(__file__).resolve().parents[2] / "05_manuscript_JESTIE"
ISO_ABBR = {
    "Data in Brief": "Data Brief",
    "Mechanical Systems and Signal Processing": "Mech. Syst. Signal Process.",
    "Signal Processing": "Signal Process.",
    "IEEE Internet of Things Journal": "IEEE Internet Things J.",
    "IEEE Sensors Journal": "IEEE Sensors J.",
    "IEEE Transactions on Industrial Informatics": "IEEE Trans. Ind. Informat.",
    "Frontiers in Neurorobotics": "Front. Neurorobot.",
    "Machine Learning": "Mach. Learn.",
    "Journal of Machine Learning Research": "J. Mach. Learn. Res.",
    "IEEE Transactions on Acoustics, Speech, and Signal Processing": "IEEE Trans. Acoust., Speech, Signal Process.",
    "Results in Engineering": "Results Eng.",
    "IEEE Transactions on Industrial Electronics": "IEEE Trans. Ind. Electron.",
    "IEEE Transactions on Control Systems Technology": "IEEE Trans. Control Syst. Technol.",
    "Mechanisms and Machine Science": "Mech. Mach. Sci.",
}
FULL = {v: k for k, v in ISO_ABBR.items()}
ACCENTS = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303"}


# --------------------------------------------------------------------------- local API
def _request(method, path, body=None, key=None):
    headers = {"Content-Type": "application/json", "Zotero-API-Version": "3"}
    if method != "GET":
        headers["Zotero-Server-ID"] = _server_id()
        headers["Zotero-API-Key"] = key or os.environ["ZOTERO_LOCAL_KEY"]
    req = urllib.request.Request(API + path, method=method, headers=headers,
                                 data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=60) as r:
        txt = r.read().decode()
        return (json.loads(txt) if txt.strip() else None), r.headers


def _server_id():
    req = urllib.request.Request(API + "/collections", method="POST", data=b"[]",
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as e:
        return e.headers.get("Zotero-Server-ID", "")
    return ""


def get_all(path):
    out, start = [], 0
    while True:
        sep = "&" if "?" in path else "?"
        page, hdr = _request("GET", f"{path}{sep}limit=100&start={start}")
        out += page
        start += 100
        if start >= int(hdr.get("Total-Results", len(out))):
            return out


def find_collection():
    return next((c for c in get_all("/collections") if c["data"]["name"] == COLLECTION), None)


# --------------------------------------------------------------------------- BibTeX in
def parse_bib(path):
    txt = open(path, encoding="utf-8").read()
    out = {}
    for typ, key, body in re.findall(r"@(\w+)\{([^,\s]+),(.*?)\n\}", txt, re.S):
        out[key] = (typ.lower(), dict(re.findall(r"^\s*(\w+)\s*=\s*\{(.*)\},?\s*$", body, re.M)))
    return out


def cited_keys(tex):
    keys = []
    for group in re.findall(r"\\cite\{([^}]*)\}", open(tex, encoding="utf-8").read()):
        for k in group.split(","):
            if k.strip() not in keys:
                keys.append(k.strip())
    return keys


def latex_to_unicode(s):
    s = re.sub(r"\{\\(['`^\"~])\{?(\w)\}?\}", lambda m: m.group(2) + ACCENTS[m.group(1)], s)
    s = re.sub(r"\\(['`^\"~])\{?(\w)\}?", lambda m: m.group(2) + ACCENTS[m.group(1)], s)
    import unicodedata
    return unicodedata.normalize("NFC", s)


def title_to_zotero(t):
    t = latex_to_unicode(t).replace("---", "—").replace("--", "–")
    return re.sub(r"\{([^{}]*)\}", r'<span class="nocase">\1</span>', t)


def creators(field, ctype="author"):
    out = []
    for name in latex_to_unicode(field).split(" and "):
        name = name.strip()
        if name == "others":
            continue
        last, _, first = name.partition(",")
        out.append({"creatorType": ctype, "lastName": last.strip(), "firstName": first.strip()})
    return out


def to_zotero(key, typ, f, journal_full=None):
    extra = [f"Citation Key: {key}"]
    if " and others" in f.get("author", ""):
        extra.append("BibTeX: and others")
    item = {"title": title_to_zotero(f["title"]), "creators": creators(f.get("author", "")),
            "date": f.get("year", ""), "tags": [], "relations": {}}
    pages = f.get("pages", "").replace("--", "-")
    if typ == "article":
        item.update(itemType="journalArticle", publicationTitle=journal_full or FULL.get(f["journal"], f["journal"]),
                    journalAbbreviation=f["journal"], volume=f.get("volume", ""), issue=f.get("number", ""),
                    pages=pages, DOI=f.get("doi", ""))
    elif typ == "incollection":
        item.update(itemType="bookSection", bookTitle=f["booktitle"], series=FULL.get(f["series"], f["series"]),
                    volume=f.get("volume", ""), publisher=f.get("publisher", ""), place=f.get("address", ""),
                    pages=pages)
        item["creators"] += creators(f.get("editor", ""), "editor")
        if f.get("doi"):
            extra.append(f"DOI: {f['doi']}")
    elif typ == "book":
        item.update(itemType="book", publisher=f.get("publisher", ""), place=f.get("address", ""))
        if f.get("doi"):
            extra.append(f"DOI: {f['doi']}")
    elif typ == "inproceedings":
        item.update(itemType="conferencePaper", proceedingsTitle=f["booktitle"])
    elif typ == "misc" and f.get("howpublished", "").startswith("arXiv:"):
        item.update(itemType="preprint", repository="arXiv", archiveID=f["howpublished"])
    else:
        raise ValueError(f"{key}: unsupported entry type {typ}")
    item["extra"] = "\n".join(extra)
    return item


def crossref_container(doi):
    mailto = os.environ.get("CROSSREF_MAILTO")          # optional contact address for Crossref's polite pool
    req = urllib.request.Request(f"https://api.crossref.org/works/{doi}",
                                 headers={"User-Agent": "zotero_refs/1.0" + (f" (mailto:{mailto})" if mailto else "")})
    try:
        m = json.load(urllib.request.urlopen(req, timeout=30))["message"]
        return (m.get("container-title") or [None])[0]
    except Exception:
        return None


def push():
    bib = parse_bib(MANUSCRIPT / "references.bib")
    keys = cited_keys(MANUSCRIPT / "main.tex")
    coll = find_collection()
    if coll is None:
        res, _ = _request("POST", "/collections", [{"name": COLLECTION}])
        ckey = res["successful"]["0"]["key"] if "successful" in res else res["success"]["0"]
        print("created collection", COLLECTION, ckey)
    else:
        ckey = coll["key"]
        print("collection exists", ckey)
    norm = lambda s: re.sub(r"[^a-z0-9]", "", re.sub(r"<[^>]+>", "", s or "").lower())
    lib = [i for i in get_all("/items") if i["data"]["itemType"] not in ("attachment", "note", "annotation")]
    by_doi = {norm(i["data"].get("DOI")): i for i in lib if i["data"].get("DOI")}
    by_title = {norm(i["data"].get("title"))[:60]: i for i in lib}
    # one POST for everything: a local key without "remember" is valid for a single write
    batch = []
    for k in keys:
        typ, f = bib[k]
        hit = (by_doi.get(norm(f.get("doi"))) if f.get("doi") else None) or by_title.get(norm(f["title"].replace("{", "").replace("}", ""))[:60])
        if hit:
            d = hit["data"]
            extra = d.get("extra") or ""
            upd = {"key": hit["key"], "version": hit["version"], "collections": sorted(set(d["collections"]) | {ckey})}
            if f"Citation Key: {k}" not in extra:
                upd["extra"] = (extra + "\n" if extra else "") + f"Citation Key: {k}"
            batch.append((k, upd, f"existing item {hit['key']} added to the collection"))
        else:
            full = crossref_container(f["doi"]) if typ == "article" and f.get("doi") else None
            item = to_zotero(k, typ, f, full)
            item["collections"] = [ckey]
            batch.append((k, item, "created"))
    res, _ = _request("POST", "/items", [it for _, it, _ in batch])
    ok = {**res.get("successful", {}), **res.get("success", {}), **res.get("unchanged", {})}
    for j, (k, _, msg) in enumerate(batch):
        print(f"  {k:24s} {msg if str(j) in ok else 'FAILED ' + str(res.get('failed', {}).get(str(j)))}")


# --------------------------------------------------------------------------- BibTeX out
def _protect_acronyms(t):
    """Brace words (hyphen parts) with two or more capitals (AI, IoT, ESP32, MLPerf) outside existing braces."""
    parts = re.split(r"(\{[^{}]*\})", t)
    acro = re.compile(r"(?<![A-Za-z0-9{])((?=[A-Za-z0-9]*[A-Z][A-Za-z0-9]*[A-Z])[A-Za-z0-9]+)")
    return "".join(p if p.startswith("{") else acro.sub(r"{\1}", p) for p in parts)


def _bib_title(t):
    t = re.sub(r'<span class="nocase">(.*?)</span>', r"{\1}", t)
    t = re.sub(r"</?(i|b|sub|sup|span)[^>]*>", "", t)
    return _protect_acronyms(t).replace("&", r"\&")


def _names(cs, ctype):
    return " and ".join(f"{c['lastName']}, {c['firstName']}".strip(", ") if "lastName" in c else "{" + c["name"] + "}"
                        for c in cs if c["creatorType"] == ctype)


def bib():
    coll = find_collection()
    if coll is None:
        sys.exit(f"collection '{COLLECTION}' not found; run 'push' first")
    items = [i["data"] for i in get_all(f"/collections/{coll['key']}/items/top")]
    entries = []
    for d in sorted(items, key=lambda d: d.get("citationKey") or d.get("extra") or ""):
        extra = d.get("extra") or ""
        m = re.search(r"^Citation Key:\s*(\S+)", extra, re.M)
        key = d.get("citationKey") or (m.group(1) if m else None)
        if not key:
            print(f"  skipped (no Citation Key): {d.get('title')}")
            continue
        authors = _names(d["creators"], "author") + (" and others" if "BibTeX: and others" in extra else "")
        year = (re.search(r"\d{4}", d.get("date", "")) or [""])[0]
        doi = d.get("DOI") or (re.search(r"^DOI:\s*(\S+)", extra, re.M) or [None, ""])[1]
        t = d["itemType"]
        f = [("author", authors), ("title", _bib_title(d["title"]))]
        if t == "journalArticle":
            pub = d.get("publicationTitle", "")
            abbr = d.get("journalAbbreviation") or ""
            journal = abbr if abbr and abbr != pub else ISO_ABBR.get(pub, pub)
            f += [("journal", journal), ("volume", d.get("volume")), ("number", d.get("issue")),
                  ("pages", (d.get("pages") or "").replace("-", "--")), ("year", year), ("doi", doi)]
            kind = "article"
        elif t == "bookSection":
            f += [("booktitle", d.get("bookTitle")), ("series", ISO_ABBR.get(d.get("series", ""), d.get("series"))),
                  ("volume", d.get("volume")), ("editor", _names(d["creators"], "editor")),
                  ("publisher", d.get("publisher")), ("address", d.get("place")),
                  ("pages", (d.get("pages") or "").replace("-", "--")), ("year", year), ("doi", doi)]
            kind = "incollection"
        elif t == "book":
            f += [("publisher", d.get("publisher")), ("address", d.get("place")), ("year", year), ("doi", doi)]
            kind = "book"
        elif t == "conferencePaper":
            f += [("booktitle", d.get("proceedingsTitle")), ("year", year), ("doi", doi)]
            kind = "inproceedings"
        elif t == "preprint":
            f += [("howpublished", d.get("archiveID")), ("year", year)]
            kind = "misc"
        else:
            print(f"  skipped (item type {t}): {key}")
            continue
        body = ",\n".join(f"  {k:<12s} = {{{v}}}" for k, v in f if v)
        entries.append(f"@{kind}{{{key},\n{body}\n}}")
    out = MANUSCRIPT / "references.bib"
    header = (f"% Generated from the Zotero collection \"{COLLECTION}\" by\n"
              "%   python -m tools.zotero_refs bib   (in 03_code). Edit the references in Zotero, not here.\n\n")
    out.write_text(header + "\n\n".join(entries) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(entries)} entries)")


if __name__ == "__main__":
    {"push": push, "bib": bib}[sys.argv[1]]()
