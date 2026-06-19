import csv
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIT_DIR = ROOT / "literature"
PDF_DIR = LIT_DIR / "open_access_pdfs"
MATRIX = LIT_DIR / "literature_matrix.csv"

QUERIES = [
    'all:"conformal prediction" AND all:"time series"',
    'all:"conformal prediction" AND all:"distribution shift"',
    'all:"adaptive conformal inference"',
    'all:"conformal prediction" AND all:"covariate shift"',
    'all:"conformal prediction" AND all:"non-exchangeable"',
    'all:"conformal prediction" AND all:"optimal transport"',
    'all:"conformal prediction" AND all:"Wasserstein"',
    'all:"conformal prediction" AND all:"streaming"',
    'all:"conformal prediction" AND all:"online prediction"',
    'all:"conformal prediction" AND all:"sequential"',
]

NS = {"atom": "http://www.w3.org/2005/Atom"}


def clean_name(text, max_len=120):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^A-Za-z0-9._ -]+", "", text)
    text = text.replace(" ", "_")
    return text[:max_len].strip("._-") or "paper"


def arxiv_id_from_url(url):
    return url.rstrip("/").split("/")[-1]


def fetch(query, start=0, max_results=25):
    params = {
        "search_query": query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=45) as response:
        return response.read()


def parse_entries(xml_bytes):
    root = ET.fromstring(xml_bytes)
    for entry in root.findall("atom:entry", NS):
        title = " ".join(entry.findtext("atom:title", "", NS).split())
        summary = " ".join(entry.findtext("atom:summary", "", NS).split())
        published = entry.findtext("atom:published", "", NS)[:10]
        abs_url = entry.findtext("atom:id", "", NS)
        arxiv_id = arxiv_id_from_url(abs_url)
        authors = [
            " ".join(a.findtext("atom:name", "", NS).split())
            for a in entry.findall("atom:author", NS)
        ]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        yield {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": "; ".join(authors),
            "published": published,
            "abstract_url": abs_url,
            "pdf_url": pdf_url,
            "summary": summary,
        }


def download_pdf(row):
    filename = f"{row['published']}_{clean_name(row['title'])}_{row['arxiv_id'].replace('/', '_')}.pdf"
    path = PDF_DIR / filename
    if path.exists() and path.stat().st_size > 10_000:
        return path.name, "exists"
    request = urllib.request.Request(row["pdf_url"], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    if len(data) < 10_000:
        return "", "small-or-empty"
    path.write_bytes(data)
    return path.name, "downloaded"


def main():
    LIT_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)

    seen = {}
    for query in QUERIES:
        if len(seen) >= 60:
            break
        print(f"Searching: {query}")
        try:
            xml_bytes = fetch(query, max_results=25)
            for row in parse_entries(xml_bytes):
                seen.setdefault(row["arxiv_id"], row)
        except Exception as exc:
            print(f"Query failed: {query}: {exc}")
        time.sleep(3)

    rows = list(seen.values())[:50]
    completed = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx:02d}/{len(rows)}] {row['title']}")
        try:
            local_file, status = download_pdf(row)
        except Exception as exc:
            local_file, status = "", f"failed: {exc}"
        row = dict(row)
        row["local_file"] = local_file
        row["download_status"] = status
        completed.append(row)
        time.sleep(2)

    with MATRIX.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "arxiv_id",
                "title",
                "authors",
                "published",
                "abstract_url",
                "pdf_url",
                "local_file",
                "download_status",
                "summary",
            ],
        )
        writer.writeheader()
        writer.writerows(completed)

    print(f"Wrote {MATRIX}")
    print(f"Downloaded/checked {sum(1 for r in completed if r['local_file'])} PDFs")


if __name__ == "__main__":
    main()
