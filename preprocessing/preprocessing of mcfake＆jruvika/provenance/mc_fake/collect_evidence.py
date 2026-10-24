"""Read-only source investigation; writes only within this evidence directory."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pypdf


OUT = Path(__file__).resolve().parent
WORK = OUT.parent.parent
SOURCE = WORK / "MC-Fake" / "MC_Fake_text.csv"
URLS = {
    "official_github_README.md": "https://raw.githubusercontent.com/qwerfdsaplking/MC-Fake/master/README.md",
    "official_nactem_page.html": "https://www.nactem.ac.uk/mc-fake/",
    "Min_et_al_WWW2022.pdf": "https://www.nactem.ac.uk/mc-fake/Min_et_al.pdf",
}
HEAD_URLS = [
    "https://www.nactem.ac.uk/data/MC_Fake_dataset.csv.gz",
    "https://nactem.ac.uk/data/MC_Fake_dataset.csv.gz",
    "https://www.nactem.ac.uk/mc-fake/MC_Fake_dataset.csv.gz",
    "https://drive.usercontent.google.com/download?id=1WAyOVLqM1JOQdlpsCdKKuRltOGVC9bGt&export=download&confirm=t",
]
DOMAIN_MAP = {
    "politifact": "politics",
    "RealPolitics": "politics",
    "gossipcop": "entertainment",
    "HealthStory": "health",
    "HealthRelease": "health",
    "RealHealth": "health",
    "FakeCovidClaimFiltered": "covid19",
    "FakeCovid": "covid19",
    "RealCovid": "covid19",
    "FA-KES": "syria_war",
    "RealSyria": "syria_war",
}


def fetch(item):
    name, url = item
    with urllib.request.urlopen(url, timeout=40) as response:
        content = response.read()
        (OUT / name).write_bytes(content)
        return name, {
            "requested_url": url,
            "resolved_url": response.geturl(),
            "status": response.status,
            "headers": dict(response.headers),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }


def head(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=35) as response:
            return {"requested_url": url, "resolved_url": response.geturl(), "status": response.status, "headers": dict(response.headers)}
    except urllib.error.HTTPError as error:
        return {"requested_url": url, "resolved_url": error.geturl(), "status": error.code, "headers": dict(error.headers), "error": str(error)}
    except Exception as error:
        return {"requested_url": url, "error": str(error)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        documents = dict(pool.map(fetch, URLS.items()))
        head_checks = list(pool.map(head, HEAD_URLS))
    pages = [page.extract_text() for page in pypdf.PdfReader(OUT / "Min_et_al_WWW2022.pdf").pages]
    (OUT / "paper_excerpts.txt").write_text(
        "SOURCE: https://www.nactem.ac.uk/mc-fake/Min_et_al.pdf\n\n"
        + "\n\n".join(f"PDF PAGE {i+1}\n{pages[i]}" for i in [2, 4, 9]), encoding="utf-8"
    )
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    cross = Counter((row["data_name"], row["labels"]) for row in rows)
    domain_cross = Counter((DOMAIN_MAP[row["data_name"]], row["labels"]) for row in rows)
    metadata = json.loads((WORK / "MC-Fake" / "metadata.json").read_text(encoding="utf-8"))
    checks = [
        {"domain": "entertainment", "paper_fake": 2587, "paper_real": 8846},
        {"domain": "health", "paper_fake": 590, "paper_real": 5120},
        {"domain": "covid19", "paper_fake": 843, "paper_real": 5393},
        {"domain": "syria_war", "paper_fake": 194, "paper_real": 2230},
        {"domain": "politics", "paper_fake": 225, "paper_real": 1026},
    ]
    for check in checks:
        check["current_label_1"] = domain_cross[(check["domain"], "1")]
        check["current_label_0"] = domain_cross[(check["domain"], "0")]
        check["fake_count_agrees"] = check["paper_fake"] == check["current_label_1"]
        check["real_count_agrees"] = check["paper_real"] == check["current_label_0"]
    findings = {
        "dataset": "MC-Fake",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_repository": "https://github.com/qwerfdsaplking/MC-Fake",
        "official_dataset_page": URLS["official_nactem_page.html"],
        "paper": {"title": "Divide-and-Conquer: Post-User Interaction Network for Fake News Detection on Social Media", "year": 2022, "doi": "10.1145/3485447.3512163", "url": URLS["Min_et_al_WWW2022.pdf"], "label_count_table_pdf_page": 5},
        "documents": documents,
        "download_head_checks": head_checks,
        "official_fields_verbatim": {"title": "title of the news", "text": "text content of the news", "labels": "veracity label of the news", "data_name": "news category"},
        "numeric_label_verification": {
            "mapping": {"0": "legitimate/real", "1": "fake"},
            "unified_mapping": {"0": 0, "1": 1},
            "method": "Cross-verification against the original paper Table 1: label 1 matches every fake-news count; label 0 matches real-news counts in four domains. Current politics real count differs from the paper version.",
            "limitation": "The official column documentation calls labels a veracity label but does not explicitly state the numeric encoding. The direction is verified by independent table counts rather than an unsupported convention assumption.",
            "table_1_cross_checks": checks,
            "do_not_infer_labels_from_data_name": "FakeCovid contains both 393 label-0 real articles and 284 label-1 fake articles. data_name is a category/subsource and is only used for domain analysis.",
        },
        "local_input": {"path": str(SOURCE), "bytes": SOURCE.stat().st_size, "sha256": hashlib.file_digest(SOURCE.open("rb"), "sha256").hexdigest(), "columns": columns, "rows": len(rows), "unique_news_ids": len({row["news_id"] for row in rows}), "label_counts": dict(Counter(row["labels"] for row in rows)), "empty_body_rows": sum(not row["text"].strip() for row in rows)},
        "existing_download_provenance": {k: metadata[k] for k in ["source_download_url", "downloaded_at_utc", "source_bytes", "source_sha256", "output_sha256"]},
        "data_name_domain_mapping": DOMAIN_MAP,
        "data_name_label_counts": [{"data_name": name, "domain": DOMAIN_MAP[name], "label_0_real": cross[(name, "0")], "label_1_fake": cross[(name, "1")]} for name in sorted(DOMAIN_MAP)],
        "domain_label_counts": [{"domain": domain, "label_0_real": domain_cross[(domain, "0")], "label_1_fake": domain_cross[(domain, "1")]} for domain in sorted(set(DOMAIN_MAP.values()))],
        "text_policy": "Use title plus text only. Retain all 202 empty-body rows because their titles are present. Do not use source/url/social fields.",
        "excluded_fields": ["url", "publish_date", "source", "n_tweets", "n_retweets", "n_replies", "n_users", "tweet_ids", "retweet_ids", "reply_ids", "user_ids", "retweet_relations", "reply_relations"],
        "social_network_policy": "No Twitter API calls, tweet/user hydration or follow-network downloads are needed. No social information is present in the existing text projection.",
        "version_note": "The official README/page state 28334 news events, but the current author-hosted CSV projection has 28335 unique news IDs. Do not remove a valid row to force the documented count. HEAD metadata alone cannot prove byte equality of two download versions.",
        "license": "CC BY 4.0; attribute NaCTeM, University of Manchester, and cite Min et al. WWW 2022.",
    }
    (OUT / "findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"documents": {name: value["bytes"] for name, value in documents.items()}, "head": [{"url": value["requested_url"], "status": value.get("status"), "content_length": value.get("headers", {}).get("Content-Length"), "error": value.get("error")} for value in head_checks], "rows": len(rows), "domains": findings["domain_label_counts"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
