"""Lightly normalize MC-Fake/Jruvika and export AMT/MultiFC-compatible profiles.

Python 3.10+, standard library only. Source files remain read-only. The split is
a new convenience split, not either dataset's official benchmark partition.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import unicodedata

ROOT = Path(__file__).resolve().parent
BINARY_FIELDS = ["id", "text", "label", "label_id", "original_label", "domain", "group_id", "split"]
UNIFIED_FIELDS = ["id", "text", "label", "domain", "dataset"]
SPLITS = ("train", "dev", "test")
RATIOS = (0.8, 0.1, 0.1)
DATASETS = {"mc_fake": "MC-Fake", "jruvika": "Jruvika-Fake-News"}
MC_DOMAINS = {
    "FakeCovidClaimFiltered": "health", "FakeCovid": "health", "RealCovid": "health",
    "HealthRelease": "health", "HealthStory": "health", "RealHealth": "health",
    "politifact": "politics", "RealPolitics": "politics", "FA-KES": "conflict",
    "RealSyria": "conflict", "gossipcop": "entertainment",
}
INPUTS = {
    "source_text/MC_Fake_text.csv": "adfee5fcc0953cede73da8271970cb4ab9c9f638b46f8cd8ef6cd478ff33ff2e",
    "source_text/Jruvika_text.csv": "b1e572834f3b95a55f01f9b707050b47a14dc4c1b59a1088fdb56be4501e2aa1",
    "audit/Jruvika_domain_and_labels.csv": "3efb3c09e7a5245487ba8fa6f3ca5576363935d8e96d0042355a38bc4f9b3f0d",
}
AUDIT_FIELDS = ["id", "dataset", "source_row", "original_label", "label", "label_id",
                "domain", "domain_original", "domain_method", "domain_needs_review", "data_name",
                "group_id", "split", "eligible", "exclusion_reason", "text", "text_raw",
                "title", "body", "title_raw", "body_raw", "body_empty", "body_repetition_count",
                "word_count", "character_count", "replacement_characters", "literal_escape_xa0",
                "peer_train_anchor"]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(stream)


def _digest(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def clean(text):
    """Same model-text normalization as the pinned AMT/MultiFC implementation."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def key(text):
    """Casefold only for overlap/group checks; model text preserves its case."""
    return clean(text).casefold()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path, fields=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if fields is not None:
            require(reader.fieldnames == fields, f"Unexpected columns: {path}")
        rows = list(reader)
    require(all(None not in row and all(value is not None for value in row.values()) for row in rows),
            f"Malformed CSV records: {path}")
    return rows


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    require(read_csv(path, fields) == [{field: str(row.get(field, "")) for field in fields} for row in rows],
            f"CSV readback mismatch: {path}")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with Path(path).open(encoding="utf-8") as stream:
        require([json.loads(line) for line in stream] == rows, f"JSONL readback mismatch: {path}")


def load_sources(data_root):
    for relative, expected in INPUTS.items():
        require(sha256(data_root / relative) == expected, f"Source checksum mismatch: {relative}")
    mc = read_csv(data_root / "source_text/MC_Fake_text.csv", ["news_id", "title", "text", "labels", "data_name"])
    jr = read_csv(data_root / "source_text/Jruvika_text.csv", ["Headline", "Body", "Label"])
    topics = read_csv(data_root / "audit/Jruvika_domain_and_labels.csv")
    require(len(mc) == 28335 and len(jr) == len(topics) == 4009, "Source record count mismatch")
    require(Counter(row["labels"] for row in mc) == {"0": 23896, "1": 4439}, "MC native labels changed")
    require(Counter(row["Label"] for row in jr) == {"0": 2137, "1": 1872}, "Jruvika native labels changed")
    rows = []
    for dataset_key, source in (("mc_fake", mc), ("jruvika", jr)):
        for index, row in enumerate(source, start=1):
            is_mc = dataset_key == "mc_fake"
            identifier = "MC-Fake:" + row["news_id"] if is_mc else f"Jruvika-Fake-News:{index:06d}"
            title_raw, body_raw = (row["title"], row["text"]) if is_mc else (row["Headline"], row["Body"])
            original = row["labels"] if is_mc else row["Label"]
            label = int(original) if is_mc else 1 - int(original)
            native_domain = MC_DOMAINS[row["data_name"]] if is_mc else topics[index - 1]["domain"]
            if not is_mc:
                require(topics[index - 1]["id"] == identifier, "Jruvika topic source order changed")
                require(int(topics[index - 1]["unified_label"]) == label, "Jruvika verified label mismatch")
            raw = title_raw + "\n\n" + body_raw
            text = clean(raw)
            rows.append({
                "id": identifier, "dataset": DATASETS[dataset_key], "source_row": index,
                "original_label": original, "label": label, "label_id": label,
                "domain": "technology" if native_domain == "science_technology" else native_domain,
                "domain_original": native_domain,
                "domain_method": "native_data_name_mapping" if is_mc else "inherited_text_rules_keyword-topics-1.1",
                "domain_needs_review": 0 if is_mc else int(topics[index - 1]["needs_review"]),
                "data_name": row["data_name"] if is_mc else "", "group_id": "", "split": "unsplit",
                "eligible": 1, "exclusion_reason": "", "text": text, "text_raw": raw,
                "title": clean(title_raw), "body": clean(body_raw), "title_raw": title_raw, "body_raw": body_raw,
                "body_empty": int(not body_raw.strip()), "body_repetition_count": 0,
                "word_count": len(text.split()), "character_count": len(text),
                "replacement_characters": raw.count("\ufffd"), "literal_escape_xa0": raw.count("\\xa0"),
                "peer_train_anchor": 0,
            })
    require(len({row["id"] for row in rows}) == len(rows), "Duplicate source IDs")
    counts = Counter(key(row["body"]) for row in rows if row["body"])
    for row in rows:
        row["body_repetition_count"] = counts[key(row["body"])] if row["body"] else 0
    return rows


def prepare_groups(rows, reference_overlap):
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    seen = {}
    by_text = defaultdict(list)
    for index, row in enumerate(rows):
        if row["text"]:
            by_text[key(row["text"])].append(index)
        for kind, content in (("text", row["text"]), ("title", row["title"]), ("body", row["body"])):
            if not content or (kind == "body" and len(content) < 80):
                continue
            token = (kind, key(content))
            if token in seen:
                union(index, seen[token])
            else:
                seen[token] = index
    conflicting = []
    for members in by_text.values():
        if len({rows[index]["label_id"] for index in members}) > 1:
            conflicting.append([rows[index]["id"] for index in members])
            for index in members:
                rows[index]["eligible"] = 0
                rows[index]["exclusion_reason"] = "conflicting_targets_for_same_normalized_text"
    for row in rows:
        if not row["text"]:
            row["eligible"], row["exclusion_reason"] = 0, "empty_text"
    components = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    groups = {}
    for members in components.values():
        ids = sorted(rows[index]["id"] for index in members)
        group_id = "news-" + hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        for index in members:
            rows[index]["group_id"] = group_id
        eligible = [index for index in members if rows[index]["eligible"]]
        if eligible:
            groups[group_id] = eligible
    anchors = set()
    for match in reference_overlap.get("matches", []):
        # The reference audit may identify headline/claim matches; they are not
        # assumed to be identical articles or newly verified factual labels.
        if match.get("peer_split") == "train":
            anchors.add(match["local_id"])
    require(anchors.issubset({row["id"] for row in rows}), "Reference anchor has an unknown local ID")
    for row in rows:
        row["peer_train_anchor"] = int(row["id"] in anchors)
    return groups, conflicting


def assign_splits(rows, groups, seed):
    strata = Counter((row["dataset"], row["domain"], row["label_id"]) for row in rows if row["eligible"])
    total = sum(strata.values())
    current = {split: Counter() for split in SPLITS}
    sizes = Counter()
    targets = {split: {stratum: count * ratio for stratum, count in strata.items()}
               for split, ratio in zip(SPLITS, RATIOS)}
    assignments = {}
    vectors = {gid: Counter((rows[i]["dataset"], rows[i]["domain"], rows[i]["label_id"])
                            for i in members) for gid, members in groups.items()}
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda gid: -len(groups[gid]))
    anchored = {gid for gid, members in groups.items() if any(rows[i]["peer_train_anchor"] for i in members)}
    # Assign existing peer-training headline matches before optimising new splits.
    for gid in sorted(anchored):
        assignments[gid] = "train"
        current["train"].update(vectors[gid])
        sizes["train"] += len(groups[gid])
    for gid in order:
        if gid in assignments:
            continue
        vector, size = vectors[gid], len(groups[gid])

        def cost(split):
            delta = sum((2 * (current[split][s] - targets[split][s]) * n + n * n) / strata[s]
                        for s, n in vector.items())
            desired = total * RATIOS[SPLITS.index(split)]
            delta += 0.1 * (2 * (sizes[split] - desired) * size + size * size) / total
            return delta, SPLITS.index(split)

        split = min(SPLITS, key=cost)
        assignments[gid] = split
        current[split].update(vector)
        sizes[split] += size
    for row in rows:
        row["split"] = assignments[row["group_id"]] if row["eligible"] else "excluded"
    verify_partitions(rows)
    return {"seed": seed, "requested_ratios": list(RATIOS), "actual_counts": dict(sizes),
            "group_count": len(groups), "largest_group": max(map(len, groups.values())),
            "peer_train_anchored_groups": len(anchored),
            "peer_train_anchor_rows": sum(row["peer_train_anchor"] for row in rows),
            "all_rows_in_peer_anchored_groups": sum(len(groups[gid]) for gid in anchored),
            "official_split": False,
            "method": "Greedy group assignment minimising squared dataset/domain/label-stratum count deviations; large groups first, seed-42 order for equal sizes. Existing peer train headline matches anchored to train.",
            "group_policy": "Connected components of equal NFC+whitespace+casefold full text, nonempty title, or body at least 80 characters. Content grouping is conservative and does not assert semantic equivalence."}


def verify_partitions(rows):
    seen = {"group": {}, "text": {}, "title": {}, "body": {}}
    for row in rows:
        if not row["eligible"]:
            continue
        require(type(row["label_id"]) is int and row["label"] == row["label_id"] in (0, 1), "Bad target type/mapping")
        require(clean(row["text"]) == row["text"] and row["text"], "Empty/noncanonical model text")
        values = {"group": row["group_id"], "text": key(row["text"]), "title": key(row["title"]),
                  "body": key(row["body"]) if len(row["body"]) >= 80 else ""}
        for kind, value in values.items():
            if value:
                prior = seen[kind].setdefault(value, row["split"])
                require(prior == row["split"], f"Cross-split {kind} overlap")
        require(not row["peer_train_anchor"] or row["split"] == "train", "Peer train anchor split mismatch")


def binary_record(row, split=None):
    return {field: split if field == "split" and split is not None else row[field] for field in BINARY_FIELDS}


def distribution(rows):
    return {"rows": len(rows), "label_counts": dict(sorted(Counter(r["label_id"] for r in rows).items())),
            "domains": dict(sorted(Counter(r["domain"] for r in rows).items())),
            "domain_label_counts": [{"domain": domain, "label": label, "rows": count}
                                    for (domain, label), count in sorted(Counter((r["domain"], r["label_id"]) for r in rows).items())]}


def numeric_summary(values):
    values = sorted(values)
    if not values:
        return {}
    return {"min": values[0], "mean": sum(values) / len(values),
            "median": (values[(len(values) - 1) // 2] + values[len(values) // 2]) / 2,
            "p95_nearest_rank": values[min(len(values) - 1, int(len(values) * 0.95))], "max": values[-1]}


def duplicate_summary(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if row[field]:
            groups[key(row[field])].append(row)
    duplicates = [members for members in groups.values() if len(members) > 1]
    return {"groups": len(duplicates), "rows_in_groups": sum(map(len, duplicates)),
            "extra_rows": sum(len(members) - 1 for members in duplicates),
            "conflicting_label_groups": sum(len({r["label_id"] for r in m}) > 1 for m in duplicates)}


def build_report(dataset_key, rows, global_split, conflicts, overlap):
    eligible = [row for row in rows if row["eligible"]]
    split_reports = {split: distribution([r for r in eligible if r["split"] == split]) for split in SPLITS}
    warnings = []
    for name, count in (("empty body, title retained", sum(r["body_empty"] for r in rows)),
                        ("existing U+FFFD replacement character", sum(bool(r["replacement_characters"]) for r in rows)),
                        ("literal backslash-xa0 escape", sum(bool(r["literal_escape_xa0"]) for r in rows)),
                        ("fewer than 20 whitespace words", sum(r["word_count"] < 20 for r in rows)),
                        ("inferred domain marked for review", sum(r["domain_needs_review"] for r in rows))):
        if count:
            warnings.append({"issue": name, "rows": count, "action": "retained and flagged in articles.csv"})
    ids = {row["id"] for row in rows}
    return {
        "dataset": DATASETS[dataset_key], "source": distribution(rows), "binary": distribution(eligible),
        "excluded_rows": len(rows) - len(eligible),
        "exclusions": dict(Counter(r["exclusion_reason"] for r in rows if not r["eligible"])),
        "normalization": "NFC and Unicode whitespace collapse only; preserve case, punctuation, numbers, stopwords and negation. Group keys separately casefold.",
        "text_raw": "Exact decoded title + two LF characters + exact decoded body, with title_raw/body_raw separately preserved.",
        "label_meaning": {"0": "real/legitimate", "1": "fake"},
        "original_numeric_mapping": {"0": 0, "1": 1} if dataset_key == "mc_fake" else {"0": 1, "1": 0},
        "label_mapping_evidence": "See original D:/NLP Group 43/provenance/; original numeric values, not the already unified labels, are mapped once.",
        "domain_definition": "News topic; unlike MultiFC source codes. technology is an alias of the earlier science_technology to align AMT's spelling.",
        "domain_method": "MC data_name mapping" if dataset_key == "mc_fake" else "Existing keyword-topics-1.1 text-only domains reused, not newly verified gold labels.",
        "domain_inference_needs_review_rows": sum(r["domain_needs_review"] for r in rows),
        "duplicates": {field: duplicate_summary(rows, field) for field in ("text", "title", "body")},
        "deduplication_policy": "Retain same-label within-split duplicates, matching teammate policy; jointly group both news datasets before splitting. Exclude all conflicting full-text binary targets if present; do not relabel title/body groups.",
        "word_counts": numeric_summary([r["word_count"] for r in rows]),
        "character_counts": numeric_summary([r["character_count"] for r in rows]),
        "word_count_note": "Whitespace words are diagnostic counts, not transformer tokens; no model-specific truncation applied.",
        "split": {**global_split, "dataset_actual": split_reports,
                  "dataset_actual_ratios": {s: split_reports[s]["rows"] / len(eligible) for s in SPLITS}},
        "cross_reference_audit": {"reference_commit": overlap.get("commit", overlap.get("repository_commit")),
                                  "matches_for_this_dataset": [m for m in overlap.get("matches", []) if m["local_id"] in ids],
                                  "scope": "Saved exact/casefold full-text and title-vs-claim checks against teammate binary profiles. Not a semantic near-duplicate audit."},
        "conflicting_text_group_ids": [group for group in conflicts if any(i in ids for i in group)],
        "replacement_character_ids": [r["id"] for r in rows if r["replacement_characters"]],
        "warnings": warnings,
        "feature_policy": "Use text only as input; label_id as target. No URL/social columns. id, dataset, domain, original_label, group_id and audit fields are not input features.",
        "validation": {"source_hashes_unchanged": True, "csv_and_jsonl_readback": True,
                       "binary_targets_integer": True, "no_group_text_title_or_long_body_cross_split_overlap": True,
                       "global_two_dataset_grouping": True, "peer_train_anchors_enforced": True},
    }


def verify_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == 3 and manifest.get("complete") is True, "Incomplete output manifest")
    for name, expected in manifest["files"].items():
        target = (path.parent / name).resolve()
        require(target.is_relative_to(path.parent.resolve()), "Manifest path escapes output directory")
        require(target.stat().st_size == expected["bytes"] and sha256(target) == expected["sha256"],
                f"Output checksum mismatch: {target}")
    return manifest


def load_preprocessed(dataset="mc_fake", profile="binary", output_dir=None):
    """Compatible handoff: {split: list[8-field records]}, with verified output hashes.

    Default binary is unsplit, just like AMT. Use profile='binary_grouped' for
    train/dev/test. All groups were assigned jointly across the two news sets.
    """
    require(dataset in DATASETS, f"Unknown dataset {dataset}")
    directory = Path(output_dir) if output_dir else ROOT / "output" / dataset
    manifest = verify_manifest(directory / "manifest.json")
    require(profile in manifest["profiles"], f"Unavailable profile: {profile}")
    result = {}
    for split, relative in manifest["profiles"][profile].items():
        with (directory / relative).open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        require(all(list(row) == BINARY_FIELDS and type(row["label"]) is int
                    and type(row["label_id"]) is int and row["label"] == row["label_id"] in (0, 1)
                    and row["split"] == split and row["text"] for row in records), "Loaded binary schema mismatch")
        result[split] = records
    return result


def write_dataset(output, dataset_key, rows, report, source_records, seed):
    require(not output.exists(), f"Output already exists; select a new --output-dir: {output}")
    output.mkdir(parents=True)
    eligible = [r for r in rows if r["eligible"]]
    write_csv(output / "articles.csv", rows, AUDIT_FIELDS)
    write_csv(output / "binary_excluded.csv", [r for r in rows if not r["eligible"]],
              ["id", "original_label", "label_id", "domain", "group_id", "exclusion_reason"])
    unsplit = [binary_record(row, "unsplit") for row in eligible]
    write_csv(output / "binary.csv", unsplit, BINARY_FIELDS)
    write_jsonl(output / "binary.jsonl", unsplit)
    write_csv(output / "unified.csv", eligible, UNIFIED_FIELDS)
    profiles = {"binary": {"unsplit": "binary.jsonl"}, "binary_grouped": {}}
    for split in SPLITS:
        selected = [binary_record(row) for row in eligible if row["split"] == split]
        require({row["label_id"] for row in selected} == {0, 1}, f"{dataset_key}/{split} lacks a binary class")
        write_csv(output / f"binary_{split}.csv", selected, BINARY_FIELDS)
        write_jsonl(output / f"binary_{split}.jsonl", selected)
        profiles["binary_grouped"][split] = f"binary_{split}.jsonl"
    write_json(output / "quality_report.json", report)
    files = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}
    manifest = {
        "schema_version": 3, "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASETS[dataset_key], "source_hashes": source_records,
        "script_sha256": sha256(__file__), "output_directory": str(output.resolve()),
        "profiles": profiles, "files": files,
        "next_stage": {"default_profile": "binary", "split_profile": "binary_grouped", "feature_text": "text",
                       "target": "label_id", "native_rating": "original_label", "group": "group_id",
                       "fit_vocabulary_on": "train only", "split_seed": seed,
                       "official_split": False, "shared_group_namespace": "MC-Fake+Jruvika-Fake-News"},
    }
    # Success marker is published only after every output passed readback.
    write_json(output / "manifest.json", manifest)
    verify_manifest(output / "manifest.json")
    print(f"[DONE] {dataset_key}: {output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--reference-overlap", type=Path, default=ROOT / "reference/cross_reference_overlap.json")
    parser.add_argument("--dataset", choices=["both", *DATASETS], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    csv.field_size_limit(2**31 - 1)
    selected = list(DATASETS) if args.dataset == "both" else [args.dataset]
    for dataset_key in selected:
        require(not (args.output_dir / dataset_key).exists(), "Output folder already exists; use a fresh --output-dir")
    source_records = {relative: {"sha256": sha256(args.data_root / relative),
                                 "bytes": (args.data_root / relative).stat().st_size} for relative in INPUTS}
    overlap = json.loads(args.reference_overlap.read_text(encoding="utf-8"))
    overlap_hash = sha256(args.reference_overlap)
    rows = load_sources(args.data_root)
    print(f"[PASS] Verified source hashes, 32,344 IDs, original labels and inherited domains", flush=True)
    groups, conflicts = prepare_groups(rows, overlap)
    global_split = assign_splits(rows, groups, args.seed)
    require(all(sha256(args.data_root / relative) == record["sha256"] for relative, record in source_records.items()),
            "Source changed during preprocessing")
    source_records["reference_overlap"] = {"sha256": overlap_hash,
                                            "bytes": args.reference_overlap.stat().st_size}
    print(f"[PASS] Joint group split verified: {global_split['actual_counts']}; largest group {global_split['largest_group']}", flush=True)
    for dataset_key in selected:
        dataset_rows = [row for row in rows if row["dataset"] == DATASETS[dataset_key]]
        report = build_report(dataset_key, dataset_rows, global_split, conflicts, overlap)
        write_dataset(args.output_dir / dataset_key, dataset_key, dataset_rows, report, source_records, args.seed)
        counts = {s: report["split"]["dataset_actual"][s]["rows"] for s in SPLITS}
        print(f"[BINARY] {dataset_key}: {len(dataset_rows):,} source rows; split counts {counts}; exclusions {report['excluded_rows']}", flush=True)
    require(all(sha256(args.data_root / relative) == record["sha256"] for relative, record in source_records.items() if relative in INPUTS),
            "Source changed after writing")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, csv.Error) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        sys.exit(1)
