"""Build auditable text-only MC-Fake/Jruvika datasets using only the standard library.

Run from the delivered folder:
    python preprocessing/normalize_datasets.py --output-dir rebuilt
The source-text files contain no social columns or URL column.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import unicodedata

from topic_classifier import classify

FIELDS = ["id", "text", "label", "domain", "dataset"]
DOMAINS = ["health", "politics", "entertainment", "conflict", "sports", "business",
           "science_technology", "environment", "society", "other"]
MC_DOMAINS = {
    "FakeCovidClaimFiltered": "health", "FakeCovid": "health", "RealCovid": "health",
    "HealthRelease": "health", "HealthStory": "health", "RealHealth": "health",
    "politifact": "politics", "RealPolitics": "politics",
    "FA-KES": "conflict", "RealSyria": "conflict", "gossipcop": "entertainment",
}
SOURCE_HASHES = {
    "MC_Fake_text.csv": "adfee5fcc0953cede73da8271970cb4ab9c9f638b46f8cd8ef6cd478ff33ff2e",
    "Jruvika_text.csv": "b1e572834f3b95a55f01f9b707050b47a14dc4c1b59a1088fdb56be4501e2aa1",
}


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path, fields):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        if reader.fieldnames != fields:
            raise ValueError(f"Unexpected fields in {path}: {reader.fieldnames}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed rows in {path}")
    return rows


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def combine(title, body):
    # Retain internal whitespace and wording; strip only outer whitespace.
    return "\n\n".join(part for part in (title.strip(), body.strip()) if part)


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def grouped_counts(rows, keys):
    result = Counter(tuple(str(row[key]) for key in keys) for row in rows)
    return [{**dict(zip(keys, key)), "count": count} for key, count in sorted(result.items())]


def build_duplicates(rows, titles):
    text_groups = defaultdict(list)
    title_groups = defaultdict(list)
    for row in rows:
        text_groups[canonical(row["text"])].append(row)
        title_groups[canonical(titles[row["id"]])].append(row)
    output = []
    for kind, groups in (("normalized_text", text_groups), ("normalized_title", title_groups)):
        for text, members in groups.items():
            if len(members) < 2:
                continue
            group_id = kind + "_" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            conflict = len({row["label"] for row in members}) > 1
            for row in members:
                output.append({"group_type": kind, "group_id": group_id,
                               "group_size": len(members), "label_conflict": int(conflict),
                               "id": row["id"], "dataset": row["dataset"]})
    stats = {}
    for kind, groups in (("normalized_text", text_groups), ("normalized_title", title_groups)):
        duplicated = [members for members in groups.values() if len(members) > 1]
        stats[kind] = {
            "duplicate_groups": len(duplicated),
            "rows_in_duplicate_groups": sum(map(len, duplicated)),
            "extra_rows": sum(len(members) - 1 for members in duplicated),
            "groups_with_conflicting_labels": sum(len({r["label"] for r in m}) > 1 for m in duplicated),
            "cross_dataset_duplicate_groups": sum(len({r["dataset"] for r in m}) > 1 for m in duplicated),
        }
    return output, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    package_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--mc-source", type=Path, default=package_root / "source_text/MC_Fake_text.csv")
    parser.add_argument("--jruvika-source", type=Path, default=package_root / "source_text/Jruvika_text.csv")
    parser.add_argument("--evidence-dir", type=Path, default=package_root / "provenance")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    csv.field_size_limit(2**31 - 1)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be absent or empty: {output}")
    for name, source in (("MC_Fake_text.csv", args.mc_source), ("Jruvika_text.csv", args.jruvika_source)):
        if sha256(source) != SOURCE_HASHES[name]:
            raise ValueError(f"Source checksum does not match the verified download: {source}")
    mc = read_csv(args.mc_source, ["news_id", "title", "text", "labels", "data_name"])
    jr = read_csv(args.jruvika_source, ["Headline", "Body", "Label"])
    if len(mc) != 28335 or len(jr) != 4009:
        raise ValueError("Unexpected number of source records")
    if set(Counter(row["data_name"] for row in mc)) != set(MC_DOMAINS):
        raise ValueError("MC-Fake includes an unverified source category")
    if Counter(row["labels"] for row in mc) != {"0": 23896, "1": 4439}:
        raise ValueError("Unexpected MC-Fake label counts")
    if Counter(row["Label"] for row in jr) != {"0": 2137, "1": 1872}:
        raise ValueError("Unexpected Jruvika label counts")

    body_counts = Counter(row["Body"].strip() for row in jr if row["Body"].strip())
    mc_rows, jr_rows, mc_audit, jr_audit = [], [], [], []
    titles = {}
    for index, row in enumerate(mc, start=1):
        identifier = "MC-Fake:" + row["news_id"]
        normalized = {"id": identifier, "text": combine(row["title"], row["text"]),
                      "label": int(row["labels"]), "domain": MC_DOMAINS[row["data_name"]],
                      "dataset": "MC-Fake"}
        mc_rows.append(normalized)
        titles[identifier] = row["title"]
        mc_audit.append({"id": identifier, "source_row": index, "news_id": row["news_id"],
                         "data_name": row["data_name"], "original_label": row["labels"],
                         "unified_label": normalized["label"], "domain": normalized["domain"],
                         "domain_method": "native_data_name_mapping",
                         "body_empty": int(not row["text"].strip())})
    for index, row in enumerate(jr, start=1):
        identifier = f"Jruvika-Fake-News:{index:06d}"
        repetitions = body_counts[row["Body"].strip()]
        ignore_body = repetitions >= 10
        topic = classify(row["Headline"], row["Body"], ignore_body=ignore_body)
        if topic["domain"] not in DOMAINS:
            raise ValueError(f"Unknown domain: {topic['domain']}")
        normalized = {"id": identifier, "text": combine(row["Headline"], row["Body"]),
                      "label": 1 - int(row["Label"]), "domain": topic["domain"],
                      "dataset": "Jruvika-Fake-News"}
        jr_rows.append(normalized)
        titles[identifier] = row["Headline"]
        jr_audit.append({"id": identifier, "source_row": index, "original_label": row["Label"],
                         "unified_label": normalized["label"], "domain": topic["domain"],
                         "domain_method": "text_topic_rules_v1", "score": topic["score"],
                         "runner_up": topic["runner_up"], "runner_up_score": topic["runner_up_score"],
                         "margin": topic["margin"],
                         "needs_review": int(topic["needs_review"]),
                         "matched_terms": json.dumps(topic["matched_terms"], ensure_ascii=False),
                         "runner_up_matched_terms": json.dumps(topic["runner_up_matched_terms"], ensure_ascii=False),
                         "title_domain": topic["title_domain"], "body_domain": topic["body_domain"],
                         "title_body_disagreement": int(topic["title_body_disagreement"]),
                         "body_empty": int(not row["Body"].strip()),
                         "body_repetition_count": repetitions,
                         "body_excluded_from_topic_input": int(ignore_body)})

    all_rows = mc_rows + jr_rows
    if len({row["id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Output IDs are not unique")
    if any(not row["text"] or row["label"] not in (0, 1) for row in all_rows):
        raise ValueError("Output text/label validation failed")
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("audit", "source_text", "preprocessing"):
        (output / directory).mkdir(exist_ok=True)
    if args.evidence_dir.is_dir():
        shutil.copytree(args.evidence_dir, output / "provenance")
    else:
        raise FileNotFoundError("Label verification evidence directory is required")
    for source, name in ((args.mc_source, "MC_Fake_text.csv"), (args.jruvika_source, "Jruvika_text.csv")):
        shutil.copy2(source, output / "source_text" / name)
    shutil.copy2(__file__, output / "preprocessing/normalize_datasets.py")
    shutil.copy2(Path(__file__).with_name("topic_classifier.py"), output / "preprocessing/topic_classifier.py")
    for name, rows in (("MC_Fake_unified.csv", mc_rows), ("Jruvika_Fake_News_unified.csv", jr_rows),
                       ("combined_unified.csv", all_rows)):
        path = output / name
        write_csv(path, FIELDS, rows)
        roundtrip = read_csv(path, FIELDS)
        if roundtrip != [{key: str(row[key]) for key in FIELDS} for row in rows]:
            raise ValueError(f"Output round-trip validation failed: {name}")
    write_csv(output / "audit/MC_Fake_domain_and_labels.csv", list(mc_audit[0]), mc_audit)
    write_csv(output / "audit/Jruvika_domain_and_labels.csv", list(jr_audit[0]), jr_audit)
    duplicates, duplication_stats = build_duplicates(all_rows, titles)
    write_csv(output / "audit/duplicate_groups.csv",
              ["group_type", "group_id", "group_size", "label_conflict", "id", "dataset"], duplicates)
    write_csv(output / "audit/domain_label_counts.csv", ["dataset", "domain", "label", "count"],
              grouped_counts(all_rows, ["dataset", "domain", "label"]))
    review = [row for row in jr_audit if row["needs_review"] or row["title_body_disagreement"]]
    review_rows = [{**row, "headline": jr[row["source_row"] - 1]["Headline"]} for row in review]
    write_csv(output / "audit/Jruvika_topic_review.csv", [*jr_audit[0], "headline"], review_rows)
    metadata = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema": FIELDS, "encoding": "UTF-8 with BOM", "labels": {"0": "legitimate/real", "1": "fake"},
        "label_mappings": {"MC-Fake": {"0": 0, "1": 1}, "Jruvika-Fake-News": {"0": 1, "1": 0}},
        "label_verification": {
            "MC-Fake": "Official paper Table 1 fake/real counts agree exactly with current 0/1 counts in four domains; official column documentation itself does not state the numeric direction.",
            "Jruvika-Fake-News": "Independent public code tied to Kaggle dataset assigns labels==0 to fake_data and labels==1 to real_data; its 2137/1872 counts match the source. Publisher metadata does not explicitly state numeric semantics.",
        },
        "rows": {"MC-Fake": len(mc_rows), "Jruvika-Fake-News": len(jr_rows), "combined": len(all_rows)},
        "label_counts": grouped_counts(all_rows, ["dataset", "label"]),
        "domain_counts": grouped_counts(all_rows, ["dataset", "domain"]),
        "domain_label_counts": grouped_counts(all_rows, ["dataset", "domain", "label"]),
        "mc_data_name_mapping": MC_DOMAINS,
        "mc_original_category_label_counts": grouped_counts(mc, ["data_name", "labels"]),
        "source_projection_sha256": SOURCE_HASHES,
        "source_urls": {
            "MC-Fake": ["https://github.com/qwerfdsaplking/MC-Fake", "https://www.nactem.ac.uk/mc-fake/",
                        "https://drive.google.com/file/d/1WAyOVLqM1JOQdlpsCdKKuRltOGVC9bGt/view"],
            "Jruvika-Fake-News": ["https://www.kaggle.com/datasets/jruvika/fake-news-detection",
                                  "https://www.kaggle.com/api/v1/datasets/download/jruvika/fake-news-detection/data.csv"],
        },
        "text_transformation": "Strip outer whitespace from title/headline and body separately; join nonempty parts with two LF characters. Retain wording and internal whitespace. No metadata/URL column appended.",
        "topic_classification": {
            "method": "Deterministic topic keyword/phrase rules, weighted headline and truncated body; scores are NOT probabilities. No supervised model or human gold validation.",
            "inputs": ["Headline", "Body"], "native_domain_available": False,
            "domains": DOMAINS,
            "body_template_exclusion_threshold": 10,
            "frequent_body_templates": sum(count >= 10 for count in body_counts.values()),
            "rows_with_template_body_excluded_from_topic_input": sum(r["body_excluded_from_topic_input"] for r in jr_audit),
            "needs_review_rows": sum(r["needs_review"] for r in jr_audit),
            "title_body_disagreement_rows": sum(r["title_body_disagreement"] for r in jr_audit),
            "review_file_rows": len(review),
        },
        "quality": {
            "empty_mc_body": sum(not r["text"].strip() for r in mc),
            "empty_jruvika_body": sum(not r["Body"].strip() for r in jr),
            "empty_merged_text": sum(not r["text"] for r in all_rows),
            "jruvika_unique_nonempty_body": len(body_counts),
            "jruvika_rows_with_repeated_nonempty_body": sum(count for count in body_counts.values() if count > 1),
            "duplicates": duplication_stats,
            "row_policy": "Preserve every source record, including duplicate, conflicting-label and empty-body records; no splits or label corrections beyond verified numeric remapping.",
        },
        "excluded_source_columns": ["URLs", "url", "publish_date", "source", "n_tweets", "n_retweets",
                                    "n_replies", "n_users", "tweet_ids", "retweet_ids", "reply_ids",
                                    "user_ids", "retweet_relations", "reply_relations"],
        "id_policy": "MC-Fake:<original news_id>; Jruvika-Fake-News:<one-based source row, six digits>.",
        "validation": "Source projection SHA256, exact field sets, expected row/label counts, all IDs unique, all texts nonempty, all labels binary, all domains allowed, CSV round-trip exact.",
    }
    write_json(output / "metadata.json", metadata)
    (output / "README.md").write_text(make_readme(metadata), encoding="utf-8")
    entries = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            entries.append({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
                            "sha256": sha256(path)})
    write_json(output / "manifest.json", {"files": entries})
    print(json.dumps({"output": str(output), "rows": metadata["rows"],
                      "topic_classification": metadata["topic_classification"],
                      "file_count": len(entries)}, ensure_ascii=True, indent=2))


def make_readme(m):
    counts = {(r["dataset"], r["label"]): r["count"] for r in m["label_counts"]}
    topic = m["topic_classification"]
    table = "\n".join(
        f"| {dataset} | {m['rows'][dataset]:,} | {counts[dataset, '0']:,} | {counts[dataset, '1']:,} |"
        for dataset in ("MC-Fake", "Jruvika-Fake-News"))
    mapping = "\n".join(f"| {source} | {domain} |" for source, domain in MC_DOMAINS.items())
    return f'''# MC-Fake 与 Jruvika Fake News 统一文本数据

统一字段顺序：`id,text,label,domain,dataset`。CSV 使用 UTF-8 BOM；正文可含换行，请使用 CSV 解析器读取。统一标签：**0=legitimate/real，1=fake**。

| 数据集 | 样本数 | 0：真实 | 1：虚假 |
| --- | ---: | ---: | ---: |
{table}

`MC_Fake_unified.csv` 和 `Jruvika_Fake_News_unified.csv` 为分别处理后的数据；`combined_unified.csv` 合并全部 **32,344** 条。`text` 为“标题 + 两个换行 + 正文”，某部分为空时仅保留非空部分。仅修剪两端空白，保留内部内容。`dataset` 为 `MC-Fake` 或 `Jruvika-Fake-News`。MC 的 ID 保留原 news_id 并加来源前缀；Jruvika 无原生 ID，按原文件数据行顺序生成稳定 ID。

## 标签核实与映射

MC-Fake 原始 `labels`：0=real，1=fake，直接映射。官方字段文档仅称其为真假标签，没有直接写数字含义；原论文 Table 1 中娱乐、健康、COVID、叙利亚四领域的真假计数与当前 CSV 交叉统计完全吻合，支持此方向。不要按 `FakeCovid` 等来源名称推断真假，该来源同时包含两种标签。

Jruvika 原始 `Label`：0=fake，1=real，因此统一时执行 `label = 1 - Label`。这依据是明确引用原 Kaggle 数据集的公开分析代码，其中 `fake_data=data[labels==0]`、`real_data=data[labels==1]`，输出 2,137/1,872 条，与本次原始数据一致；并用新闻样例交叉检查。Kaggle 发布者元数据未明确写数字含义，不能把第三方代码当作发布者声明。固定版本链接、保存的摘录及核实细节在 `provenance/`。

## 领域

两者使用同一组宽主题：`health` 健康与COVID、`politics` 政治、`entertainment` 娱乐、`conflict` 战争与武装冲突、`sports` 体育、`business` 商业经济、`science_technology` 科技、`environment` 环境与自然灾害、`society` 社会与司法、`other` 无法确定。

MC-Fake 使用原生 `data_name` 显式映射。原始 `data_name`、news_id、原标签逐行保存在 `audit/MC_Fake_domain_and_labels.csv`，可继续分析 COVID/健康、叙利亚等细分主题。主表 domain 不包含 Real/Fake 来源字样。

| 原始 data_name | 统一 domain |
| --- | --- |
{mapping}

Jruvika 没有原生领域标签，本版本以**确定性的文本主题规则**分类，只读取 Headline/Body。标题优先，正文截断；出现 ≥10 次的重复正文模板不参与分类，合并 text 仍保留原正文。这是自动推断的领域，不是人工金标准。得分是规则分数，不是置信概率；规则见 `preprocessing/topic_classifier.py`。共 {topic['needs_review_rows']:,} 条标记 needs_review；标题/正文主题分歧 {topic['title_body_disagreement_rows']:,} 条。需要复查的记录、命中词、前两主题和分差见 `audit/Jruvika_topic_review.csv`；全部分类记录见 `audit/Jruvika_domain_and_labels.csv`。

## 保留范围与质量

主表只有上述五列。未保留 tweets、retweets、replies、用户 ID、传播边或互动统计；Jruvika 原始 `URLs` 列未复制、未用于任何分类。原始字段的文本投影保存在 `source_text/`，也不含上述字段。标题/正文内原有的文本保持原样。

保留全部原始记录：MC-Fake 有 202 条空正文，Jruvika 有 21 条空正文，均有标题，因此统一后无空 text。当前 MC-Fake 原文件为 28,335 条，与官方历史说明的 28,334 条存在差异，本版本按实际文件保留。

Jruvika 有重复正文和标题与正文错配现象；{topic['frequent_body_templates']} 个正文模板共影响 {topic['rows_with_template_body_excluded_from_topic_input']} 行，分类时排除了这些正文。没有改写错配文本或擅自删除样本。去重统计和相互冲突的原始标签组见 `metadata.json` 与 `audit/duplicate_groups.csv`。用于训练时应先审查重复/冲突组，并按重复组划分训练测试集，以减少数据泄漏。domain、dataset、id 及审计字段不应直接作为真假分类器输入，新闻文本输入使用 text。

## 复现与完整性

在本文件夹运行：`python preprocessing/normalize_datasets.py --output-dir rebuilt`。只需要 Python 标准库，无额外依赖或联网。输出目录必须为空；原始文本投影 SHA256 已固定，数据行数、原标签计数、列集合、唯一 ID、二元标签、非空文本及 CSV 写回均自动验证。`manifest.json` 保存所有交付文件的大小和 SHA256。

## 来源与许可

MC-Fake：[作者仓库](https://github.com/qwerfdsaplking/MC-Fake)、[NaCTeM 官方发布页](https://www.nactem.ac.uk/mc-fake/)。原始 Google Drive 文件由作者仓库链接。NaCTeM 声明 CC BY 4.0；请归属 NaCTeM, University of Manchester，并引用 Min et al. (WWW 2022), *Divide-and-Conquer: Post-User Interaction Network for Fake News Detection on Social Media*。

Jruvika：[Kaggle / jruvika / Fake News detection](https://www.kaggle.com/datasets/jruvika/fake-news-detection)，版本 1。Kaggle 许可原文为 `Database: Open Database, Contents: © Original Authors`；新闻内容权利属于原作者。可访问性及证据限制在 provenance 中记录。
'''


if __name__ == "__main__":
    main()
