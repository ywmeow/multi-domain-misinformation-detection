# MC-Fake and Jruvika Fake News: Harmonized Text Data

The common field order is `id,text,label,domain,dataset`. CSV files use UTF-8 with a BOM. Article text may contain newlines, so read these files with a CSV parser. The harmonized labels are **0=legitimate/real, 1=fake**.

| Dataset | Records | 0: Real | 1: Fake |
| --- | ---: | ---: | ---: |
| MC-Fake | 28,335 | 23,896 | 4,439 |
| Jruvika-Fake-News | 4,009 | 1,872 | 2,137 |

`MC_Fake_unified.csv` and `Jruvika_Fake_News_unified.csv` contain the separately processed datasets. `combined_unified.csv` combines all **32,344** records. The `text` field joins the title, two newline characters, and the body. When either part is empty, only the nonempty part is retained. Only leading and trailing whitespace around each part is stripped; internal content is preserved. The `dataset` value is `MC-Fake` or `Jruvika-Fake-News`. MC-Fake IDs retain the original `news_id` with a dataset prefix. Jruvika has no native ID, so stable IDs are generated from the original data-row order.

## Label Verification and Mapping

MC-Fake's original `labels` use 0=real and 1=fake, so the numeric mapping is unchanged. The official field documentation describes a veracity label without explicitly defining the numeric direction. Cross-tabulated real/fake counts in the current CSV match Table 1 of the original paper exactly for entertainment, health, COVID, and Syria, supporting this direction. Do not infer veracity from source names such as `FakeCovid`: that source contains both labels.

Jruvika's original `Label` uses 0=fake and 1=real, so harmonization applies `label = 1 - Label`. The evidence is public analysis code explicitly referencing the original Kaggle dataset: it assigns `fake_data=data[labels==0]` and `real_data=data[labels==1]`, producing counts of 2,137 and 1,872, respectively. These match the downloaded source, and article examples provide an additional cross-check. The Kaggle publisher's metadata does not explicitly define the numeric meanings; third-party code must not be represented as a publisher statement. Pinned links, saved excerpts, and verification details are in `provenance/`.

## Domains

Both datasets use the same broad topics: `health` for health and COVID; `politics` for politics; `entertainment` for entertainment; `conflict` for war and armed conflict; `sports` for sports; `business` for business and economics; `science_technology` for science and technology; `environment` for the environment and natural disasters; `society` for society and justice; and `other` when the topic cannot be determined.

MC-Fake uses an explicit mapping from its native `data_name`. The original `data_name`, `news_id`, and label are preserved for each row in `audit/MC_Fake_domain_and_labels.csv`, allowing further analysis of subtopics such as COVID versus other health news, or Syria. The main table's `domain` values do not contain the Real/Fake wording of source names.

| Original data_name | Harmonized domain |
| --- | --- |
| FakeCovidClaimFiltered | health |
| FakeCovid | health |
| RealCovid | health |
| HealthRelease | health |
| HealthStory | health |
| RealHealth | health |
| politifact | politics |
| RealPolitics | politics |
| FA-KES | conflict |
| RealSyria | conflict |
| gossipcop | entertainment |

Jruvika has no native domain labels. This version uses **deterministic text-topic rules** that read only `Headline` and `Body`. Titles take priority, and bodies are truncated for classification. Repeated body templates occurring at least 10 times are excluded from the classification input, while the merged `text` still retains the original body. These are automatically inferred domains, not a human-annotated gold standard. Scores are rule-evidence points, not confidence probabilities. The rules are in `preprocessing/topic_classifier.py`. A total of 1,467 records are marked `needs_review`, and 224 have title/body topic disagreements. Records requiring review, matched terms, the top two topics, and score margins are in `audit/Jruvika_topic_review.csv`. All classification records are in `audit/Jruvika_domain_and_labels.csv`.

## Retained Content and Data Quality

The main tables contain only the five fields listed above. Tweets, retweets, replies, user IDs, propagation edges, and interaction statistics are not retained. Jruvika's original `URLs` column was not copied and was not used for classification. Text projections of the original fields are stored in `source_text/` and also exclude those fields. Text already present within titles and bodies is preserved.

All original records are retained. MC-Fake has 202 records with empty bodies, and Jruvika has 21. All have titles, so no harmonized `text` is empty. The current MC-Fake source contains 28,335 records, whereas the historical official description reports 28,334. This version retains the records actually present in the file.

Jruvika contains repeated bodies and mismatched titles and bodies. Seven body templates affect 668 rows; those bodies were excluded from topic-classification input. Mismatched text was not rewritten, and records were not silently deleted. Duplicate statistics and groups with conflicting original labels are recorded in `metadata.json` and `audit/duplicate_groups.csv`. Before training, review duplicate/conflict groups and keep duplicate groups together when creating training and test partitions to reduce leakage. Use `text` as news-text input; `domain`, `dataset`, `id`, and audit fields should not be used directly as veracity-classifier inputs.

## Reproduction and Integrity

From this folder, run `python preprocessing/normalize_datasets.py --output-dir rebuilt`. Only the Python standard library is required; no extra dependencies or network access are needed. The output directory must be empty. Source text-projection SHA-256 hashes are fixed. Record counts, original label counts, column sets, unique IDs, binary labels, nonempty text, and CSV readback are checked automatically. `manifest.json` records file sizes and SHA-256 hashes.

## Sources and Licenses

MC-Fake: [authors' repository](https://github.com/qwerfdsaplking/MC-Fake) and [official NaCTeM dataset page](https://www.nactem.ac.uk/mc-fake/). The original Google Drive file is linked from the authors' repository. NaCTeM specifies CC BY 4.0. Attribute NaCTeM, University of Manchester, and cite Min et al. (WWW 2022), *Divide-and-Conquer: Post-User Interaction Network for Fake News Detection on Social Media*.

Jruvika: [Kaggle / jruvika / Fake News detection](https://www.kaggle.com/datasets/jruvika/fake-news-detection), version 1. The Kaggle license wording is `Database: Open Database, Contents: © Original Authors`; article-content rights remain with the original authors. Access limitations and evidence qualifications are documented in `provenance/`.

## English Provenance Documents

- [Download-record metadata](provenance/download_records/README_EN.md)
- [MC-Fake label, domain, and source verification](provenance/mc_fake/LABEL_AND_DOMAIN_VERIFICATION_EN.md)
- [MC-Fake official repository README](provenance/mc_fake/official_github_README_EN.md)
- [Jruvika label verification](provenance/jruvika/LABEL_VERIFICATION_EN.md)
