# MC-Fake / Jruvika Preprocessing and Training Handoff

The `preprocess_news.py` script in this directory lightly normalizes both datasets using the team's AMT / MultiFC conventions. The reference repository is pinned to `SunLZP/multi-domain-misinformation-detection@30ce93af1b7b11e8af654763b49652b1226fe69d`. The reference documents provide methodological guidance; processing uses the local scripts and keeps results in this D: drive directory rather than using the reference implementation's automatic fallback location.

## Generated Outputs

Results are stored in `output/mc_fake/` and `output/jruvika/`.

| Dataset | Total | Train | Dev | Test | Real / Fake |
|---|---:|---:|---:|---:|---:|
| MC-Fake | 28,335 | 22,668 | 2,835 | 2,832 | 23,896 / 4,439 |
| Jruvika | 4,009 | 3,209 | 399 | 401 | 1,872 / 2,137 |

Neither dataset has any exclusions from the binary profile in this run. All 32,344 original records are retained. Duplicates with the same label are kept within a single partition rather than deleted.

| File | Contents |
|---|---|
| `binary.csv` / `binary.jsonl` | Normalized binary data with `split=unsplit` |
| `binary_train/dev/test.csv` / `.jsonl` | Newly generated training, development, and test partitions with groups kept together |
| `unified.csv` | The original five-field contract: `id,text,label,domain,dataset` |
| `articles.csv` | Audit of every original record, including `text_raw`, original title and body, original label, domain assignment method, quality flags, `group_id`, and final `split` |
| `binary_excluded.csv` | Excluded records and reasons; header only when there are no exclusions |
| `quality_report.json` | Actual counts, class and domain distributions, text lengths, duplicates, quality issues, and partitioning policies |
| `manifest.json` | Version 3 completion marker, source/script/output SHA-256 hashes, profiles, and loading information |

The binary format uses the same eight fields as the team's AMT / MultiFC exports:

`id,text,label,label_id,original_label,domain,group_id,split`

**Both `label` and `label_id` are integers: 0=real/legitimate, 1=fake. Use only `text` as model input and `label_id` as the target.** Parse numeric CSV targets as integers; JSONL targets are already integers. The `dataset` field remains available in the five-field `unified.csv` and the full audit table. The dataset prefixes in `id` also identify the source of each record in the eight-field files.

## Preprocessing Rules

- Combine the original title and body, apply NFC Unicode normalization, and collapse all whitespace to single spaces. Preserve case, punctuation, numbers, negation, and stopwords. Stemming, TF-IDF fitting, and tokenizer-specific truncation are left to the modeling stage.
- Preserve the original decoded fields in `title_raw` and `body_raw`. Set `text_raw` to the original title, two newline characters, and the original body. Retain records with an empty body if the title is nonempty. Flag short texts, existing replacement characters, and literal escape sequences without guessing repairs.
- Map MC-Fake's original `labels` as `0 -> 0` and `1 -> 1`. Map Jruvika's original `Label` as `0 -> 1` and `1 -> 0`. Read the original text projections to avoid inverting previously harmonized labels a second time. Evidence for the earlier label verification remains in the parent directory's `provenance/` folder.
- Assign MC-Fake domains through an explicit `data_name` mapping and retain the original `data_name` for every record. Reuse Jruvika's earlier automatic domains, inferred from title and body only, and preserve their review flags. Rename the `science_technology` domain to `technology` to match AMT's topic spelling; retain the previous value in `domain_original`. MultiFC's `domain` identifies a fact-checking source, whereas these datasets use topical domains; their meanings differ.
- Exclude the original `URLs` column, social relationships, user IDs, propagation networks, and interaction statistics from model features. Preserve wording already present in the title and body.

## Grouping and Partitioning

These MC-Fake / Jruvika inputs do not include native train/dev/test assignments. The outputs therefore provide **new convenience partitions**, rather than official benchmark partitions. Compute groups jointly across both datasets, then use `seed=42` to approximate an 80/10/10 split by sample count and the `dataset x domain x label` distribution. A shared group stays in one partition even when it contains records from both datasets.

Grouping keys use NFC normalization, whitespace collapse, and casefolding separately from the model text. Connected components join records with identical full text, identical nonempty titles, or identical bodies of at least 80 characters. Model `text` still preserves case. This conservative rule controls visible text overlap; it does not establish that all records in a group describe the same semantic event. Repeated body templates can connect unrelated titles into large groups, so partitioning targets sample counts rather than group counts. The report records the largest group.

Comparison against the pinned version of the team's published binary datasets found no identical combined full texts. However, 14 MC-Fake titles either exactly match MultiFC training claims or contain the complete claim. Their groups are assigned to `train`. Details and reference-file hashes are saved in `reference/cross_reference_overlap.json` and `.csv`. These checks do not rule out every semantic near duplicate.

Retain duplicates with the same label within one partition, following the team's policy of retaining within-partition duplicates. If identical normalized full texts have conflicting binary targets, exclude all records sharing that conflicting full text from binary input while retaining them in the audit table. Matching titles or bodies with different labels do not trigger automatic relabeling.

## Loading the Data

Run from this `preprocessing` directory:

```python
import preprocess_mcfake
import preprocess_jruvika

mc = preprocess_mcfake.load_preprocessed(profile="binary_grouped")
jr = preprocess_jruvika.load_preprocessed(profile="binary_grouped")

train = mc["train"] + jr["train"]
dev = mc["dev"] + jr["dev"]
test = mc["test"] + jr["test"]
train_text = [row["text"] for row in train]
train_target = [row["label_id"] for row in train]
```

Omitting `profile` returns `{"unsplit": ...}`, matching AMT's default binary profile. The loader first verifies every output SHA-256 listed in the manifest, then checks the binary JSONL fields, integer targets, and partition values. Use `id`, `domain`, `group_id`, and `original_label` for auditing, grouping, and evaluation; keep them out of the input text.

## Reproducing the Outputs

Python 3.10 or later is required, with no third-party dependencies. Run from this directory:

```powershell
python preprocess_news.py --output-dir .\output_rerun
```

The default command processes both datasets. `--dataset mc_fake` or `--dataset jruvika` exports only the selected dataset but still reads both to preserve consistent grouping. The selected dataset's subdirectory under the output location must not already exist; the script refuses to overwrite it. The original `source_text` files and earlier five-field exports remain unchanged. A source checksum mismatch, malformed input, or cross-partition overlap stops processing. Each manifest is published only after the output files pass readback checks.

Choose model features during training. Fit the TF-IDF vocabulary and IDF on `train` only, and use the selected transformer's own tokenizer. The report's whitespace `word_count` is a diagnostic measure, not a model token count. Describe Jruvika's inferred domains, mismatched bodies, repeated templates, and class imbalance when reporting experiments.

See [Preprocessing Notes](NEWS_PREPROCESSING_NOTES_EN.md) for the correspondence with the team's AMT / MultiFC policies.
