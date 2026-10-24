# Verification of Jruvika Fake News Native Labels

The harmonized conversion is as follows: native `Label=0` means fake and maps to harmonized `label=1`; native `Label=1` means real/legitimate and maps to harmonized `label=0`. The expression is `1 - int(Label)`. This interpretation is supported by cross-checking explicit third-party usage code, the original data preview, and the local data. No explicit definition of the numeric labels authored by Jruvika was obtained, so this conclusion must not be presented as a statement by the dataset creator.

The original source is [Kaggle jruvika/fake-news-detection](https://www.kaggle.com/datasets/jruvika/fake-news-detection). The local dataset contains 4,009 records with the columns `URLs, Headline, Body, Label`: 2,137 records have native label 0 and 1,872 have native label 1.

## Verification Evidence and Its Status

The project description of the third-party repository [tbabinet/fake-news-challenge](https://github.com/tbabinet/fake-news-challenge) directly cites this Kaggle dataset. The successfully retrieved `main.ipynb` explicitly contains:

```python
fake_data = data[labels==0]
real_data = data[labels==1]
```

The same cell outputs `Nombre de fake news : 2137` (number of fake news records) and `Nombre de news correctes : 1872` (number of real news records), for a total of 4,009. The repository tree API reported commit `7820ad5014eca63bd2324348df20844e6bf5173b` and the `main.ipynb` blob `4bc100e63de0e3179bb1710823aa9439d23770d2`. A separate read-only network request also successfully verified the [file at the pinned commit](https://raw.githubusercontent.com/tbabinet/fake-news-challenge/7820ad5014eca63bd2324348df20844e6bf5173b/main.ipynb). Its code and output at zero-based cell index 6 match the excerpts above. The complete response was 62,449 bytes, with SHA-256 `eb9b6baaeee24e286e8a02306cbd597645c8eda8043779d150dd3fa81821b7fd`. Only the excerpt and response verification information are saved here; the complete notebook was not archived. See `notebook_label_excerpt.json`.

The official Kaggle data preview retrieved through Jina Reader likewise showed 2,137 records with label 0 and 1,872 with label 1. It showed the BBC article "Four ways Bob Corker skewered Donald Trump" and a Reuters film article labeled 1, while the BeforeItsNews article "JetNation FanDuel League; Week 4" was labeled 0. The local original data contains the same examples and labels. These checks provide high confidence in the numeric orientation. However, the preview itself does not explain the 0/1 semantics, and patterns in source websites do not replace article-by-article fact-checking.

The paper [Defending Democracy: Using Deep Learning to Identify and Prevent Misinformation](https://arxiv.org/html/2106.02607) mentions and cites Jruvika, but its `1=fake, 0=real` statement concerns its own merged and transformed dataset. That statement cannot establish the meaning of Jruvika's native labels.

## Access and License Records

Direct requests made during verification to the Kaggle dataset page and its view, metadata, list, and download APIs all returned HTTP 404. These results describe those particular requests and do not establish that the dataset was deleted. Pages retrieved through Jina Reader did return successful responses. One response containing a complete data card displayed the license text `Database: Open Database, Contents: © Original Authors`, linked to [ODbL 1.0](http://opendatacommons.org/licenses/odbl/1.0/). Later responses from the same address returned only a data preview. The complete original page was not archived during this verification, so the license text is recorded as an observed page statement. Copyright in the content remains with its original authors.

This directory contains observed and rechecked excerpts, verification findings, and a local read-only quality audit. Only the label excerpt from the pinned notebook version was saved; neither the complete notebook nor the original Kaggle page was archived. No GitHub CSV mirror was downloaded, and no complete mirror comparison was performed. The limitations on original-document archival and full-data comparison are explicitly listed in `findings.json`.

## Data Quality and Harmonized Fields

The local read-only audit found 21 records with an empty `Body`; their `Headline` fields are nonempty, so their titles can be retained. After whitespace normalization, there are 467 additional records with an exactly duplicated headline-plus-body combination, with no conflicting labels for identical text. There are 1,127 additional duplicates of nonempty bodies. Some repeated bodies resemble website recommendations and clearly do not match their headlines, indicating quality issues in the original scraping. This verification preserves the meaning of the original annotations; it does not guarantee that every article has been fact-checked. These records should not be silently deleted or relabeled. Detailed counts, hashes, and repeated-template examples are in `local_quality_audit.json`.

The harmonized `text` combines only `Headline` and `Body`, separated by two newline characters. When `Body` is empty, only `Headline` is retained. `URLs` is excluded from model text. Website statistics in the evidence are used only for source verification, not for topical domain classification.

The dataset has no native topical domain column. `domain` should be classified from the news text and described as an inferred domain in the processing documentation. Website identity, URLs, or veracity labels must not substitute for topic labels. Since some bodies appear to be recommendation templates, topic classification should give appropriate extra weight to the headline and allow subsequent manual spot checks.
