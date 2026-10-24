# MC-Fake Label, Domain, and Source Verification

Verification date: 2026-10-24. This directory contains the original official documents, the paper, and machine-readable `findings.json`. Tweets, users, and propagation networks were not downloaded.

## Sources and File Versions

- Author repository: https://github.com/qwerfdsaplking/MC-Fake . The original documentation is saved as `official_github_README.md`.
- University of Manchester NaCTeM dataset page: https://www.nactem.ac.uk/mc-fake/ . The original page is saved as `official_nactem_page.html`.
- Original paper: Min et al. (WWW 2022), *Divide-and-Conquer: Post-User Interaction Network for Fake News Detection on Social Media*, DOI `10.1145/3485447.3512163`. See `Min_et_al_WWW2022.pdf`; relevant excerpts are saved in `paper_excerpts.txt`.
- The official GitHub README links to Google Drive file `1WAyOVLqM1JOQdlpsCdKKuRltOGVC9bGt`. The HEAD request in this verification returned 200, filename `MC_Fake_dataset.csv`, `Content-Length=588955484`, and `Last-Modified=Wed, 19 Oct 2022 04:58:37 GMT`. The byte count matches the existing download record.
- The NaCTeM page lists `https://www.nactem.ac.uk/data/MC_Fake_dataset.csv.gz`, which returned 404 to the HEAD request. Variants without `www` and under the `/mc-fake/` path also returned 404. The large file was not downloaded again. This unavailable link cannot establish byte equality between the gzip file and the Google Drive file.
- The recorded SHA-256 of the existing original download is `7ac5f4830aa679d8610fcb4878ab793e9a96c75d772e2cf7d325e45e6dac3949`. The SHA-256 independently recomputed from the existing text projection, `MC_Fake_text.csv`, is `adfee5fcc0953cede73da8271970cb4ab9c9f638b46f8cd8ef6cd478ff33ff2e`, matching the original metadata.
- The official website and README state 28,334 records. The current file actually contains 28,335 records, with unique `news_id` values throughout. The published data are retained as found; rows are not arbitrarily removed to match the documentation count.

## Verify `labels` Before Mapping

The official field description states `labels: veracity label of the news` without explicitly defining the correspondence between numeric 0/1 and real/fake. Therefore, `0=real, 1=fake` must not be presented as a verbatim definition from that field documentation. This verification independently cross-checks domain-specific fake/real counts from Table 1 of the original paper (PDF page 5, printed page 1152) against the existing CSV.

Excerpt from the paper's Table 1:

```text
Topics     Politics  Entertainment  Health     Covid      Syria War
Labels     Fake Real Fake Real      Fake Real  Fake Real  Fake Real
News Count 225  1026 2587 8846       590  5120 843  5393 194  2230
```

| Domain | Paper: Fake | Current label=1 | Paper: Real | Current label=0 |
| --- | ---: | ---: | ---: | ---: |
| entertainment | 2,587 | 2,587 | 8,846 | 8,846 |
| health | 590 | 590 | 5,120 | 5,120 |
| covid19 | 843 | 843 | 5,393 | 5,393 |
| syria_war | 194 | 194 | 2,230 | 2,230 |
| politics | 225 | 225 | 1,026 | 2,307 |

Both class counts match exactly in four domains, and the politics Fake count also matches. This supports **0=legitimate/real, 1=fake** for the current file. The unified labels can therefore use the identity mapping `{0: 0, 1: 1}`. The current politics Real count is higher than in the paper's version; that version difference does not justify changing current labels or deleting samples.

In particular, `FakeCovid` does not mean that every record in that category is fake news. It actually contains **393 records with label=0 and 284 records with label=1**. Do not regenerate veracity labels from the words Fake or Real in `data_name`.

## Use `data_name` for Domain Information

The official README and NaCTeM page explicitly state `data_name: news category`. Page 3 of the paper lists five topics: `Politics, Entertainment, Health, Covid-19, Syria War`. The native categories and their sources correspond to these five topics. The complete observed cross-tabulation is given below. Domain names are the normalized lowercase names used in this verification.

| data_name | domain | label=0 / real | label=1 / fake |
| --- | --- | ---: | ---: |
| FA-KES | syria_war | 231 | 194 |
| FakeCovid | covid19 | 393 | 284 |
| FakeCovidClaimFiltered | covid19 | 0 | 559 |
| HealthRelease | health | 286 | 247 |
| HealthStory | health | 869 | 343 |
| RealCovid | covid19 | 5,000 | 0 |
| RealHealth | health | 3,965 | 0 |
| RealPolitics | politics | 2,193 | 0 |
| RealSyria | syria_war | 1,999 | 0 |
| gossipcop | entertainment | 8,846 | 2,587 |
| politifact | politics | 114 | 225 |
| Total | - | 23,896 | 4,439 |

All current MC-Fake samples have one of these native categories, so `domain` can be mapped from them without inferring a topic again from the text. Preserve original `data_name` values in separate audit records, but do not append them to model input text.

## News Text and Excluded Fields

The official documentation explicitly defines `title: title of the news` and `text: text content of the news`. Model `text` is formed only by combining these two columns. Currently, 202 records have an empty body but a nonempty title, so the title can be retained as `text`. No record has both an empty title and an empty body.

The existing text projection contains `news_id,title,text,labels,data_name`. Do not use `url`, `publish_date`, `source`, `n_tweets`, `n_retweets`, `n_replies`, `n_users`, `tweet_ids`, `retweet_ids`, `reply_ids`, `user_ids`, `retweet_relations`, or `reply_relations`. The separately released user-follows-user network is also not downloaded.

The original dataset is licensed under CC BY 4.0. Attribute NaCTeM, University of Manchester, and cite the WWW 2022 paper above when using it.
