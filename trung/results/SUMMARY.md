## Pooled multi-domain training (all 4 datasets), test macro-F1

| run | model | optimizer | lr | best ep | overall | amt | multifc | mc_fake | jruvika | train min | peak GB | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deberta_adamw | microsoft/deberta-v3-base | adamw | 2e-05 | 3 | 0.8337 | 0.8510 | 0.7052 | 0.8105 | 0.9750 | 18.4 | 24.4 | 29710 |
| roberta_adamw | roberta-base | adamw | 2e-05 | 3 | 0.8317 | 0.8322 | 0.6901 | 0.8155 | 0.9850 | 10.6 | 8.7 | 52517 |
| roberta_muon_lr1e-4 | roberta-base | muon | 0.0001 | 3 | 0.8346 | 0.7822 | 0.7047 | 0.8101 | 0.9825 | 21.0 | 8.3 | 26536 |
| roberta_muon_lr2e-5 | roberta-base | muon | 2e-05 | 3 | 0.8352 | 0.8741 | 0.7099 | 0.7969 | 0.9775 | 21.0 | 8.3 | 26540 |
| roberta_muon_lr5e-5 | roberta-base | muon | 5e-05 | 3 | 0.8391 | 0.8947 | 0.7084 | 0.8172 | 0.9875 | 21.0 | 8.3 | 26535 |
| solo_deberta_adamw | microsoft/deberta-v3-base | adamw | 2e-05 | 3 | 0.8349 | 0.8510 | 0.7076 | 0.8125 | 0.9775 | 12.2 | 24.4 | 44761 |
| solo_roberta_adamw | roberta-base | adamw | 2e-05 | 3 | 0.8360 | 0.8526 | 0.7022 | 0.8121 | 0.9850 | 4.4 | 8.7 | 127637 |
| solo_roberta_muon | roberta-base | muon | 5e-05 | 3 | 0.8408 | 0.8730 | 0.7114 | 0.8196 | 0.9800 | 5.7 | 8.3 | 97443 |
| tfidf_lr (reference) | TF-IDF+LR | - | - | - | 0.8144 | 0.5781 | 0.6463 | 0.7577 | 0.9450 | - | - | - |

## Cross-domain transfer (roberta-base, train on row, test on column, test macro-F1)

| train \ test | amt | multifc | mc_fake | jruvika |
|---|---|---|---|---|
| amt | 0.916 | 0.484 | 0.632 | 0.852 |
| multifc | 0.604 | 0.704 | 0.459 | 0.596 |
| mc_fake | 0.459 | 0.471 | 0.809 | 0.470 |
| jruvika | 0.531 | 0.351 | 0.639 | 0.993 |
