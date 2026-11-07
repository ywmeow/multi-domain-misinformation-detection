import json
from itertools import product
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from predict import CHECKPOINT, load_model, write_checkpoint
from graph.test import graph_scores
from model import DATASETS
from prep_data import load_splits, split_fingerprint


def main():
    start = perf_counter()
    config = json.loads(Path(__file__).with_name('training.json').read_text())['config']
    fingerprint = split_fingerprint('dev')
    if config['data_hashes']['dev'] != fingerprint:
        raise ValueError('Development split differs from the recorded training split')
    dev = load_splits(('dev',))['dev']
    bundle = load_model().bundle
    if bundle['training_sources'] != list(DATASETS) or set(dev.dataset) != set(DATASETS):
        raise ValueError('Tuning requires all four datasets')

    baseline = (config['k'], .5)
    neighbors = sorted({3, 5, 8, 10, 12, 15, 20, 25, 30, baseline[0]})
    thresholds = [value / 100 for value in range(35, 66)]
    with threadpool_limits(limits=4):
        features = bundle['vectorizer'].transform(dev.text)
        probabilities = graph_scores(bundle, features, neighbors)
    labels = dev.label_id.to_numpy()
    groups = list(dev.groupby('dataset').indices.values())

    scores = {}
    for k, threshold in product(neighbors, thresholds):
        predicted = probabilities[k] >= threshold
        scores[k, threshold] = float(np.mean([
            f1_score(labels[indices], predicted[indices], labels=[0, 1], average='macro', zero_division=0)
            for indices in groups
        ]))
    k, threshold = max(scores, key=lambda setting: (scores[setting], setting == baseline))
    bundle.update(k=k, threshold=threshold, tuning={
        'dev_split_fingerprint': fingerprint,
        'baseline_mean_dev_macro_f1': scores[baseline],
        'selected_mean_dev_macro_f1': scores[k, threshold],
        'query_neighbors_searched': neighbors,
        'thresholds_searched': thresholds,
        'selection_seconds': perf_counter() - start,
    })
    staging = CHECKPOINT.with_suffix('.tuning.tmp')
    write_checkpoint(bundle, staging)
    for attempt in range(5):
        try:
            staging.replace(CHECKPOINT)
            break
        except PermissionError as error:
            if getattr(error, 'winerror', None) != 32 or attempt == 4:
                raise
            sleep(1)
    print(f'Model_pooled: k={k}, threshold={threshold:.2f}, '
          f'mean dev macro-F1 {scores[baseline]:.4f} -> {scores[k, threshold]:.4f}')


if __name__ == '__main__':
    main()
