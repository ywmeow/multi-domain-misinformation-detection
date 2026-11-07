import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from predict import CHECKPOINT, GRAPH_DIR, MODEL_NAME, load_model
from graph.test import evaluation_rows
from model import DATASETS
from prep_data import SPLIT_MANIFEST, digest, load_splits, split_fingerprint, split_metadata


def main():
    config = json.loads(Path(__file__).with_name('training.json').read_text())['config']
    test = load_splits(('test',))['test']
    for split in ('train', 'dev'):
        if config['data_hashes'][split] != split_fingerprint(split):
            raise ValueError(f'{split} split differs from the recorded training split')
        metadata = split_metadata(split)
        if set(metadata.key) & set(test.key) or set(metadata.group_id) & set(test.group_id):
            raise ValueError(f'Test data overlaps the {split} split')
    detector = load_model()
    if detector.bundle['training_sources'] != list(DATASETS):
        raise ValueError('Expected a checkpoint trained on all four datasets')
    with threadpool_limits(limits=4):
        probability = detector.predict_proba(test.text)
    targets = {source: np.flatnonzero(test.dataset.eq(source)) for source in DATASETS}
    targets['pooled'] = np.arange(len(test))
    fingerprints = {
        'test_split_fingerprint': split_fingerprint('test'),
        'split_manifest_sha256': digest(SPLIT_MANIFEST),
    }
    settings = {key: config[key] for key in ('selected_epoch', 'training_seconds',
                'hidden_dimensions', 'dropout', 'learning_rate', 'weight_decay')}
    settings.update(query_k=detector.bundle['k'], model_file=CHECKPOINT.relative_to(GRAPH_DIR).as_posix(),
                    model_sha256=digest(CHECKPOINT))
    result = (MODEL_NAME, probability, detector.bundle.get('threshold', .5),
              config['selected_dev_macro_f1'], settings)
    rows = list(evaluation_rows('pooled', result, test, targets, fingerprints))
    for row in rows:
        row['evaluation'] = 'pooled overall' if row['test_dataset'] == 'pooled' else 'pooled per dataset'
        print(f'{row["test_dataset"]}: macro-F1 {row["macro_f1"]:.4f}')

    path = GRAPH_DIR / 'results' / 'results.csv'
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        original = [row for row in reader if row['model'] != MODEL_NAME]
    staging = path.with_suffix('.pooled.tmp')
    with staging.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(original)
        writer.writerows(rows)
    staging.replace(path)


if __name__ == '__main__':
    main()
