import argparse
import json
import joblib
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from evaluation_utils import metrics
from implementation import MisinformationDetector
from model import DATASETS, build_graph, make_network, numpy_logits, query_adjacency, weighted_adjacency
from prep_data import BASE, SPLIT_MANIFEST, digest, load_splits, split_fingerprint
from train import torch_sparse, train_dataset

NEIGHBORS = (5, 10, 20, 30)
THRESHOLDS = (.5, .35, .4, .45, .55, .6, .65)
TRAINING_SETTINGS = ((64, .1, .003, .001), (96, .3, .002, .001), (128, .5, .003, .01))


def graph_scores(bundle, tfidf, choices):
    anchors = bundle['anchors']
    x = normalize(bundle['svd'].transform(tfidf)).astype('float32')
    scores = {k: [] for k in choices}
    for start in range(0, len(x), 256):
        batch = x[start:start+256]
        distance, indices = bundle['neighbors'].kneighbors(batch, n_neighbors=max(choices))
        for k in choices:
            adjacency = weighted_adjacency(distance[:, :k], indices[:, :k], len(anchors))
            logits = numpy_logits(batch, adjacency, anchors, bundle['anchor_hidden'], bundle['state'])
            exp = np.exp(logits-logits.max(axis=1, keepdims=True))
            scores[k].extend((exp/exp.sum(axis=1, keepdims=True))[:, 1])
    return {k: np.asarray(value) for k, value in scores.items()}


def choose(scores, labels, choices, thresholds):
    candidates = [
        (f1_score(labels, scores[k] >= threshold, average='macro', labels=[0, 1], zero_division=0), k, threshold)
        for k in choices for threshold in thresholds
    ]
    return max(candidates, key=lambda row: row[0])


def feature_graph_batches(bundle, features, k):
    for start in range(0, features.shape[0], 256):
        values = normalize(bundle['svd'].transform(features[start:start+256])).astype('float32')
        links = query_adjacency(values, bundle['neighbors'], len(bundle['anchors']), k)
        yield values, links


def retrained_graphsage(bundle, adjacency, train_labels, dev_features, dev_labels, test_features):
    started = perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    anchors = bundle['anchors']
    k = bundle['k']
    graph = torch_sparse(adjacency, device)
    x = torch.as_tensor(anchors, device=device)
    neighbor_x = torch.sparse.mm(graph, x)
    counts = np.bincount(train_labels, minlength=2)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(train_labels)/(2*counts), dtype=torch.float32, device=device))
    labels = torch.as_tensor(train_labels, dtype=torch.long, device=device)

    def batches(features):
        for values, links in feature_graph_batches(bundle, features, k):
            yield torch.as_tensor(values, device=device), torch.as_tensor(links @ anchors, device=device), torch_sparse(links, device)

    @torch.no_grad()
    def predict(network, parts):
        anchor_hidden = network.encode(x, neighbor_x)
        probabilities = []
        for values, neighbors, links in parts:
            own = network.self2(network.encode(values, neighbors))
            logits = own + network.neighbor2(torch.sparse.mm(links, anchor_hidden))
            probabilities.append(logits.softmax(dim=1)[:, 1].cpu().numpy())
        return np.concatenate(probabilities)

    dev_batches = list(batches(dev_features))
    best_score = -1.
    for width, dropout, lr, decay in TRAINING_SETTINGS:
        torch.manual_seed(42)
        network = make_network(x.shape[1], width, dropout).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=decay)
        candidate_score, stale = -1., 0
        for epoch in range(1, 301):
            network.train()
            optimizer.zero_grad()
            loss_fn(network(x, graph, neighbor_x), labels).backward()
            optimizer.step()
            if epoch != 1 and epoch % 5:
                continue
            network.eval()
            score, _, threshold = choose({k: predict(network, dev_batches)}, dev_labels, (k,), THRESHOLDS)
            if score > candidate_score:
                candidate_score, stale = score, 0
                if score > best_score:
                    best_score, best_threshold = score, threshold
                    settings = {'query_k':k,'selected_epoch':epoch,'hidden_dimensions':width,'dropout':dropout,'learning_rate':lr,'weight_decay':decay}
                    best_weights = {name:value.detach().clone() for name,value in network.state_dict().items()}
            else:
                stale += 1
                if stale == 8:
                    break
        print(f'GraphSAGE width={width} dropout={dropout} lr={lr} decay={decay} dev F1={candidate_score:.4f}', flush=True)
    selected = make_network(x.shape[1], settings['hidden_dimensions'], settings['dropout']).to(device)
    selected.load_state_dict(best_weights)
    selected.eval()
    settings['selection_seconds'] = perf_counter()-started
    with torch.no_grad():
        saved = dict(bundle, anchor_hidden=selected.encode(x, neighbor_x).cpu().numpy(),
                     state={key:value.cpu().numpy() for key,value in selected.state_dict().items()})
    return predict(selected, batches(test_features)), best_threshold, best_score, settings, saved


def baseline_scores(method, bundle, adjacency, train_labels, dev_features, dev_labels, test_features):
    started = perf_counter()
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    anchors = bundle['anchors']
    k = bundle['k']
    def prepare_links(links):
        if method == 'gin':
            links = links.copy()
            links.eliminate_zeros()
            links.data.fill(1.)
        divisor = torch.tensor(1 + np.asarray(links.sum(axis=1)), device=device)
        return links, torch_sparse(links, device), divisor

    _, graph, degree = prepare_links(adjacency)
    x = torch.as_tensor(anchors, device=device)
    neighbor_x = torch.sparse.mm(graph, x)
    if method == 'gin':
        first = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64))
        second = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
    else:
        first = torch.nn.Linear(x.shape[1], 64)
        second = torch.nn.Linear(64, 2)
    layers = torch.nn.ModuleList([first, second]).to(device)
    optimizer = torch.optim.AdamW(layers.parameters(), lr=.003, weight_decay=1e-4)
    counts = np.bincount(train_labels, minlength=2)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(train_labels)/(2*counts), dtype=torch.float32, device=device))
    labels = torch.tensor(train_labels, dtype=torch.long, device=device)

    def combine(own, neighbor, divisor):
        return own + neighbor if method == 'gin' else (own + neighbor) / divisor

    x_aggregate = combine(x, neighbor_x, degree)

    def batches(features):
        for batch, links in feature_graph_batches(bundle, features, k):
            links, sparse_links, divisor = prepare_links(links)
            aggregate = combine(torch.tensor(batch, device=device), torch.tensor(links @ anchors, device=device), divisor)
            yield aggregate, sparse_links, divisor

    @torch.no_grad()
    def predict(parts):
        anchor_hidden = first(x_aggregate).relu()
        return np.concatenate([
            second(combine(first(aggregate).relu(), torch.sparse.mm(links, anchor_hidden), divisor))
            .softmax(dim=1)[:, 1].cpu().numpy()
            for aggregate, links, divisor in parts
        ])

    dev_batches = list(batches(dev_features))
    best_score, stale = -1., 0
    for epoch in range(1, 301):
        optimizer.zero_grad()
        hidden = torch.nn.functional.dropout(first(x_aggregate).relu(), p=.3)
        logits = second(combine(hidden, torch.sparse.mm(graph, hidden), degree))
        loss_fn(logits, labels).backward()
        optimizer.step()
        if epoch != 1 and epoch % 5:
            continue
        score, _, threshold = choose({k: predict(dev_batches)}, dev_labels, (k,), THRESHOLDS)
        if score > best_score:
            best_score, best_epoch, best_threshold, stale = score, epoch, threshold, 0
            best_weights = {name: value.detach().clone() for name, value in layers.state_dict().items()}
        else:
            stale += 1
            if stale == 8:
                break
    layers.load_state_dict(best_weights)
    elapsed = perf_counter()-started
    return predict(batches(test_features)), best_threshold, best_score, {
        'query_k':k,'selected_epoch':best_epoch,'training_seconds':elapsed,
    }


def source_predictions(source, splits, config):
    started = perf_counter()
    train = splits['train'].loc[splits['train'].dataset.eq(source)]
    dev = splits['dev'].loc[splits['dev'].dataset.eq(source)]
    path = BASE/'Models'/f'{source}_graphsage.joblib'
    bundle = MisinformationDetector(path=path).bundle
    if 'selected_method' in bundle:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bundle, config = train_dataset(source, splits['train'], splits['dev'], device, save=False)
    if bundle['training_sources'] != [source] or len(bundle['anchors']) != len(train):
        raise ValueError(f'{source}: checkpoint does not match its training source')
    vectorizer = bundle['vectorizer']
    dev_features = vectorizer.transform(dev.text)
    labels = train.label_id.to_numpy()
    dev_scores = graph_scores(bundle, dev_features, NEIGHBORS)
    original_k = bundle['k']
    original_dev, _, _ = choose(dev_scores, dev.label_id, (original_k,), (.5,))
    tuned_dev, tuned_k, tuned_threshold = choose(dev_scores, dev.label_id, (original_k, 5, 20, 30), THRESHOLDS)

    selection_seconds = perf_counter()-started

    test_features = vectorizer.transform(splits['test'].text)
    test_scores = graph_scores(bundle, test_features, sorted({original_k, tuned_k}))
    checkpoint = {'selected_epoch':config['selected_epoch'],'training_seconds':config['training_seconds']}
    _, adjacency, _ = build_graph(bundle['anchors'], original_k)
    probability, threshold, score, settings, retrained = retrained_graphsage(bundle, adjacency, labels, dev_features, dev.label_id, test_features)
    retrained_result = ('graphsage_retrained', probability, threshold, score, settings)
    tuned_result = (
        'graphsage_tuned', test_scores[tuned_k], tuned_threshold, tuned_dev,
        {'query_k': tuned_k, 'selection_seconds': selection_seconds, **checkpoint},
    )
    methods = [
        ('graphsage', test_scores[original_k], .5, original_dev, {'query_k':original_k, **checkpoint}),
        tuned_result,
        retrained_result,
        ('gcn', *baseline_scores('gcn', bundle, adjacency, labels, dev_features, dev.label_id, test_features)),
        ('gin', *baseline_scores('gin', bundle, adjacency, labels, dev_features, dev.label_id, test_features)),
    ]
    winner = max((retrained_result, tuned_result), key=lambda result: result[3])
    name, expected, cutoff, dev_f1, winner_settings = winner
    selected = retrained if name == 'graphsage_retrained' else bundle.copy()
    selected.update(k=winner_settings['query_k'], threshold=cutoff, selected_method=name,
                    source_dev_macro_f1=dev_f1, tuning_settings=winner_settings)
    staging = path.with_suffix('.joblib.tmp')
    joblib.dump(selected, staging, compress=3)
    reloaded = MisinformationDetector(path=staging)
    deployed = reloaded.predict_proba(splits['test'].text)
    np.testing.assert_allclose(deployed, expected, rtol=1e-4, atol=2e-5)
    np.testing.assert_array_equal(deployed >= cutoff, expected >= cutoff)
    winner_settings.update(model_file=path.relative_to(BASE).as_posix(), model_sha256=digest(staging))
    print(f'{source}: selected {name}, dev F1={dev_f1:.4f}, threshold={cutoff}', flush=True)
    return methods, staging, path


def evaluation_rows(source, method_result, test, target_indices, fingerprints):
    method, probability, threshold, dev_score, settings = method_result
    predicted = (probability >= threshold).astype(int)
    for target, indices in target_indices.items():
        report = metrics(test.label_id.iloc[indices], predicted[indices], probability[indices])
        row = {
            'model': method,
            'train_dataset': source,
            'test_dataset': target,
            'evaluation': 'within-dataset' if source == target else 'cross-dataset transfer',
        }
        row.update({key: value for key, value in report.items() if not isinstance(value, (dict, list))})
        for label in ('real', 'fake'):
            for key, value in report['classification_report'][label].items():
                row[f'{label}_{key.replace("f1-score", "f1")}'] = value
        (true_real, false_fake), (false_real, true_fake) = report['confusion_matrix_real_fake']
        row.update({
            'true_real': true_real,
            'false_fake': false_fake,
            'false_real': false_real,
            'true_fake': true_fake,
            'threshold': threshold,
            'source_dev_macro_f1': dev_score,
            **fingerprints,
            **settings,
        })
        yield row


def gin_predictions(source, splits):
    train = splits['train'].loc[splits['train'].dataset.eq(source)]
    dev = splits['dev'].loc[splits['dev'].dataset.eq(source)]
    bundle = MisinformationDetector(dataset=source).bundle
    if bundle['training_sources'] != [source] or len(bundle['anchors']) != len(train):
        raise ValueError(f'{source}: checkpoint does not match its training source')
    _, adjacency, _ = build_graph(bundle['anchors'], 10)
    vectorizer = bundle['vectorizer']
    baseline_bundle = dict(bundle, k=10)
    return ('gin', *baseline_scores(
        'gin', baseline_bundle, adjacency, train.label_id.to_numpy(),
        vectorizer.transform(dev.text), dev.label_id, vectorizer.transform(splits['test'].text),
    ))


def main(gin_only=False):
    splits = load_splits()
    for left,right in [('train','dev'),('train','test'),('dev','test')]:
        assert not set(splits[left].key) & set(splits[right].key)
        assert not set(splits[left].group_id) & set(splits[right].group_id)
    out = BASE/'results'
    training = json.loads((out/'training.json').read_text())
    fingerprints = {
        'test_split_fingerprint': split_fingerprint('test'),
        'split_manifest_sha256': digest(SPLIT_MANIFEST),
    }
    test = splits['test'].reset_index(drop=True)
    target_indices = {target: np.flatnonzero(test.dataset.eq(target).to_numpy()) for target in DATASETS}
    records = []
    checkpoints = []
    if gin_only:
        existing = pd.read_csv(out/'results.csv')
        retained = existing.loc[existing.model.isin({'graphsage', 'graphsage_tuned', 'graphsage_retrained', 'gcn'})]
        assert len(retained) == 64
        for key, value in fingerprints.items():
            assert retained[key].eq(value).all(), f'Existing results do not match {key}'
        records.extend(retained.to_dict('records'))
    for source in DATASETS:
        print(f'Evaluating {source}', flush=True)
        if gin_only:
            methods = [gin_predictions(source, splits)]
        else:
            methods, staging, destination = source_predictions(source, splits, training[source]['config'])
            checkpoints.append((staging, destination))
        for method in methods:
            for row in evaluation_rows(source, method, test, target_indices, fingerprints):
                records.append(row)
                print(f'{row["model"]:16s} {source:8s} -> {row["test_dataset"]:8s} F1={row["macro_f1"]:.4f}', flush=True)
    frame = pd.DataFrame(records)
    assert len(frame)==80 and not frame.duplicated(['model','train_dataset','test_dataset']).any()
    temp = out/'results.csv.tmp'
    frame.to_csv(temp,index=False)
    for staging, destination in checkpoints:
        staging.replace(destination)
    temp.replace(out/'results.csv')
    summary = frame.groupby(['model','evaluation']).macro_f1.mean().round(4).to_string()
    print('\n'.join(line.rstrip() for line in summary.splitlines()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin-only', action='store_true', help='Update the GIN baseline and keep existing comparison rows and checkpoints.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        main(gin_only=args.gin_only)
