import argparse
import json
import platform
import time
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from prep_data import BASE, split_fingerprint, load_splits
from model import DATASETS, build_graph, query_adjacency, numpy_logits, make_network


def torch_sparse(a, device):
    a = a.tocoo()
    return torch.sparse_coo_tensor(torch.tensor(np.vstack([a.row,a.col]), dtype=torch.long), torch.tensor(a.data, dtype=torch.float32), a.shape).coalesce().to(device)


def train_dataset(dataset, train, dev, device):
    sources = (dataset,)
    start = time.time()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    models_dir = BASE/'Models'
    models_dir.mkdir(exist_ok=True)
    results = BASE/'results'
    results.mkdir(exist_ok=True)
    train = train.loc[train.dataset.isin(sources)].reset_index(drop=True)
    dev = dev.loc[dev.dataset.isin(sources)].reset_index(drop=True)
    print(f'{dataset}: fitting source-only TF-IDF/SVD ({len(train)} training rows)',flush=True)
    assert not set(train.key) & set(dev.key)
    assert not set(train.group_id) & set(dev.group_id)
    vectorizer = TfidfVectorizer(max_features=40000, ngram_range=(1,2), min_df=2, max_df=.98, sublinear_tf=True, dtype=np.float32)
    tfidf = vectorizer.fit_transform(train.text)
    dimensions = min(128, tfidf.shape[0]-1, tfidf.shape[1]-1)
    svd = TruncatedSVD(n_components=dimensions, n_iter=5, random_state=42)
    x = normalize(svd.fit_transform(tfidf)).astype('float32')
    d = normalize(svd.transform(vectorizer.transform(dev.text))).astype('float32')
    print(f'{dataset}: train={len(train)}, dev={len(dev)}; building training-only graph', flush=True)
    neighbors, adjacency, k = build_graph(x)
    da = query_adjacency(d, neighbors, len(x), k)
    tx = torch.tensor(x, device=device)
    tax = torch.tensor(adjacency @ x, device=device)
    td = torch.tensor(d, device=device)
    tdax = torch.tensor(da @ x, device=device)
    ta, tda = torch_sparse(adjacency,device), torch_sparse(da,device)
    targets = torch.tensor(train.label_id.to_numpy(), dtype=torch.long, device=device)
    counts = np.bincount(train.label_id, minlength=2)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(train)/(2*counts),dtype=torch.float32,device=device))
    network = make_network(dimensions,64,.3).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.003, weight_decay=1e-4)
    best, best_state, best_epoch, stale, history = -1, None, 0, 0, []
    for epoch in range(1,301):
        network.train()
        optimizer.zero_grad()
        logits = network(tx,ta,tax)
        loss = criterion(logits,targets)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0:
            network.eval()
            with torch.no_grad():
                train_hidden = network.encode(tx,tax)
                dev_hidden = network.encode(td,tdax)
                dev_logits = network.self2(dev_hidden) + network.neighbor2(torch.sparse.mm(tda,train_hidden))
                prediction = dev_logits.argmax(1).cpu().numpy()
            score = f1_score(dev.label_id,prediction,average='macro',labels=[0,1],zero_division=0)
            history.append({'epoch':epoch,'training_loss':float(loss.item()),'dev_macro_f1':float(score)})
            if score > best + 1e-8:
                best, best_epoch, stale = score, epoch, 0
                best_state = {key:value.detach().cpu().clone() for key,value in network.state_dict().items()}
            else:
                stale += 1
            if stale >= 8:
                break
    network.load_state_dict(best_state)
    network.eval()
    state = {key:value.numpy() for key,value in best_state.items()}
    with torch.no_grad():
        anchor_hidden = network.encode(tx,tax).cpu().numpy()

        dev_hidden = network.encode(td,tdax)
        expected = (network.self2(dev_hidden)+network.neighbor2(torch.sparse.mm(tda,torch.tensor(anchor_hidden,device=device)))).cpu().numpy()
    deployed = numpy_logits(d,da,x,anchor_hidden,state)
    if not np.allclose(expected,deployed,atol=2e-5,rtol=2e-5):
        raise AssertionError('Deployment logits differ from PyTorch')
    bundle = {'architecture':'two-layer cosine-weighted mean GraphSAGE','dataset':dataset,'training_sources':list(sources),'vectorizer':vectorizer,'svd':svd,'neighbors':neighbors,'anchors':x,'anchor_hidden':anchor_hidden,'state':state,'k':k,'label_mapping':{0:'real',1:'fake'}}
    joblib.dump(bundle,models_dir/f'{dataset}_graphsage.joblib',compress=3)
    config = {'dataset':dataset,'architecture':bundle['architecture'],'independently_trained':True,'seed':42,'train_rows':len(train),'dev_rows':len(dev),'dimensions':dimensions,'hidden_dimensions':64,'dropout':.3,'learning_rate':.003,'weight_decay':1e-4,'optimizer':'AdamW','loss':'training-class-balanced cross entropy','max_epochs':300,'early_stopping':'8 non-improving dev checks, evaluated at epoch 1 then every 5 epochs','selected_epoch':best_epoch,'selected_dev_macro_f1':best,'last_epoch':epoch,'k':k,'graph_edges':int(adjacency.nnz),'graph_policy':'directed cosine-weighted kNN; no self edges; row normalized; separate learned self transform','selection':'source train/dev partitions only; test partition not requested','training_seconds':time.time()-start,'device':str(device),'python':platform.python_version(),'versions':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','torch','joblib','threadpoolctl']},'deployment_logits_checked':True}
    config.update({'training_sources':list(sources),'dev_sources':list(sources),'source_counts':train.dataset.value_counts().to_dict(),'tfidf_max_features':40000,'dev_selection':'source-development macro-F1 only; other datasets dev labels are not used'})
    config['data_hashes'] = {split:split_fingerprint(split) for split in ['train','dev']}
    config['data_hash_method'] = 'text-free split metadata with verified text keys'
    config['split_manifest'] = 'results/split_manifest.json.gz'
    training_path = results/'training.json'
    training = json.loads(training_path.read_text()) if training_path.exists() else {}
    training[dataset] = {'config':config,'history':history}
    training_path.write_text(json.dumps(training,indent=2),encoding='utf-8')
    print(f'{dataset}: saved GNN, best epoch={best_epoch}, dev macro-F1={best:.4f}, seconds={config["training_seconds"]:.1f}',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args = parser.parse_args()
    device = torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    torch.set_num_threads(4)
    print(f'Training GraphSAGE on {device}',flush=True)
    splits = load_splits(('train','dev'))
    train, dev = splits['train'], splits['dev']
    with threadpool_limits(limits=4):
        for dataset in DATASETS:
            train_dataset(dataset,train,dev,device)

if __name__ == '__main__':
    main()
