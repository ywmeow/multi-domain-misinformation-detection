import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from evaluation_utils import metrics
from implementation import MisinformationDetector
from model import DATASETS
from prep_data import BASE, digest, load_splits, split_fingerprint, SPLIT_MANIFEST


def main():
    splits = load_splits()
    for left,right in [('train','dev'),('train','test'),('dev','test')]:
        assert not set(splits[left].key) & set(splits[right].key)
        assert not set(splits[left].group_id) & set(splits[right].group_id)
    out = BASE/'results'
    out.mkdir(exist_ok=True)
    records = []
    training = json.loads((out/'training.json').read_text())
    test_fingerprint = split_fingerprint('test')
    manifest_hash = digest(SPLIT_MANIFEST)
    test = splits['test'].reset_index(drop=True)
    for source in DATASETS:
        path = BASE/'Models'/f'{source}_graphsage.joblib'
        detector = MisinformationDetector(path=path)
        if detector.bundle.get('training_sources',[detector.bundle['dataset']]) != [source]:
            raise ValueError(f'{source}: expected a single-source model')
        probability = detector.predict_proba(test.text)
        predicted = (probability >= .5).astype(int)
        model_hash = digest(path)
        config = training[source]['config']
        for target in DATASETS:
            idx = np.flatnonzero(test.dataset.eq(target).to_numpy())
            report = metrics(test.label_id.iloc[idx],predicted[idx],probability[idx])
            row = {'train_dataset':source,'test_dataset':target,'evaluation':'within-dataset' if source==target else 'cross-dataset transfer',**{k:v for k,v in report.items() if not isinstance(v,(dict,list))}}
            for label in ('real','fake'):
                row.update({f'{label}_{key.replace("f1-score","f1")}':value for key,value in report['classification_report'][label].items()})
            tn,fp,fn,tp = np.asarray(report['confusion_matrix_real_fake']).ravel()
            row.update({'true_real':int(tn),'false_fake':int(fp),'false_real':int(fn),'true_fake':int(tp),'selected_epoch':config['selected_epoch'],'source_dev_macro_f1':config['selected_dev_macro_f1'],'training_seconds':config['training_seconds'],'model_file':str(path.relative_to(BASE)),'model_sha256':model_hash,'test_split_fingerprint':test_fingerprint,'split_manifest_sha256':manifest_hash})
            records.append(row)
            print(f'train={source:8s} test={target:8s} n={len(idx):4d} accuracy={report["accuracy"]:.4f} precision={report["macro_precision"]:.4f} recall={report["macro_recall"]:.4f} F1={report["macro_f1"]:.4f} (macro)',flush=True)
    frame = pd.DataFrame(records)
    assert len(frame)==16 and not frame.duplicated(['train_dataset','test_dataset']).any()
    frame.to_csv(out/'results.csv',index=False)
    matrix = frame.pivot(index='train_dataset',columns='test_dataset',values='macro_f1').reindex(index=DATASETS,columns=DATASETS)
    matrix.index.name = 'train / test'
    print('\nMacro-F1: train row -> test column\n'+matrix.round(4).to_string(),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
