import hashlib
import io
import json
from pathlib import Path
import subprocess
import unicodedata
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
SPLIT_MANIFEST = BASE/'results'/'split_manifest.json.gz'


def digest(path):
    h = hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def split_metadata(split):
    if split not in ('train','dev','test'):
        raise ValueError('Expected train, dev or test')
    manifest = pd.read_json(SPLIT_MANIFEST,orient='records',compression='gzip')
    return manifest.loc[manifest.split.eq(split)].reset_index(drop=True)


def _text_key(text):
    return hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()


def load_splits(requested=('train','dev','test')):
    requested = tuple(requested)
    if not requested or not set(requested).issubset({'train','dev','test'}):
        raise ValueError('Select train, dev and/or test')
    manifest = pd.read_json(SPLIT_MANIFEST,orient='records',compression='gzip')
    selected = manifest.loc[manifest.split.isin(requested)].copy()
    selected['label_id'] = selected.label_id.astype('int32')
    texts = {}
    for dataset in ('amt','multifc'):
        wanted = set(selected.loc[selected.dataset.eq(dataset),'id'])
        for split in requested:
            source = ROOT/'preprocessing'/'output'/dataset/f'binary_{split}.csv'
            frame = pd.read_csv(source,usecols=['id','text'],keep_default_na=False)
            texts.update({(dataset,identifier):text for identifier,text in zip(frame.id,frame.text) if identifier in wanted})
    news = next((ROOT/'preprocessing').glob('preprocessing of mcfake*'))
    archive = news/'source_text'/'MC_Fake_text.7z'


    result = subprocess.run(['tar','-xOf',archive.name,'MC_Fake_text.csv'],cwd=archive.parent,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    mc = pd.read_csv(io.BytesIO(result.stdout),keep_default_na=False,dtype=str)
    jr = pd.read_csv(news/'source_text'/'Jruvika_text.csv',keep_default_na=False,dtype=str)
    for dataset,frame,title_col,body_col in [('mc_fake',mc,'title','text'),('jruvika',jr,'Headline','Body')]:
        wanted = set(selected.loc[selected.dataset.eq(dataset),'id'])
        for index,row in frame.iterrows():
            identifier = 'MC-Fake:'+row['news_id'] if dataset=='mc_fake' else f'Jruvika-Fake-News:{index+1:06d}'
            if identifier in wanted:
                text = ' '.join(unicodedata.normalize('NFC',row[title_col]+'\n\n'+row[body_col]).split())
                texts[(dataset,identifier)] = text
    selected['text'] = [texts[(dataset,identifier)] for dataset,identifier in zip(selected.dataset,selected.id)]
    actual_keys = selected.text.map(_text_key)
    if not actual_keys.equals(selected.key):
        raise ValueError('Shared source text differs from the recorded experiment manifest')
    columns = ['id','text','label_id','dataset','domain','group_id','split','key']
    return {split:selected.loc[selected.split.eq(split),columns].reset_index(drop=True) for split in requested}


def split_fingerprint(split):
    metadata = split_metadata(split)

    return hashlib.sha256(metadata.to_csv(index=False,lineterminator='\n').encode()).hexdigest()


if __name__ == '__main__':
    splits = load_splits()
    for left,right in [('train','dev'),('train','test'),('dev','test')]:
        assert not set(splits[left].key) & set(splits[right].key)
        assert not set(splits[left].group_id) & set(splits[right].group_id)
    record = {'manifest_sha256':digest(SPLIT_MANIFEST),'split_fingerprints':{split:split_fingerprint(split) for split in splits},'rows':{split:len(frame) for split,frame in splits.items()},'text_hashes_verified':True,'split_overlap_checks_passed':True}
    print(json.dumps(record,indent=2))
