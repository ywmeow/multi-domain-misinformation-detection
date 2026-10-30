import joblib
import numpy as np
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from prep_data import BASE
from model import DATASETS, query_adjacency, numpy_logits

DISPLAY_NAMES = {'amt':'AMT','multifc':'MultiFC','jruvika':'Jruvika','mc_fake':'MC-Fake'}


class MisinformationDetector:
    def __init__(self,path=None,dataset='multifc'):
        self.bundle = joblib.load(path or BASE/'Models'/f'{dataset}_graphsage.joblib')
        if self.bundle.get('architecture') != 'two-layer cosine-weighted mean GraphSAGE':
            raise ValueError('Expected a new GraphSAGE artifact')

    def predict_proba(self,texts,batch_size=256):
        texts = list(texts)
        if not texts or any(not isinstance(t,str) or not t.strip() for t in texts):
            raise ValueError('Provide at least one nonempty text string')
        probabilities = []
        b = self.bundle
        for start in range(0,len(texts),batch_size):
            tfidf = b['vectorizer'].transform(texts[start:start+batch_size])
            x = normalize(b['svd'].transform(tfidf)).astype('float32')
            adjacency = query_adjacency(x,b['neighbors'],len(b['anchors']),b['k'])
            logits = numpy_logits(x,adjacency,b['anchors'],b['anchor_hidden'],b['state'])
            exp = np.exp(logits-logits.max(axis=1,keepdims=True))
            probabilities.extend((exp/exp.sum(axis=1,keepdims=True))[:,1].tolist())
        return np.asarray(probabilities)

    def predict(self,texts):
        return [{'dataset_model':self.bundle['dataset'],'label_id':int(p>=.5),'label':'fake' if p>=.5 else 'real','fake_probability':float(p)} for p in self.predict_proba(texts)]


def predict_all_models(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Provide a nonempty article or claim')
    results = []
    with threadpool_limits(limits=4):
        for dataset in DATASETS:
            detector = MisinformationDetector(dataset=dataset)
            results.append(detector.predict([text])[0])
    return results


def main():
    print('Model: two-layer cosine-weighted mean GraphSAGE graph neural network')
    print('TF-IDF -> SVD (128 features) -> GraphSAGE (64, ReLU) -> GraphSAGE (2) -> softmax')
    print('Paste one article or claim below. Finish with a blank line (press Enter twice):')
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    text = '\n'.join(lines)
    if not text.strip():
        print('No text entered. Run the script again and enter an article or claim.')
        return
    print('\nRunning the same input through all four saved single-source models...')
    predictions = predict_all_models(text)
    print(f'\n{"Training datasets":<42}{"Prediction":<14}{"Fake probability":>18}')
    print('-'*74)
    for prediction in predictions:
        name = ' + '.join(DISPLAY_NAMES[source] for source in prediction['dataset_model'].split('__'))
        print(f'{name:<42}{prediction["label"]:<14}{prediction["fake_probability"]:>17.2%}')
    print('\nProbabilities are uncalibrated model scores, not factual verification.')


if __name__ == '__main__':
    main()
