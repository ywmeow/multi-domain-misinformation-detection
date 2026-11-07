from pathlib import Path
import sys

from threadpoolctl import threadpool_limits

GRAPH_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(GRAPH_DIR), str(GRAPH_DIR.parent)]

from implementation import MisinformationDetector

MODEL_NAME = 'Model_pooled'
CHECKPOINT = GRAPH_DIR / 'Models' / f'{MODEL_NAME}.joblib'


def load_model():
    return MisinformationDetector(path=CHECKPOINT)


def main():
    print('Paste an article or claim, then finish with a blank line.')
    lines = []
    for line in sys.stdin:
        if not line.strip():
            break
        lines.append(line)
    if not lines:
        raise SystemExit('Provide a nonempty article or claim.')
    with threadpool_limits(limits=4):
        result = load_model().predict([''.join(lines)])[0]
    print(f'{MODEL_NAME}: {result["label"]} ({result["fake_probability"]:.2%} fake probability)')


if __name__ == '__main__':
    main()
