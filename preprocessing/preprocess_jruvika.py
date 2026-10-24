"""Jruvika adapter with the same loader interface as the teammate's AMT script."""
import sys
import preprocess_news


def load_preprocessed(profile="binary", output_dir=None):
    return preprocess_news.load_preprocessed("jruvika", profile, output_dir)


if __name__ == "__main__":
    sys.argv.extend(["--dataset", "jruvika"])
    preprocess_news.main()
