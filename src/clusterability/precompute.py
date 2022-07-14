import argparse
from .dataset import Dataset


def main():
    """CLI for the generation
    Usage:  `python generate_data.py dataset_id`
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompute all similarity scores and so on"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="ID of the dataset to generate,\
        such as `creighton` or `boehme`, or `liber-antiphons`, `liber-kyries`, etc.\
        So either it corresponds to a directory in datasets/, or it is of the form \
        liber-genre.",
    )
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    dataset.precompute_all()


if __name__ == "__main__":
    main()
