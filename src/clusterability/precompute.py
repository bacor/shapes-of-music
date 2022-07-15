import os
import argparse
import logging
from .dataset import Dataset
from ..config import SERIALIZED_DIR


def main():
    """CLI for the generation
    Usage:  `python generate_data.py dataset_id`
    """
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

    log_fn = os.path.join(SERIALIZED_DIR, f"{args.dataset}.log")
    logging.basicConfig(filename=log_fn, 
        filemode="a",
        format="%(levelname)s %(asctime)s %(message)s",
        datefmt="%d-%m-%y %H:%M:%S",
        level=logging.INFO,
    )
    logging.info(f"Precomputing similarities for dataset: {args.dataset}")

    dataset = Dataset(args.dataset)
    dataset.precompute_all()

if __name__ == "__main__":
    main()
