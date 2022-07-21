import os
import logging
import numpy as np
import pandas as pd
from typing import List
from ..config import CONTOUR_DIR
from ..helpers import relpath, md5checksum
from .extract_contours import store_shuffled_indices, LOGGING_OPTIONS


def combine_datasets(
    name: str, datasets: List[str], sample_size: int = 1000
) -> pd.DataFrame:
    logging.info("Creating combined dataset {name} (sample_size={sample_size})")
    for dataset in datasets:
        logging.info(f"> {dataset}")

    dataframes = []
    for dataset in datasets:
        contours_fn = os.path.join(CONTOUR_DIR, f"{dataset}-contours.csv.gz")
        df = pd.read_csv(contours_fn, index_col=0)
        df["dataset"] = dataset
        columns = df.columns[:-1].tolist()
        columns.insert(1, "dataset")
        df = df.reindex(columns=columns)
        size = np.min([len(df), sample_size])
        subset = df.sample(size, replace=False)
        logging.info(f"Sampled {len(subset)} contours from dataset {dataset}")
        dataframes.append(subset)
    combined = pd.concat(dataframes)

    combined_fn = os.path.join(CONTOUR_DIR, f"{name}-contours.csv.gz")
    combined.to_csv(combined_fn, compression="gzip")
    md5 = md5checksum(combined_fn)
    logging.info(f"Stored combined contours to {relpath(combined_fn)}")
    logging.info(f"md5 checksum: {md5}")

    subset_fn = os.path.join(CONTOUR_DIR, f"{name}-contours-indices.json")
    store_shuffled_indices(combined, subset_fn)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine the contours from multiple datasets"
    )
    parser.add_argument(
        "name",
        type=str,
        help="Name of the combined dataset: combined-phrase or combined-random",
    )
    args = parser.parse_args()
    name = args.name

    log_fn = os.path.join(CONTOUR_DIR, f"{name}.log")
    logging.basicConfig(filename=log_fn, **LOGGING_OPTIONS)
    logging.info(f"Generating contour dataset: {name}")

    datasets = [
        "erk",
        "boehme",
        "creighton",
        "han",
        "shanxi",
        "natmin",
        "liber-antiphons",
        "liber-responsories",
        "liber-alleluias",
    ]
    if name == "combined-phrase":
        combine_datasets(
            "combined-phrase", [f"{d}-phrase" for d in datasets], sample_size=1000
        )

    elif name == "combined-random":
        combine_datasets(
            "combined-random", [f"{d}-random" for d in datasets], sample_size=1000
        )


if __name__ == "__main__":
    main()
