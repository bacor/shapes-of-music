# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# License:
# -------------------------------------------------------------------
"""Code for generating the phrase contour datasets: CSV files stored in
the data/ directory.

Usage:  `python -m src.extract.generate_data dataset_id`
See `python generate_data.py -h` for details.
"""
import os
import glob
import json
import logging
import pandas as pd
import numpy as np

from ..config import CONTOUR_DIR, DATASETS_DIR
from ..helpers import relpath, md5checksum
from .extraction import extract_phrase_contours
from .extraction import extract_random_contours
from .volpiano import extract_volpiano_contours

LOGGING_OPTIONS = dict(
    filemode="w",
    format="%(levelname)s %(asctime)s %(message)s",
    datefmt="%d-%m-%y %H:%M:%S",
    level=logging.INFO,
)


def store_shuffled_indices(df: pd.DataFrame, filename: str) -> None:
    """Store a JSON file with shuffled indices of the contours, and also a list
    of indices of shuffled, unique contours. We store a fixed randomized order
    so that we can later easily the same random subsets.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe of contours
    filename : str
        the json file where the indices are stored
    """
    # Extract the contours from the dataframe
    for i in range(df.shape[1]):
        if df.columns[i] == 0 or df.columns[i] == "0":
            first_column = i
            break
    contours = df.iloc[:, first_column:]
    output = {}

    # All contours, shuffled
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    output["all"] = indices.tolist()

    # Unique contours, shuffled
    unique_index = contours.drop_duplicates().index
    logging.info(
        f">  Removed {len(df) - len(unique_index)} duplicates; {len(unique_index)} contours left."
    )
    subset = df.loc[
        unique_index,
    ].sample(frac=1, replace=False)
    output["unique"] = df.index.get_indexer_for(subset.index).tolist()

    # Store
    with open(filename, "w") as handle:
        json.dump(output, handle, indent=2)
    md5 = md5checksum(filename)
    logging.info(f"Stored shuffled outputs to {relpath(filename)}")
    logging.info(f"md5 checksum: {md5}")


def generate_contour_data(
    dataset_id: str,
    filepaths: list,
    num_samples: int = 50,
    dataset_dir: str = DATASETS_DIR,
):

    # Extract phrase contours
    phrase_contours = extract_phrase_contours(
        filepaths, contour_id_tmpl=dataset_id + "-{i:0>5}", num_samples=num_samples
    )

    # Store csv file and log a checksum
    phrases_contours_fn = os.path.join(
        CONTOUR_DIR, f"{dataset_id}-phrase-contours.csv.gz"
    )
    phrase_contours.to_csv(phrases_contours_fn, compression="gzip")
    md5 = md5checksum(phrases_contours_fn)
    logging.info(f"Stored phrase contours to {relpath(phrases_contours_fn)}")
    logging.info(f"md5 checksum: {md5}")

    # Store subsets
    subset_fn = os.path.join(CONTOUR_DIR, f"{dataset_id}-phrase-contours-indices.json")
    store_shuffled_indices(phrase_contours, subset_fn)

    # Extract random phrases
    mean_phrase_length = phrase_contours["unit_length"].mean()
    logging.info(
        f"Extracting random contours with mean length lamb={mean_phrase_length:.2f}..."
    )
    random_contours = extract_random_contours(
        filepaths,
        lam=mean_phrase_length,
        contour_id_tmpl=dataset_id + "-rand-{i:0>5}",
        num_samples=num_samples,
    )
    mean_random_length = random_contours["unit_length"].mean()
    logging.info(f"Mean length of random phrases: {mean_random_length:.2f}...")

    # Store random phrases
    random_contours_fn = os.path.join(
        CONTOUR_DIR, f"{dataset_id}-random-contours.csv.gz"
    )
    random_contours.to_csv(random_contours_fn, compression="gzip")
    md5 = md5checksum(random_contours_fn)
    logging.info(f"Stored random contours to {relpath(random_contours_fn)}")
    logging.info(f"md5 checksum: {md5}")

    # Store subsets
    random_subset_fn = os.path.join(
        CONTOUR_DIR, f"{dataset_id}-random-contours-indices.json"
    )
    store_shuffled_indices(random_contours, random_subset_fn)


def generate_kern_contour_data(
    dataset_id: str,
    file_pattern: str,
    num_samples: int = 50,
    dataset_dir: str = DATASETS_DIR,
):
    """Generate a phrase contour dataset

    The dataset is stored as `data/dataset_id-contours.csv`.
    The first three columns contain metadata:

    - `contour_id`: a unique id for the contour
    - `song_id`: the if of the song from which the phrase was extracted
    - `phrase_num`: the number of the phrase, starting from 0.

    A log of the generation process is stored as `dataset_id.log`.
    After generating the CSV file, and md5 checksum of the file is
    computed and logged, allowing you to validate the dataset.

    Parameters
    ----------
    dataset_id : str
        The id of the dataset, which should also be the name of
        the directory containing the dataset
    file_pattern : str
        A file pattern passed to glob to search for all files in
        the dataset
    num_samples : int, optional
        The number of points at which the pitch is computed, by default 50
    dataset_dir : str, optional
        The directory in which to find the datasets, defaults to `datasets/`
    """
    log_fn = os.path.join(CONTOUR_DIR, f"{dataset_id}.log")
    logging.basicConfig(filename=log_fn, **LOGGING_OPTIONS)
    logging.info(f"Generating contour dataset: {dataset_id}")

    # Extract all contours
    logging.info("Extracting phrase contours...")
    pattern = os.path.join(dataset_dir, dataset_id, file_pattern)
    filepaths = sorted(glob.glob(pattern, recursive=True))
    generate_contour_data(
        dataset_id=dataset_id,
        filepaths=filepaths,
        dataset_dir=dataset_dir,
        num_samples=num_samples,
    )


def generate_gregobase_contour_data(
    genre, num_samples: int = 50, dataset_dir: str = DATASETS_DIR
):
    """Generate a phrase contour dataset from the GregoBase Corpus.
    We extract all chants of a certain genre in the Liber Usualis.
    Otherwise, the function is identical to `generate_kerndataset`;
    see that function for further details.

    Parameters
    ----------
    genre : [type]
        The liturgical genre, can be one of: `antiphons`, `hymns`, `alleluias`,
        `introits`, `communions`, `responsories`, `offertories`, `graduals`,
        `kyries`, and `tracts`
    num_samples : int, optional
        The number of points at which the pitch is computed, by default 50
    dataset_dir : str, optional
        The directory in which to find the datasets, defaults to `datasets/`
    """
    genres = {
        "antiphons": "an",
        "hymns": "hy",
        "alleluias": "al",
        "introits": "in",
        "communions": "co",
        "responsories": "re",
        "offertories": "of",
        "graduals": "gr",
        "kyries": "ky",
        "tracts": "tr",
    }
    genre_key = genres[genre]
    dataset_id = f"liber-{genre}"
    log_fn = os.path.join(CONTOUR_DIR, f"{dataset_id}.log")
    logging.basicConfig(filename=log_fn, **LOGGING_OPTIONS)
    logging.info(f"Generating contour dataset: {dataset_id}")
    logging.info(f"Contours of {genre} from the Liber Usualis in GregoBaseCorpus v0.3")

    # Load GregoBase Corpus
    csv_dir = os.path.join(DATASETS_DIR, "gregobase", "csv")
    chants = pd.read_csv(os.path.join(csv_dir, "chants.csv"), index_col=0)
    chants = chants[chants["mode"].isnull() == False]
    sources = pd.read_csv(os.path.join(csv_dir, "sources.csv"), index_col=0)
    chant_sources = pd.read_csv(os.path.join(csv_dir, "chant_sources.csv"))

    # Select the right subset
    right_genre = chants.query(f"office_part == '{genre_key}'").index
    logging.info(f"Number of {genre} in GregoBase: {len(right_genre)}")
    liber_usualis = chant_sources.query("source==3").chant_id
    logging.info(f"Number of chants in the liber usualis: {len(liber_usualis)}")
    subset = right_genre.intersection(liber_usualis)
    logging.info(f"Number of {genre} in the Liber Usualis: {len(subset)}")

    # Extract all contours
    pattern = os.path.join(DATASETS_DIR, "gregobase", "gabc", "{idx:0>5}.gabc")
    filepaths = sorted([pattern.format(idx=idx) for idx in subset])

    generate_contour_data(
        dataset_id=dataset_id,
        filepaths=filepaths,
        dataset_dir=dataset_dir,
        num_samples=num_samples,
    )


def generate_cantus_contour_data(
    genre,
    num_samples: int = 50,
    dataset_dir: str = DATASETS_DIR,
    max_num_contours=10000,
):
    """"""
    log_fn = os.path.join(CONTOUR_DIR, f"cantus-{genre}.log")
    logging.basicConfig(filename=log_fn, **LOGGING_OPTIONS)
    logging.info(f"Generating contour dataset: {genre}")

    # Extract all contours
    chants = pd.read_csv(
        os.path.join(dataset_dir, "cantus", genre, "subset", "train-chants.csv"),
        index_col=0,
    )
    df = pd.read_csv(
        os.path.join(
            dataset_dir, "cantus", genre, "subset", "train-representation-pitch.csv"
        ),
        index_col=0,
    )
    for segmentation in [
        "neumes",
        "syllables",
        "words",
        "poisson-3",
        "poisson-5",
        "poisson-7",
    ]:
        logging.info(f"Extracting {segmentation} contours...")
        contours = extract_volpiano_contours(
            chants,
            df,
            segmentation,
            max_num_contours=max_num_contours,
            num_samples=num_samples,
            contour_id_tmpl="cantus-{i:0>5}",
        )
        contours_fn = os.path.join(
            CONTOUR_DIR, f"cantus-{genre}-{segmentation}-contours.csv.gz"
        )
        contours.to_csv(contours_fn, compression="gzip")
        md5 = md5checksum(contours_fn)
        logging.info(f"Stored phrase contours to {relpath(contours_fn)}")
        logging.info(f"md5 checksum: {md5}")

        subset_fn = os.path.join(
            CONTOUR_DIR, f"cantus-{genre}-{segmentation}-contours-subsets.json"
        )
        store_shuffled_indices(contours, subset_fn)


def main():
    """CLI for the generation
    Usage:  `python generate_data.py dataset_id`
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract the contour data from a dataset"
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
    if args.dataset.startswith("liber"):
        genre = args.dataset.split("-")[1]
        generate_gregobase_contour_data(genre, num_samples=50)
    elif args.dataset.startswith("cantus"):
        genre = args.dataset.split("-")[1]
        generate_cantus_contour_data(genre, num_samples=50)
    else:
        file_patterns = dict()  # creighton='kern/*krn')
        file_pattern = file_patterns.get(args.dataset, "**/*.krn")
        generate_kern_contour_data(
            args.dataset, file_pattern=file_pattern, num_samples=50
        )


if __name__ == "__main__":
    np.random.seed(123456)
    main()
    # generate_gregobase_contour_data('antiphons', num_samples=50)
