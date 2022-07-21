import os
from typing import List, Optional, Union, Dict
import numpy as np
from itertools import product

############################################################
# Directory structure
############################################################

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

DATASETS_DIR = os.path.abspath(
    os.path.join(ROOT_DIR, os.path.pardir, "contour-typology", "datasets")
)

CONTOUR_DIR = os.path.join(ROOT_DIR, "contours")

SERIALIZED_DIR = os.path.join(ROOT_DIR, "serialized")

FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

############################################################
# Experimental setup
############################################################

VARIABLES: List[str] = [
    "dataset",
    "representation",
    "metric",
    "length",
    "unique",
    "limit",
    "dimensionality",
]

ALL_REPRESENTATIONS: List[str] = [
    "pitch",
    "pitch_centered",
    "pitch_normalized",
    "pitch_tonicized",
    "pitch_finalized",
    "cosine",
    "interval",
    "smooth_derivative",
]

ALL_METRICS: List[str] = ["eucl", "dtw"]

ALL_LENGTHS: List[Optional[int]] = [
    None,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
]

METRIC_LIMITS: Dict[str, int] = {"eucl": 3000, "dtw": 500}

INVALID_COMBINATIONS: List[Dict[str, Union[str, int]]] = [
    dict(representation="cosine", metric="dtw"),
    dict(dataset="markov", representation="pitch_tonicized"),
    dict(dataset="binom", representation="pitch_tonicized"),
]

MIN_CONTOUR_COUNT: int = 10
"""The minimum number of contours required for a condition to be valid"""


def get_conditions(
    representations: List[str],
    metrics: List[str],
    lengths: List[Optional[int]],
    uniques: List[bool],
    dims: List[int],
) -> List[Dict[str, Union[str, int]]]:
    """Take the product of all passed parameters and returns an iterable
    of all dictionaries, each representing an experimental condition"""
    tuples = product(representations, metrics, lengths, uniques, dims)
    conditions = []
    for repres, metric, length, unique, dim in tuples:
        conditions.append(
            dict(
                representation=repres,
                metric=metric,
                length=length,
                unique=unique,
                limit=METRIC_LIMITS[metric],
                dimensionality=dim,
            )
        )
    return conditions


def validate_condition(condition: Dict) -> Dict:
    for invalid in INVALID_COMBINATIONS:
        test = [condition[k] == v for k, v in invalid.items()]
        if np.all(test):
            return False
        else:
            return condition


############################################################
## Particular setups used for various datasets
############################################################


all_conditions = get_conditions(
    representations=ALL_REPRESENTATIONS,
    metrics=ALL_METRICS,
    lengths=ALL_LENGTHS,
    uniques=[True, False],
    dims=[50],
) + get_conditions(
    ALL_REPRESENTATIONS, ALL_METRICS, lengths=[None], uniques=[False], dims=[10]
)

most_conditions = get_conditions(
    representations=ALL_REPRESENTATIONS,
    metrics=ALL_METRICS,
    lengths=[None, 5, 6, 7, 8, 10],
    uniques=[False],
    dims=[50],
)

basic_conditions = get_conditions(
    representations=[
        "pitch_centered",
        "pitch_tonicized",
        "cosine",
        "interval",
    ],
    metrics=["eucl"],
    lengths=[None],
    uniques=[False],
    dims=[50],
)

cantus_conditions = get_conditions(
    representations=["pitch_centered", "pitch_tonicized", "cosine", "interval"],
    metrics=["eucl"],
    lengths=[None, 3, 4, 5, 6, 7, 8],
    uniques=[False],
    dims=[50],
)

############################################################
# The actual conditions
############################################################

CONDITIONS_PER_DATASET = {
    # Cross-cultural subset
    "combined-phrase": all_conditions,
    "combined-random": all_conditions,
    # Synthetic baselines
    "markov": all_conditions,
    "binom": basic_conditions,
    # Western folksongs
    "erk-phrase": most_conditions,
    "erk-random": most_conditions,
    "boehme-phrase": basic_conditions,
    "boehme-random": basic_conditions,
    "creighton-phrase": basic_conditions,
    "creighton-random": basic_conditions,
    # Chinese subset of Essen
    "han-phrase": most_conditions,
    "han-random": most_conditions,
    "natmin-phrase": basic_conditions,
    "natmin-random": basic_conditions,
    "shanxi-phrase": basic_conditions,
    "shanxi-random": basic_conditions,
    # Gregobase
    "liber-antiphons-phrase": most_conditions,
    "liber-antiphons-random": most_conditions,
    "liber-responsories-phrase": basic_conditions,
    "liber-responsories-random": basic_conditions,
    "liber-alleluias-phrase": basic_conditions,
    "liber-alleluias-random": basic_conditions,
    # Cantus
    "cantus-responsory-neumes": cantus_conditions,
    "cantus-responsory-syllables": cantus_conditions,
    "cantus-responsory-words": cantus_conditions,
    "cantus-antiphon-neumes": cantus_conditions,
    "cantus-antiphon-syllables": cantus_conditions,
    "cantus-antiphon-words": cantus_conditions,
}

# Validate those conditions
for dataset, conditions in CONDITIONS_PER_DATASET.items():
    valid_conditions = []
    for condition in conditions:
        condition["dataset"] = dataset
        condition = validate_condition(condition)
        if condition is not False:
            valid_conditions.append(condition)
    CONDITIONS_PER_DATASET[dataset] = valid_conditions

# Combine all conditions into one list of dictionaries
DATASETS = list(CONDITIONS_PER_DATASET.keys())
CONDITIONS = [c for conditions in CONDITIONS_PER_DATASET.values() for c in conditions]
