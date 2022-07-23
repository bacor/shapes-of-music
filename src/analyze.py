from typing import List, Dict
import logging
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from .config import CONDITIONS, RESULTS_DIR
from .condition import Condition
from .helpers import relpath, log_file


def collect_dist_dip_test_results(conditions: List[Dict] = CONDITIONS) -> pd.DataFrame:
    log_fn = os.path.join(RESULTS_DIR, f"dist-dip-test-results.log")
    logging.basicConfig(
        filename=log_fn,
        filemode="a",
        format="%(levelname)s %(asctime)s %(message)s",
        datefmt="%d-%m-%y %H:%M:%S",
        level=logging.INFO,
    )

    logging.info(f"Collecting results from {len(conditions)} conditions...")
    raw_results = []
    for i in tqdm(range(len(conditions))):
        settings = conditions[i]
        try:
            condition = Condition(**settings)
        except Exception as e:
            logging.warning(f"Condition could not be initialized: {condition}")
            continue

        outcome = dict(
            hash=condition.hash[:5],
            **settings,
            num_contours=len(condition),
            tableone_exists=False,
            tableone_p=None,
            tableone_dip=None,
            unidip_exists=False,
            unidip_p=None,
            unidip_dip=None,
            unidip_left=None,
            unidip_right=None,
            umap_exists=False,
            umap_path=None,
        )

        # Tableone
        if condition.is_serialized("tableone_dist_dip_test"):
            tableone = condition.tableone_dist_dip_test()
            if not type(tableone) is dict and np.isnan(tableone):
                logging.warning(f"Tableone result could not be computed: {condition}")
            else:
                outcome["tableone_exists"] = True
                for key, value in tableone.items():
                    outcome[f"tableone_{key}"] = value
        else:
            logging.warning(f"Tableone results have not been serialized: {condition}")

        # Unidip
        if condition.is_serialized("unidip_dist_dip_test"):
            unidip = condition.unidip_dist_dip_test()
            if not type(unidip) is dict and np.isnan(unidip):
                logging.warning(f"Unidip result could not be computed: {condition}")
            else:
                outcome["unidip_exists"] = True
                for key, value in unidip.items():
                    outcome[f"unidip_{key}"] = value
        else:
            logging.warning(f"Unidip results have not been serialized: {condition}")

        # UMAP
        if os.path.exists(condition.figure_path("umap_plot")):
            outcome["umap_exists"] = True
            outcome["umap_path"] = relpath(condition.figure_path("umap_plot"))

        raw_results.append(outcome)

    # Turn into dataframe
    results = pd.DataFrame(raw_results).set_index("hash")
    results.loc[results["length"].isnull(), "length"] = "all"
    uniques = results["unique"] == True
    results.loc[uniques, "unique"] = "unique"
    results.loc[~uniques, "unique"] = "all"

    # Store results
    results_fn = os.path.join(RESULTS_DIR, "dist-dip-test-results.csv")
    results.to_csv(results_fn)
    log_file(results_fn, "dist-dip test results")
    return results


if __name__ == "__main__":
    collect_dist_dip_test_results()
    ...
