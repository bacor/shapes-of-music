import logging
from .condition import Condition, TooFewContoursException
from .config import CONDITIONS_PER_DATASET, ALL_DATASETS
from tqdm import tqdm


def task_all(conditions):
    for condition in conditions:
        condition.similarities()
        condition.unidip_dist_dip_test()
        condition.tableone_dist_dip_test()
        condition.kde_similarities()
        condition.umap_plot()


def task_essential(conditions):
    for condition in conditions:
        condition.similarities()
        condition.tableone_dist_dip_test()
        condition.kde_similarities()
        condition.umap_plot()


def task_precompute(conditions):
    for condition in conditions:
        condition.similarities()


def task_visualize(conditions):
    for condition in conditions:
        condition.umap_plot()


def task_dist_dip(conditions):
    for condition in conditions:
        condition.unidip_dist_dip_test()
        condition.tableone_dist_dip_test()


def task_kde(conditions):
    for condition in conditions:
        condition.kde_similarities()


TASKS = {
    "all": task_all,
    "essential": task_essential,
    "precompute": task_precompute,
    "visualize": task_visualize,
    "dist_dip": task_dist_dip,
    "kde": task_kde,
}

def iter_conditions(conditions):
    for i in tqdm(range(len(conditions))):
        settings = conditions[i]
        try:
            condition = Condition(**settings, log=True)
            yield condition
        except TooFewContoursException:
            logging.warn(f'Skipping condition {i}: too few contours found: {settings}')

def run_task(task, dataset):
    # Validate inputs
    if task not in TASKS.keys():
        raise ValueError(
            f'Unknown task "{task}". Choose one of: {", ".join(TASKS.keys())}'
        )
    if dataset not in ALL_DATASETS:
        raise ValueError(f'Unknown dataset "{dataset}"')

    # Collect all conditions
    conditions = CONDITIONS_PER_DATASET[dataset]
    print(
        f'> Performing task "{task}" for dataset {dataset} ({len(conditions)} conditions)'
    )

    # Run!
    task_fn = TASKS[task]
    task_fn(iter_conditions(conditions))


def main():
    import argparse
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 


    parser = argparse.ArgumentParser(description="CLI for the experiments")
    parser.add_argument("task", type=str, help="The task to be executed")
    parser.add_argument(
        "dataset",
        type=str,
        help="ID of the dataset for which to execute the task",
    )
    args = parser.parse_args()

    if args.dataset == "combined":
        run_task(args.task, "combined-phrase")
        run_task(args.task, "combined-random")

    elif args.dataset == "synthetic":
        run_task(args.task, "markov")
        run_task(args.task, "binom")

    elif args.dataset == "western":
        run_task(args.task, "erk-phrase")
        run_task(args.task, "erk-random")
        run_task(args.task, "boehme-phrase")
        run_task(args.task, "boehme-random")
        run_task(args.task, "creighton-phrase")
        run_task(args.task, "creighton-random")

    elif args.dataset == "chinese":
        run_task(args.task, "han-phrase")
        run_task(args.task, "han-random")
        run_task(args.task, "natmin-phrase")
        run_task(args.task, "natmin-random")
        run_task(args.task, "shanxi-phrase")
        run_task(args.task, "shanxi-random")

    elif args.dataset == "liber":
        for dataset in ALL_DATASETS:
            if dataset.startswith("liber"):
                run_task(args.task, dataset)

    elif args.dataset == "cantus":
        for dataset in ALL_DATASETS:
            if dataset.startswith("cantus"):
                run_task(args.task, dataset)

    else:
        run_task(args.task, args.dataset)


if __name__ == "__main__":
    main()
