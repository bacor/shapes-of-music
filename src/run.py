from .dataset import Dataset
from .condition import Condition
from .config import CONDITIONS_PER_DATASET, DATASETS


def task_all(conditions):
    for condition in conditions:
        condition = Condition(**condition, log=True)
        condition.similarities()
        condition.dist_dip_test()
        condition.kde_similarities()
        condition.umap_plot()


def task_precompute(conditions):
    for condition in conditions:
        condition = Condition(**condition, log=True)
        condition.similarities()


def task_visualize(conditions):
    for condition in conditions:
        condition = Condition(**condition, log=True)
        condition.umap_plot()


def task_dist_dip(conditions):
    for condition in conditions:
        condition = Condition(**condition, log=True)
        condition.dist_dip_test()


def task_kde(conditions):
    for condition in conditions:
        condition = Condition(**condition, log=True)
        condition.kde_similarities()


TASKS = {
    "all": task_all,
    "precompute": task_precompute,
    "visualize": task_visualize,
    "dist_dip": task_dist_dip,
    "kde": task_kde,
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CLI for the experiments")
    parser.add_argument("task", type=str, help="The task to be executed")
    parser.add_argument(
        "dataset",
        type=str,
        help="ID of the dataset for which to execute the task",
    )
    args = parser.parse_args()

    # Validate inputs
    if args.task not in TASKS.keys():
        raise ValueError(
            f'Unknown task "{args.task}". Choose one of: {", ".join(TASKS.keys())}'
        )
    if args.dataset not in DATASETS:
        raise ValueError(f'Unknown dataset "{args.dataset}"')

    # Collect all conditions
    conditions = CONDITIONS_PER_DATASET[args.dataset]
    task_fn = TASKS[args.task]
    task_fn(conditions)


if __name__ == "__main__":
    main()
