import os

ROOT_DIR = os.path.join(os.path.dirname(__file__), os.path.pardir)

DATASETS_DIR = os.path.abspath(
    os.path.join(ROOT_DIR, os.path.pardir, "contour-typology", "datasets")
)

CONTOUR_DIR = os.path.join(ROOT_DIR, "contours")

SERIALIZED_DIR = os.path.join(ROOT_DIR, "serialized")
