import hashlib
import os
import logging

from .config import ROOT_DIR


def relpath(path):
    return os.path.relpath(path, start=ROOT_DIR)


def md5checksum(path):
    """Return an md5 checksum of a given file"""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def log_file(filename: str, name: str):
    md5 = md5checksum(filename)
    logging.info(f"Stored {name} to {relpath(filename)}")
    logging.info(f"md5 checksum: {md5}")
