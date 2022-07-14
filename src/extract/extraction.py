# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# -------------------------------------------------------------------
"""Functions for working with contours"""
import os
import numpy as np
import music21
import pandas as pd
import logging

from ..contour import stream_to_contour
from .phrases import extract_phrases_from_file
from .random_segments import extract_random_segments_from_file


def tonality_analyser(filepath):
    stream = music21.converter.parse(filepath)
    krumhansl = stream.analyze("Krumhansl")
    final = stream.flat.notes[-1]
    analysis = {
        "tonic_krumhansl": krumhansl.tonic.ps,
        "tonic_mode": None,
        "final": final.pitch.ps,
        "mode": krumhansl.mode,
    }
    ext = os.path.basename(filepath).split(".")[1]
    if ext == "gabc":
        mode_tonics = {
            1: 62.0,
            2: 62.0,
            3: 64.0,
            4: 64.0,
            5: 65.0,
            6: 65.0,
            7: 67.0,
            8: 67.0,
        }
        mode = stream.editorial["metadata"].get("mode", False)
        if mode:
            analysis["mode"] = mode
            analysis["tonic_mode"] = mode_tonics[int(mode)]

    return analysis


tonality_analyser_fields = ["tonic_krumhansl", "tonic_mode", "final", "mode"]


def extract_phrase_contours(
    filepaths: list,
    num_samples: int = 50,
    contour_id_tmpl: str = "{i:0>3}",
    extractor=extract_phrases_from_file,
    extractor_kwargs: dict = {},
    analyser=tonality_analyser,
    analyser_fields: list = tonality_analyser_fields,
) -> pd.DataFrame:
    """Extract all phrase contours from an iterable of files.
    The song ids are extracted from the filenames automatically.

    Parameters
    ----------
    filepaths : list
        An iterable of file paths
    num_samples : int, optional
        The number of points at which the pitch is computed, by default 50
    contour_id_tmpl : str, optional
        A template string for the contour ids, for example: `nova{i:0>3}`.
        Defaults to `'{i:0>3}'`

    Returns
    -------
    pd.DataFrame
        A Dataframe with song ids, phrase numbers and the contours
    """
    entries = []
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        song_id = os.path.splitext(filename)[0]
        entry = {}
        try:
            phrases = extractor(filepath, **extractor_kwargs)
            if analyser:
                analysis = analyser(filepath)

            phrase_entries = []
            for i, phrase in enumerate(phrases):
                c = stream_to_contour(phrase).interpolate(num_samples=num_samples)
                entry = {
                    "song_id": song_id,
                    "unit_num": i,
                    "unit_duration": float(phrase.quarterLength),
                    "unit_length": len(phrase.flat.notes),
                }
                if analysis:
                    entry.update(analysis)
                entry.update({t: pitch for t, pitch in enumerate(c.pitches)})
                phrase_entries.append(entry)

            # Only add phrases if all phrases could be extracted
            entries.extend(phrase_entries)
            logging.info(f"Extracted {len(phrases):0>2} contours from {filename}")
        except Exception as e:
            logging.warn(f"Skipping {song_id}: {e}")

    # Package the contours in a DataFrame
    df = pd.DataFrame(entries)
    contour_ids = [contour_id_tmpl.format(i=i) for i in range(1, len(df) + 1)]
    df["contour_id"] = contour_ids
    column_order = ["contour_id", "song_id", "unit_num", "unit_length", "unit_duration"]
    if analyser:
        column_order += analyser_fields
    column_order += list(range(0, num_samples))
    df["contour_id"] = contour_ids
    df = df[column_order].set_index("contour_id")
    return df


def extract_random_contours(
    filepaths: list,
    lam: float,
    num_samples: int = 50,
    contour_id_tmpl: str = "{i:0>3}",
    random_seed: float = 0,
):
    np.random.seed(random_seed)
    return extract_phrase_contours(
        filepaths=filepaths,
        num_samples=num_samples,
        contour_id_tmpl=contour_id_tmpl,
        extractor=extract_random_segments_from_file,
        extractor_kwargs=dict(lam=lam),
    )
