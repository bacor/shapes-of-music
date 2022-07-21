import numpy as np
import pandas as pd
import scipy.interpolate

_VOLPIANO_TO_MIDI = {
    "8": 53,  # F
    "9": 55,  # G
    "a": 57,
    "y": 58,  # B flat
    "b": 59,
    "c": 60,
    "d": 62,
    "w": 63,  # E flat
    "e": 64,
    "f": 65,
    "g": 67,
    "h": 69,
    "i": 70,  # B flat
    "j": 71,
    "k": 72,  # C
    "l": 74,
    "x": 75,  # E flat
    "m": 76,
    "n": 77,
    "o": 79,
    "p": 81,
    "z": 82,  # B flat
    "q": 83,  # B
    "r": 84,  # C
    "s": 86,
    # Liquescents
    "(": 53,
    ")": 55,
    "A": 57,
    "B": 59,
    "C": 60,
    "D": 62,
    "E": 64,
    "F": 65,
    "G": 67,
    "H": 69,
    "J": 71,
    "K": 72,  # C
    "L": 74,
    "M": 76,
    "N": 77,
    "O": 79,
    "P": 81,
    "Q": 83,
    "R": 84,  # C
    "S": 86,  # D
    # Naturals
    "Y": 59,  # Natural at B
    "W": 64,  # Natural at E
    "I": 71,  # Natural at B
    "X": 76,  # Natural at E
    "Z": 83,
}

_MODE_FINALS = {1: 62.0, 2: 62.0, 3: 64.0, 4: 64.0, 5: 65.0, 6: 65.0, 7: 67.0, 8: 67.0}


def volpiano_to_midi(volpiano, fill_na=False, skip_accidentals=False):
    """
    Translates volpiano pitches to a list of midi pitches

    All non-note characters are ignored or filled with `None`, if `fill_na=True`
    Unless `skip_accidentals=True`, accidentals are converted to midi pitches
    as well. So an i (flat at the B) becomes 70, a B flat. Or a W (a natural at
    the E) becomes 64 (E).
    """
    accidentals = "iwxyz" + "IWXYZ"
    midi = []
    for char in volpiano:
        if skip_accidentals and char in accidentals:
            pass
        elif char in _VOLPIANO_TO_MIDI:
            midi.append(_VOLPIANO_TO_MIDI[char])
        elif fill_na:
            midi.append(None)
    return midi


def volpiano_to_contour(volpiano, num_samples: int = 50):
    pitches = volpiano_to_midi(volpiano + volpiano[-1])
    xs = np.linspace(0, 1, len(pitches))
    func = scipy.interpolate.interp1d(xs, pitches, kind="previous")
    return func(np.linspace(0, 1, num_samples))


def flatten_array(iterable):
    return np.array([value for group in iterable for value in group])


def repeat_column_per_unit(column, unit_counts):
    values = [[value] * count for value, count in zip(column, unit_counts)]
    return flatten_array(values)


def extract_volpiano_contours(
    chants,
    df,
    segmentation,
    max_num_contours=20000,
    num_samples=50,
    contour_id_tmpl: str = "{i:0>4}",
    random_state=12345,
):
    segments = df[segmentation].str.split()
    units = flatten_array(segments)
    unit_counts = segments.map(len)

    chant_ids = repeat_column_per_unit(chants.index, unit_counts)
    unit_nums = flatten_array([np.arange(unit_count) for unit_count in unit_counts])
    unit_lengths = np.asarray([len(unit) for unit in units])
    unit_durations = unit_lengths

    midi_finals = [volpiano_to_midi(f)[0] for f in df[segmentation].str[-1]]
    finals = repeat_column_per_unit(midi_finals, unit_counts)
    modes = repeat_column_per_unit(chants["mode"], unit_counts)
    tonic_krumhansl = np.array([None for _ in modes])
    tonic_modes = np.array([_MODE_FINALS[mode] for mode in modes])

    np.random.seed(random_state)
    indices = np.sort(
        np.random.choice(np.arange(len(units)), size=max_num_contours, replace=False)
    )
    contours = np.array(
        [volpiano_to_contour(unit, num_samples=num_samples) for unit in units[indices]]
    )
    contour_ids = [contour_id_tmpl.format(i=i) for i in range(1, len(contours) + 1)]
    metadata = np.array(
        [
            contour_ids,
            chant_ids[indices],
            unit_nums[indices],
            unit_lengths[indices],
            unit_durations[indices],
            tonic_krumhansl[indices],
            tonic_modes[indices],
            finals[indices],
            modes[indices],
        ]
    ).T

    contours_df = pd.DataFrame(
        np.concatenate((metadata, contours), axis=1),
        columns=[
            "contour_id",
            "chant_id",
            "unit_num",
            "unit_length",
            "unit_duration",
            "tonic_krumhansl",
            "tonic_mode",
            "final",
            "mode",
        ]
        + [str(i) for i in range(num_samples)],
    ).set_index("contour_id")

    return contours_df
