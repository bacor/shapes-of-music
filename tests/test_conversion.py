import unittest
from music21 import converter
from src.contour import stream_to_contour


class TestConversion(unittest.TestCase):
    def test_durations(self):
        # Different note durations with a total duration of 12 16th notes
        s = converter.parse("tinyNotation: C4. D8 E16 F16 G8")
        c = stream_to_contour(s)
        target_times = [0, 1.5, 2, 2.25, 2.5, 3]
        target_pitches = [48, 50, 52, 53, 55, 55]
        self.assertListEqual(c.times.tolist(), target_times)
        self.assertListEqual(c.pitches.tolist(), target_pitches)

    def test_pitches(self):
        s = converter.parse("tinyNotation: C D E F")
        c = stream_to_contour(s)
        self.assertListEqual(c.pitches.tolist(), [48, 50, 52, 53, 53])

    def test_rests(self):
        """Test if rests are correctly ignored."""
        # This test is also in the docstring
        s = converter.parse("tinyNotation: C4 r4 D4 r4 r4")
        # sound = [48, --, 50, --, --]
        c = stream_to_contour(s)
        self.assertListEqual(c.times.tolist(), [0, 2, 5])
        self.assertListEqual(c.pitches.tolist(), [48.0, 50.0, 50.0])

    def test_phrase_starting_with_rest(self):
        s = converter.parse("tinyNotation: r4 C4 D8 r8")
        c = stream_to_contour(s)
        self.assertListEqual(c.times.tolist(), [0, 1, 2])
        self.assertListEqual(c.pitches.tolist(), [48, 50, 50])
