import unittest
import numpy as np
from src.contour import Contour


class TestContourClass(unittest.TestCase):
    def test_basics(self):
        c = Contour([10, 20, 40], [1, 2, 4])
        self.assertIsInstance(c.points, np.ndarray)
        self.assertIsInstance(c.times, np.ndarray)
        self.assertIsInstance(c.pitches, np.ndarray)
        self.assertIsInstance(c.positions, np.ndarray)
        self.assertEqual(len(c), 3)
        self.assertListEqual(c.times.tolist(), [1, 2, 4])
        self.assertListEqual(c.pitches.tolist(), [10, 20, 40])
        self.assertEqual(c.start, 1)
        self.assertEqual(c.end, 4)
        self.assertEqual(c.duration, 3)
        self.assertEqual(c.initial, 10)
        self.assertEqual(c.final, 40)
        self.assertEqual(c.lowest, 10)
        self.assertEqual(c.highest, 40)

        c2 = Contour([2, 1, 4, 3])
        self.assertListEqual(c2.endpoints.tolist(), [2, 3])
        self.assertListEqual(c2.extremes.tolist(), [1, 4])

    def test_sorting(self):
        c = Contour([10, 40, 20], [1, 4, 2])
        self.assertListEqual(c.times.tolist(), [1, 2, 4])
        self.assertListEqual(c.pitches.tolist(), [10, 20, 40])

    def test_regular_interpolation(self):
        c1 = Contour([0, 20, 60, 100], [0, 2, 4, 10])
        c2 = c1.interpolate(num_samples=11, kind="previous")
        self.assertEqual(len(c2), 11)
        target_times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertListEqual(c2.times.tolist(), target_times)
        target_pitches = [0, 0, 20, 20, 60, 60, 60, 60, 60, 60, 100]
        self.assertListEqual(c2.pitches.tolist(), target_pitches)

    def test_interpolation_kinds(self):
        c1 = Contour([0, 20, 10, 100], [0, 2, 4, 10])
        c2 = c1.interpolate(num_samples=11, kind="linear")
        target_c2 = [0, 10, 20, 15, 10, 25, 40, 55, 70, 85, 100]
        self.assertListEqual(c2.pitches.tolist(), target_c2)

        c3 = c1.interpolate(num_samples=11, kind="nearest")
        target_c3 = [0, 0, 20, 20, 10, 10, 10, 10, 100, 100, 100]
        self.assertListEqual(c3.pitches.tolist(), target_c3)

    # def test_interpolate_stream(self):
    #     s = converter.parse('tinyNotation: C D E F')
    #     ys = interpolate_stream(s)
    #     self.assertEqual(type(ys), np.ndarray)

    # def test_pitches(self):
    #     s = converter.parse('tinyNotation: C D E F')
    #     ys = interpolate_stream(s, num_samples=4)
    #     self.assertListEqual(list(ys), [48, 50, 52, 53])

    # def test_durations(self):
    #     # Different note durations with a total duration of 12 16th notes
    #     s = converter.parse('tinyNotation: C4. D8 E16 F16 G8')
    #     target = [48, 48, 48, 48, 48, 48, 50, 50, 52, 53, 55, 55]
    #     ys = interpolate_stream(s, num_samples=12)
    #     self.assertListEqual(list(ys), target)

    # def test_rests(self):
    #     s = converter.parse('tinyNotation: C4 r4 D4 r4 r4')
    #     #sound = [48, --, 50, --, --]
    #     target = [48, 48, 50, 50, 50]
    #     ys = interpolate_stream(s, num_samples=5)
    #     self.assertListEqual(list(ys), target)

    # def test_phrase_starting_with_rest(self):
    #     s = converter.parse('tinyNotation: r4 C4 D8 r8')
    #     target = [48, 48, 50, 50]
    #     ys = interpolate_stream(s, num_samples=4)
    #     self.assertListEqual(list(ys), target)

    # def test_num_samples(self):
    #     s = converter.parse('tinyNotation: C D E F')
    #     for N in range(10, 100, 10):
    #         ys = interpolate_stream(s, num_samples=N)
    #         self.assertEqual(len(ys), N)


if __name__ == "__main__":
    unittest.main()
