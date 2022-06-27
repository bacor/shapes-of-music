# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# License: 
# -------------------------------------------------------------------
"""Tests for the code used to analyse the kern files
"""
import unittest
import os
import numpy as np
from music21 import humdrum
from src.phrases import extract_phrases_from_spine
from src.phrases import extract_phrases_from_kern_file
from src.phrases import extract_phrases_from_file

class TestKernPhraseExtraction(unittest.TestCase):

    def test_phrase_extraction(self):
        data = """**kern
                {4c
                =1
                4d
                8f
                =2
                8g}
                4a
                {8a}
                8a
                =3
                {8g
                8f
                4d
                4c}
                {8c
                8d}
                ==""".replace('                ','')
        song = humdrum.parseData(data)
        spine = song.spineCollection.spines[0]
        phrases = extract_phrases_from_spine(spine)
        self.assertEqual(len(phrases), 4)
        phrase1 = [n.pitch.midi for n in phrases[0].notes]
        self.assertListEqual(phrase1, [60, 62, 65, 67])
        phrase2 = [n.pitch.midi for n in phrases[1].notes]
        self.assertListEqual(phrase2, [69])
        phrase3 = [n.pitch.midi for n in phrases[2].notes]
        self.assertListEqual(phrase3, [67, 65, 62, 60])
        phrase4 = [n.pitch.midi for n in phrases[3].notes]
        self.assertListEqual(phrase4, [60, 62])

    def test_single_note_phrase(self):
        data = "**kern\n{4c}"
        song = humdrum.parseData(data)
        spine = song.spineCollection.spines[0]
        phrases = extract_phrases_from_spine(spine)
        self.assertEqual(len(phrases), 1)
        self.assertEqual(phrases[0].notes[0].pitch.midi, 60)

    def test_extract_from_file(self):
        cur_dir = os.path.dirname(__file__)
        path = os.path.join(cur_dir, 'test.krn')
        phrases = extract_phrases_from_kern_file(path)
        self.assertEqual(len(phrases), 4)
        phrase1 = [n.pitch.midi for n in phrases[0].notes]
        self.assertListEqual(phrase1, [60, 62, 65, 67])
        phrase2 = [n.pitch.midi for n in phrases[1].notes]
        self.assertListEqual(phrase2, [69])
        phrase3 = [n.pitch.midi for n in phrases[2].notes]
        self.assertListEqual(phrase3, [67, 65, 62, 60])
        phrase4 = [n.pitch.midi for n in phrases[3].notes]
        self.assertListEqual(phrase4, [60, 62])

if __name__ == '__main__':
    unittest.main()    