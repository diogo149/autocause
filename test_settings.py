#!/bin/env python
import sys

from autocause.challenge import read_all
from autocause.core import featurize


if __name__ == "__main__":
    SAMPLE_SIZE = 100
    pairs = read_all("sample")[0]
    new_pairs = [(a[:SAMPLE_SIZE], a_type, b[:SAMPLE_SIZE], b_type)
                 for a, a_type, b, b_type in pairs]
    config = None if len(sys.argv) < 2 else sys.argv[1]
    f = featurize(new_pairs, config)
    print(f.shape)
