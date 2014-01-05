#!/bin/env python
from autocause.challenge import read_all
from autocause.core import featurize


if __name__ == "__main__":
    SAMPLE_SIZE = 100
    pairs = read_all("sample")[0]
    new_pairs = [(a[:SAMPLE_SIZE], a_type, b[:SAMPLE_SIZE], b_type)
                 for a, a_type, b, b_type in pairs]
    f = featurize(new_pairs)
    print(f.shape)
