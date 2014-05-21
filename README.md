autocause
=========

An extensible framework for automatically generating a large amount of 2D features (in the observations by features format) relevant to causality detection.

Goals
----
+ Understandable code
+ Customizability
+ Reproducibility

Requirements
--------
+ Python 2.7
+ numpy
+ scipy
+ pandas
+ scikit-learn
+ boomlet (http://github.com/diogo149/boomlet)
+ progress

Status
-----
More features could be added, but based on testing, features perform better than expected.

Usage
-----
Copy the settings template, make changes to it, and pass the pairs of variables and the new settings file to the autocause.core.featurize function.

Challenge Data
-----------
+ data for SUP1 and SUP2 have been split (too large for github) with the following command:
    split --bytes=90m CEdata_train_pairs.csv CEdata_train_pairs.csv
+ recombine the split files into the original with the command:
    cat CEdata_train_pairs.csv* > CEdata_train_pairs.csv

Interest
-------
If you're interested in using this or new features, send me an email at diogo149@gmail.com and we can talk about it.
