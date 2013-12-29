autocause
=========

An extensible framework for automatically generating a large amount of 2D features (in the observations by features format) relevant to causality detection.

Goals
----
+ Understandable code
+ Customizability

Not Goals
-----
+ Computational efficiency: while it is nice, the focus will be on easy of use

Requirements
--------
+ numpy
+ scipy
+ pandas
+ scikit-learn
+ boomlet

Status
-----
+ Feature creation works fine
+ No features have been tested for performance (so they may be wrong)
+ Missing metafeatures (A is, AB are, {AB, BA} are among)
+ Missing some more features (especially model complexity features)
+ Missing the "state of the art" features

Challenge Data
-----------
+ data for SUP1 and SUP2 have been split (too large for github) with the following command:
    split --bytes=90m CEdata_train_pairs.csv CEdata_train_pairs.csv
+ recombine the split files into the original with the command:
    cat CEdata_train_pairs.csv* > CEdata_train_pairs.csv
