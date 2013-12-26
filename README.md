autocause
=========

An extensible framework for automatically generating 2D features (in the observations by features format) relevant to causality detection

Requirements
--------
numpy
scipy
pandas
scikit-learn
boomlet

Status
-----
+ Numerical-Numerical data seems to be working fine
+ Numerical-Categorical data results in an empty list trying to be aggregated
+ Categorical-Numerical seems to result in flattening the 2-D matrix into a 1-D one