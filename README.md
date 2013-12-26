autocause
=========

An extensible framework for automatically generating 2D features (in the observations by features format) relevant to causality detection.

Goals
----
+ Understandable code
+ Customizability

Not Goals
-----
+ Computational efficiency: while it is nice, the focus will be on easy of use

Requirements
--------
numpy
scipy
pandas
scikit-learn
boomlet

Status
-----
+ Feature creation works fine
+ No features have been tested for performance (so they may be wrong)
+ Missing metafeatures (A is, A is not, AB are, AB are among)
+ Missing some more features (especially model complexity features)