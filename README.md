kemlglearn
==========

Different ML and preprocesing algorithms in python specially for clustering 

Trying to follow the interfaces of scikit-learn

Just for teaching purposes, most of the algorithms are not implemented efficiently

Some of the algorithms are from other authors:

KernelKMeans by Mathieu Blondel <mathieu@mblondel.org> https://gist.github.com/mblondel/6230787
KModes and KPrototypes by Nico de Vos <njdevos@gmail.com> https://github.com/nicodv/kmodes


## Dev Environment 
Tested on conda 4.5.4 in Windows 10
```
conda env create -n kemlg-devenv python=2.7 numpy scipy scikit-learn matplotlib
conda activate kemlg-devenv
pip install -e .
```