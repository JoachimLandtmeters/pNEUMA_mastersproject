========
Overview
========

.._pNEUMA Documentation:https://pneuma-mastersproject.readthedocs.io/en/latest/

Traffic characteristic estimation from large scale trajectory dataset
=====================================================================

The code in this repository was part of a masters thesis and is currently further developed at L-Mob Research Centre at KU Leuven.

The trajectory data acquired by drones from the pNEUMA Experiment are linked to the underlying network to get macroscopic traffic characteristics at any location in the network by placing virtual loops.

Using the code is pretty straightforward, first transforming the dataset to the right format, afterwards extracting the network of the research area and map-matching all trajectories to this network. Placing virtual loops at any location in the network makes it then possible to get macroscopic traffic characteristics.

For more information I refer to the thesis text in this repository.
In the near future some updates, edits and bug fixes will be performed, as well as a short wiki to make usage easier.

The map-matching performed in this code uses an existing package 
'LeuvenMapMatching' based on the HMM method of Newson & Krumm (2009).
More information and documentation about this package can be found on https://leuvenmapmatching.readthedocs.io/en/latest/ and the GitHub repository https://github.com/wannesm/LeuvenMapMatching.

The 'LeuvenMapMatching' is licensed as:
Copyright 2015-2018, KU Leuven - DTAI Research Group, Sirris - Elucidata Group
Apache License, Version 2.0.

* Free software: Apache Software License 2.0

Installation
============

setting up the needed environment, run:

conda env create -f pneuma_environment.yml

This should load all necessary dependencies and packages used in the scripts.
If this would not work, install the listed packages manually in a newly created environment.

Next, clone the repo and use the package locally.
Do not forget to activate the newly created environment when using the package.

Documentation
=============

Some updates were needed since the open data from the pNEUMA experiment had some name changes in the csv-files. The code now also uses HDF5 to store the datasets, this provides a cleaner way to store multiple datasets before and after some processing, such as map-matched datasets or resampled datasets, all in one place with the possibility to compress easily as well as use across platforms.

The local paths to the csv data files, available from https://open-traffic.epfl.ch , are used to load all data to a HDF5 database. To keep overview a dictionary is created with all data paths with keys generated in the form of a 'group_id'. This is the same id that is later also used to communicate easily with the HDF5 instance. Such a group id is automatically generated from the csv filename, e.g. 20181029_d5_0900__0930.csv gives an id=1029_d5_0900.

In the repository there is also already a 'data' folder with a pickle file containing a network object. This object contains the needed network graph used for matching and later analysis. A new object can be created, however the hard coded lists with '_id' values may not correspond to the right edges in the network anymore. Consequently a manual check to select the right edges might be needed, e.g. using Qgis.

Automatic compressing is not done when overwriting datasets, therefore a workaround is to run the following command in the CLI (Command Line Interface):

ptrepack --complevel=1 --complib=blosc {hdf_filename_original} {hdf_filename_new}


https://pneuma-mastersproject.readthedocs.io/en/latest/


Development
===========

under development
