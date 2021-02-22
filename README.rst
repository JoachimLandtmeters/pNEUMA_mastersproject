========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
.. |docs| image:: https://readthedocs.org/projects/pNEUMA_mastersproject/badge/?style=flat
    :target: https://readthedocs.org/projects/pNEUMA_mastersproject
    :alt: Documentation Status
.. end-badges

# Traffic characteristic estimation from large scale trajectory dataset
The code in this repository was part of a masters thesis.
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

No installation provided, just clone the repo and use the package locally.
Following packages will be needed:
- osmnx
- leuvenmapmatching
- h5py
- geopy
- numpy
- tqdm
- pyproj

Documentation
=============

Some updates were needed since the open data from the pNEUMA experiment had some name changes in the csv-files. The code now also uses hdf5 to store the datasets, this provides a cleaner way to store multiple datasets before and after some processing, such as ma-matched datasets or resampled datasets, all in one place with the possibility to compress easily as well as use across platforms.

Automatic compressing is not done when overwriting datasets, therefore a workaround is to run the following command in the CLI (Command Line Interface):

ptrepack --complevel=1 --complib=blosc {hdf_filename_original} {hdf_filename_new}

(This was all done on Mac OS, so no assurance that everything will work immediately on other OS)

https://pNEUMA_mastersproject.readthedocs.io/


Development
===========

under development
