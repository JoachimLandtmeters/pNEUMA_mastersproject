# Traffic characteristic estimation from large scale trajectory dataset
The code in this repository was part of a masters thesis.
The trajectory data acquired by drones from the pNEUMA Experiment are linked to the underlying network to get macroscopic traffic characteristics at any location in the network by placing virtual loops.

Using the code is pretty straightforward, first transforming the dataset to the right format, afterwards extracting the network of the research area and mapmatching all trajectories to this network. Placing virtual loops at any location in the network makes it then possible to get macroscopic traffic charateristics.

For more information I refer to the thesis text in this repository.
In the near future some updates, edits and bug fixes will be peroformed, as well as a short wiki to make usage easier.

The map-matching performed in this code uses an existing package 
'LeuvenMapMatching' based on the HMM method of Newson & Krumm (2009).
More information and documentation about this package can be found on https://leuvenmapmatching.readthedocs.io/en/latest/ and the GitHub repository https://github.com/wannesm/LeuvenMapMatching.
The 'LeuvenMapMatching' is licensed as:
Copyright 2015-2018, KU Leuven - DTAI Research Group, Sirris - Elucidata Group
Apache License, Version 2.0.
