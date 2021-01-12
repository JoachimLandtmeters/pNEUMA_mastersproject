========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/pNEUMA_mastersproject/badge/?style=flat
    :target: https://readthedocs.org/projects/pNEUMA_mastersproject
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.com/JoachimLandtmeters/pNEUMA_mastersproject.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.com/github/JoachimLandtmeters/pNEUMA_mastersproject

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/JoachimLandtmeters/pNEUMA_mastersproject?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/JoachimLandtmeters/pNEUMA_mastersproject

.. |requires| image:: https://requires.io/github/JoachimLandtmeters/pNEUMA_mastersproject/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/JoachimLandtmeters/pNEUMA_mastersproject/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/JoachimLandtmeters/pNEUMA_mastersproject/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/JoachimLandtmeters/pNEUMA_mastersproject

.. |version| image:: https://img.shields.io/pypi/v/pneumampjl.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/pneumampjl

.. |wheel| image:: https://img.shields.io/pypi/wheel/pneumampjl.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/pneumampjl

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/pneumampjl.svg
    :alt: Supported versions
    :target: https://pypi.org/project/pneumampjl

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/pneumampjl.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/pneumampjl

.. |commits-since| image:: https://img.shields.io/github/commits-since/JoachimLandtmeters/pNEUMA_mastersproject/v0.0.1.svg
    :alt: Commits since latest release
    :target: https://github.com/JoachimLandtmeters/pNEUMA_mastersproject/compare/v0.0.1...master



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

::

    pip install pneumampjl

You can also install the in-development version with::

    pip install https://github.com/JoachimLandtmeters/pNEUMA_mastersproject/archive/master.zip


Documentation
=============


https://pNEUMA_mastersproject.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
