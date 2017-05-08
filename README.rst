.. image:: https://travis-ci.org/ibaidev/bolib.svg?branch=master
  :target: https://travis-ci.org/ibaidev/bolib
  :alt: Build Status
.. image:: https://readthedocs.org/projects/bolib/badge/?version=latest
  :target: http://bolib.readthedocs.io/?badge=latest
  :alt: Documentation Status

BOlib
=====

A python library for Bayesian Optimization.

Setup BOlib
-----------

- The following packages must be installed before installing BOlib

.. code-block:: bash

  # for ptyhon3
  apt-get install python3-numpy python3-scipy python3-matplotlib
  # or for python2
  apt-get install python-numpy python-scipy python-matplotlib
  # gfortran is required if DIRECT optimizer is needed
  apt-get install gfortran
  # is also recommended to install libopenblas-base for multi-core systems
  apt-get install libopenblas-base

- Create and activate virtualenv (for python2) or
  venv (for ptyhon3)

.. code-block:: bash

  # for ptyhon3
  python3 -m venv --system-site-packages .env
  # or for python2
  virtualenv --system-site-packages .env

  source .env/bin/activate

- Install BOlib package

.. code-block:: bash

  python -m pip install bolib
  # if DIRECT optimizer is needed
  python -m pip install bolib[direct]


Run BOlib
---------

- Create an example configuration

.. code-block:: bash

  vi example/example_config.json
  {
    "of": "branin",
    "solver": {
      "module": "bayesian_optimization",
      "optimizer": {
        "module": "random_grid",
        "initial_sample": 10
      },
      "af": {
        "module": "ei",
        "xi": 0.001
      },
      "model": {
        "module": "gaussian_process",
        "fit": {
          "module": "hparam_optimization",
          "noiseless": 1,
          "restarts": 10,
          "opt": 1,
          "estimator": "ML"
        },
        "kernel": "squared_exponential"
      }
    }
  }

- Run example

.. code-block:: bash

  bolib run --input example/example_config.json --output output.json --verbose

- Generate an animation of the previous experiment

.. code-block:: bash

  bolib view --input output.json --output output.gif


Use BOlib as a library
----------------------

- You can also install BOlib and use its modules in your python script

.. code-block:: python

  import bolib.models.gp.gaussian_process as GP


- Hint: Try the following line to execute with the working directory in
  the current location of the bash:

.. code-block:: bash

  python -m bolib run --input example/example_config.json --output output.json --verbose


Extend BOlib with your own modules
----------------------------------

- You can also add you own modules. BOlib imports modules from the current
  working directory. Edit the example_config.json as follows

.. code-block:: bash

  ...
  "of": "YOUR_PACKAGE.YOUR_OF",
  ...


Develop BOlib
-------------

-  Download the repository using git

.. code-block:: bash

  git clone https://github.com/ibaidev/bolib.git
  cd bolib
  git config user.email 'MAIL'
  git config user.name 'NAME'
  git config credential.helper 'cache --timeout=300'
  git config push.default simple

- The following packages must be installed after installing BOlib

.. code-block:: bash

  python -m pip install bolib twine wheel

- Upload distribution

.. code-block:: bash

  python setup.py sdist bdist_wheel
  twine upload dist/*

