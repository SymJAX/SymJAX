.. _installation:

Installation
============


SymJAX has a couple of prerequisites that need to be installed first.


CPU only installation
---------------------

Installation of SymJAX and all its dependencies (including Jax). For CPU only support is done simply as follows

    .. code-block:: bash

        $ pip install --upgrade jaxlib
        $ pip install --upgrade jax
        $ pip install --upgrade symjax
 

GPU installation
----------------

For the GPU support, the Jax installation needs to be done first and based on the
local cuda settings following `Jax Installation <https://github.com/google/jax#installation>`_.
In short, the steps involve


1. Installation of GPU drivers/libraries/compilers (``cuda``, ``cudnn``, ``nvcc``).

2. Install ``jax`` following `Jax Installation <https://github.com/google/jax#installation>`_. 

3. Install SymJAX with

    .. code-block:: bash

        $ pip install --upgrade symjax



Manual (local/bleeding-edge) installation of SymJAX
---------------------------------------------------

In place of the base installation of SymJAX from the latest official release from PyPi, one can install the latest version of SymJAX from the github repository as follows


1. Clone this repository with

    .. code-block:: bash

        $ git clone https://github.com/RandallBalestriero/SymJAX

2. Install.

    .. code-block:: bash

        $ cd SymJAX
        $ pip install .

Note that whenever changes are made to the SymJAX github repository, one can pull those changes bu running

    .. code-block:: bash

        $ git pull

from within the cloned repository. However the changes won't impact the installed version unless the install was done with

    .. code-block:: bash

        $ pip install -e .



