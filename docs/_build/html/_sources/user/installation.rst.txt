Installation
============


SymJAX has a couple of prerequisites that need to be installed first without
hard version requirements. However, as SymJAX is tightly coupled with Jax, it will
require a recent version of Jax.


CPU only installation
---------------------

Installation of SymJAX and all its dependencies (including Jax) for CPU only
support is done simply as follows

    .. code-block:: bash

        $ pip install symjax
 

GPU installation
----------------

For the GPU support, the Jax installation needs to be done first and based on the
local cuda settings following `Jax Installation <https://github.com/google/jax#installation>`_.
In short, the steps involve


1. Installation of GPU drivers/libraries/compilers (``cuda``, ``cudnn``, ``nvcc``).

2. Install ``jax`` following `Jax Installation <https://github.com/google/jax#installation>`_. 

3. Install SymJAX with

    .. code-block:: bash

        $ pip install symjax



Manual (bleeding-edge) installation of SymJAX
---------------------------------------------

In place of the base installation of SymJAX from the latest official release from PyPi, one can install the latest version of SymJAX from the github repository as follows


1. Clone this repository with

    .. code-block:: bash

        $ git clone https://github.com/RandallBalestriero/SymJAX

2. Install.

    .. code-block:: bash

        $ cd SymJAX
        $ pip install -r requirements.txt
        $ pip install .


