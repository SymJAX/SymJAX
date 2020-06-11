Development
===========

.. toctree::
  :hidden:

The SymJAX project was started by Randall Balestriero in early 2020.
As an open-source project, we highly welcome contributions (`current contributors: <https://github.com/Lasagne/Lasagne/graphs/contributors>`_) ! If you are willing to help, we recommand to follow the following steps before requesting a pull request


Philosophy
----------

SymJAX started from the need to combine the best functionalities
of Theano, Tensorflow (v1) and Lasagne. While we propose various deep learning
oriented methods, SymJAX shall remain as general as possible in its core,
methods should be grouped as much as possible into specialized submodules, and
a complete documentation should be provided, preferably along a working example
located in the Gallery.


How to contribute
-----------------


If you are willing to help, we recommand to follow the following steps before requesting a pull request

#. **Coding conventions**: we used the `PEP8 style guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_

#. **Docstrings**: we used the `numpydoc docstring guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for documenting the functions directly from the docstrings and automatically generating the documentation with `sphinx <https://www.sphinx-doc.org/en/master/>`_. Please provide codes with up-to-date docstrings.

#. **Documentation build**: make sure than the documentation successfully builds by running ``make html`` inside the ``SymJAX/docs`` repository.

#. **Testing your code**: please ensure that your code passes the benchmark tests by running ``pytest`` in the repository root.

#. **Pull requests**: once all the tests pass and the documentation is appropriate, commit your changes to a new branch, push
that branch to your fork and send us a Pull Request via GitHub's web interface (`guide <
https://guides.github.com/introduction/flow/>`_), the PR should include a brief description.
