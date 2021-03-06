.. _developer:

Development
===========


The SymJAX project was started by Randall Balestriero in early 2020.
As an open-source project, we highly welcome contributions (`current contributors <https://github.com/RandallBalestriero/SymJAX/graphs/contributors>`_) !


Philosophy
----------

SymJAX started from the need to combine the best functionalities
of Theano, Tensorflow (v1) and Lasagne. While we propose various deep learning
oriented methods, SymJAX shall remain as general as possible in its core,
methods should be grouped as much as possible into specialized submodules, and
a complete documentation should be provided, preferably along with a working example
located in the Gallery.


How to contribute
-----------------


If you are willing to help, we recommend to follow the following steps before requesting a pull request. Recall that

#. **Coding conventions**: we used the `PEP8 style guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_ and the `black <https://black.readthedocs.io/en/stable/>`_ formatting

#. **Docstrings**: we use the `numpydoc docstring guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for documenting the functions directly from the docstrings and automatically generating the documentation with `sphinx <https://www.sphinx-doc.org/en/master/>`_. Please provide codes with up-to-date docstrings.

#. **Continuous Integration**: to ensure that all the SymJAX functionalities are tested after each modifition run ``pytest`` from the main SymJAX directory. All tests should pass before considering a change to be successful. If new functionalities are added, it is highly preferable to also add a simple test in the ``tests/`` directory to ensure that results are as expected. A Github action will automatically test the code at each ``push`` (see :ref:`testcode`).



Build/Test the doc
''''''''''''''''''


To rebuild the documentation, install several packages::

  pip install -r docs/requirements.txt

to generate the documentation, you can do in the ``docs`` directory and run::

  make html

You can then see the generated documentation in
``docs/_build/html/index.html``.

If examples/code-blocks are added to the documension, it has to be tested.
To do so, add the specific module/function in the ``tests/doc.py`` and run::

    >>> python tests/doc.py

if all tests pass, then the changes are ready to be put in a PR.
Once the documentation has been changed and all tests pass, the change is ready
for review and should be put in a PR.

Every time changes are pushed to Github ``master`` branch the SymJAX
documentations (at `symjax.readthedocs.io <https://symjax.readthedocs.io/>`_) is rebuilt based on
the ``.readthedocs.yml`` and the ``docs/conf.py`` configuration files.
For each automated documentation build you can see the
`documentation build logs <https://readthedocs.org/projects/symjax/builds/>`_.


.. _testcode:


Test the code
'''''''''''''


To run all the SymJAX tests, we recommend using ``pytest`` or ``pytest-xdist``. First, install ``pytest-xdist`` and ``pytest-benchmark`` by running
``pip install pytest-xdist pytest-benchmark``.
Then, from the repository root directory run::

    pytest

If all tests pass successfully, the code is ready for a PR.

Pull requests
'''''''''''''

Once all the tests pass and the documentation is appropriate, commit your changes to a new branch, push
that branch to your fork and send us a Pull Request via GitHub's web interface
(`guide <https://guides.github.com/introduction/flow/>`_), the PR should include a brief description.
