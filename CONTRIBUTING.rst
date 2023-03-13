============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

Bug reports
===========

When `reporting a bug <https://github.com/dtscalibration/python-dts-calibration/issues>`_ please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.

Documentation improvements
==========================

dtscalibration could always use more documentation, whether as part of the
official dtscalibration docs, in docstrings, or even on the web in blog posts,
articles, and such.

Feature requests and feedback
=============================

The best way to send feedback is to file an issue at https://github.com/dtscalibration/python-dts-calibration/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that code contributions are welcome :)

Development
===========

To set up `python-dts-calibration` for local development:

1. Fork `python-dts-calibration <https://github.com/dtscalibration/python-dts-calibration>`_
   (look for the "Fork" button).
2. Clone your fork locally::

    git clone git@github.com:your_name_here/python-dts-calibration.git

3. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

4. Activate your desired development environment (e.g., a python venv or conda environment), and install the package in editable mode, with the dev dependencies::

    pip install -e .[dev]

4. When you're done making changes, make sure the code follows the right style, that all tests pass, and that the docs build with the following commands::

    hatch run format
    hatch run test
    hatch run docs:build

5. Commit your changes and push your branch to GitHub::

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

6. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests (do ``hatch run test``) [1]_.
2. Update documentation when there's new API, functionality etc.
3. Add a note to ``CHANGELOG.rst`` about the changes.
4. Add yourself to ``AUTHORS.rst``.

.. [1] Sometimes there are issues with different python versions. For this you can do
       ``hatch run test_matrix:test``. Generally, on Github tests will be run using Github Actions,
       where all versions, as well as the documentation, are tested.
       This will be slower though ...

Tips
----

To run a subset of tests::

    hatch run pytest -k test_myfeature
