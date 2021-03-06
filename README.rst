
.. image:: https://circleci.com/gh/sdpython/td3a_cpp_deep/tree/main.svg?style=svg
    :target: https://circleci.com/gh/sdpython/td3a_cpp_deep/tree/main

.. image:: https://travis-ci.com/sdpython/td3a_cpp_deep.svg?branch=main
    :target: https://app.travis-ci.com/github/sdpython/td3a_cpp_deep
    :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/9db19ijdr8xplptj?svg=true
    :target: https://ci.appveyor.com/project/sdpython/td3a-cpp
    :alt: Build Status Windows

.. image:: https://dev.azure.com/xavierdupre3/td3a_cpp_deep/_apis/build/status/sdpython.td3a_cpp_deep
    :target: https://dev.azure.com/xavierdupre3/td3a_cpp_deep/

.. image:: https://badge.fury.io/py/td3a_cpp_deep.svg
    :target: http://badge.fury.io/py/td3a_cpp_deep

.. image:: http://img.shields.io/github/issues/sdpython/td3a_cpp_deep.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/td3a_cpp_deep/issues

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

.. image:: https://img.shields.io/github/repo-size/sdpython/td3a_cpp_deep
    :target: https://github.com/sdpython/td3a_cpp_deep/
    :alt: size

td3a_cpp_deep: template to implement torch extension
====================================================

.. image:: https://raw.githubusercontent.com/sdpython/td3a_cpp_deep/main/doc/_static/logo.png
    :width: 50

`documentation <http://www.xavierdupre.fr/app/td3a_cpp_deep/helpsphinx/index.html>`_

Simple template to implement an algorithm with *cython* and *openmp*.
It implements simple examples to demonstrate the speed up
obtained by using *cython*. The module must be compiled
to be used inplace:

::

    python setup.py build_ext --inplace

Generate the setup in subfolder ``dist``:

::

    python setup.py sdist

Generate the documentation in folder ``dist/html``:

::

    python -m sphinx -T -b html doc dist/html

Run the unit tests:

::

    python -m unittest discover tests

Or:

::

    python -m pytest

To check style:

::

    python -m flake8 td3a_cpp_deep tests examples

The function *check* or the command line ``python -m td3a_cpp_deep check``
checks the module is properly installed and returns processing
time for a couple of functions or simply:

::

    import td3a_cpp_deep
    td3a_cpp_deep.check()
