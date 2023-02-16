Project comparisons
===================

Here is a list of NumPy APIs and corresponding cuNumeric implementations.

A dot in the cunumeric column denotes that cuNumeric implementation
is not provided yet. We welcome contributions for these functions.

NumPy vs cuNumeric APIs
-----------------------

.. comparison-table::


Measuring API coverage
----------------------

When running applications that use cunumeric, various command line options may
be used to generate coverage reports.

Overall coverage report
~~~~~~~~~~~~~~~~~~~~~~~

The command line flag ``-cunumeric:report:coverage`` may be added to print an
overall percentage of cunumeric coverage:

.. code-block:: sh

    legate test.py -cunumeric:report:coverage

After execution completes, the percentage of NumPy API calls that were handled
by cunumeric is printed:

.. code-block::

    cuNumeric API coverage: 26/26 (100.0%)

Detailed coverage report
~~~~~~~~~~~~~~~~~~~~~~~~

The command line flag ``-cunumeric:report:dump-csv`` may be added to save a
detailed coverage report:

.. code-block:: sh

    legate test.py -cunumeric:report:dump-csv out.csv

After execution completes, a CSV file will be saved to the specified location
(in this case ``out.csv``). The file shows exactly what NumPy API functions
were called, whether the are implemented by cunumeric, and the location of
the call site:

.. code-block::

    function_name,location,implemented
    numpy.array,tests/dot.py:27,True
    numpy.ndarray.__init__,tests/dot.py:27,True
    numpy.array,tests/dot.py:28,True
    numpy.ndarray.__init__,tests/dot.py:28,True
    numpy.ndarray.dot,tests/dot.py:31,True
    numpy.ndarray.__init__,tests/dot.py:31,True
    numpy.allclose,tests/dot.py:33,True
    numpy.ndarray.__init__,tests/dot.py:33,True

Call stack reporting
~~~~~~~~~~~~~~~~~~~~

The command line flag ``-cunumeric:report:dump-callstack`` may be added to
include full call stack information in a CSV report:

.. code-block:: sh

    legate test.py -cunumeric:report:dump-callstack -cunumeric:report:dump-csv out.csv

After execution completes, the CSV output file have full call stack
information in the location column, with individual stack frames separated
by pipe (``|``) characters:

Zero code-change patching
-------------------------

The ``lgpatch`` script in the same location as the ``legate`` executable) can
help facilitate quick demonstrations of ``cunumeric`` on existing codebases
that make use of ``numpy``.

To use this tool, invoke it as shown below, with the name of the program to
patch:

.. code-block:: sh

    lgpatch <program> -patch numpy

For example, here is a small ``test.py`` program that imports and uses various
``numpy`` funtions:

.. code-block:: python

    # test.py

    import numpy as np
    input = np.eye(10, dtype=np.float32)
    np.linalg.cholesky(input)

You can invoke ``lgpatch`` to run ``test.py`` using ``cunumeric`` functions
instead, without any changes to the original source code. Any standard
``cunumeric`` runtime options (e.g. for coverage reporting) may also be added:

.. code-block:: sh

    $ lgpatch test.py -patch numpy -cunumeric:report:coverage
    cuNumeric API coverage: 4/4 (100.0%)
