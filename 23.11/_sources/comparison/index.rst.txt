Project comparisons
===================

Here is a list of NumPy APIs and corresponding cuNumeric implementations.

A dot in the cunumeric column denotes that cuNumeric implementation
is not provided yet. We welcome contributions for these functions.

NumPy vs cuNumeric APIs
-----------------------

.. comparison-table::

.. _measuring api coverage:

Measuring API coverage
----------------------

When running applications that use cunumeric, various command line options may
be used to generate coverage reports.

Overall coverage report
~~~~~~~~~~~~~~~~~~~~~~~

The environment variable ``CUNUMERIC_REPORT_COVERAGE`` may be used to print an
overall percentage of cunumeric coverage:

.. code-block:: sh

    CUNUMERIC_REPORT_COVERAGE=1 legate test.py

After execution completes, the percentage of NumPy API calls that were handled
by cunumeric is printed:

.. code-block::

    cuNumeric API coverage: 26/26 (100.0%)

Detailed coverage report
~~~~~~~~~~~~~~~~~~~~~~~~

The environment variable ``CUNUMERIC_REPORT_DUMP_CSV`` may be used to save a
detailed coverage report:

.. code-block:: sh

    CUNUMERIC_REPORT_COVERAGE=1 CUNUMERIC_REPORT_DUMP_CSV="out.csv" legate test.py

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

The environment variable ``CUNUMERIC_REPORT_DUMP_CALLSTACK`` may be added to
include full call stack information in a CSV report:

.. code-block:: sh

   CUNUMERIC_REPORT_COVERAGE=1 CUNUMERIC_REPORT_DUMP_CALLSTACK=1 CUNUMERIC_REPORT_DUMP_CALLSTACK=1 legate test.py

After execution completes, the CSV output file have full call stack
information in the location column, with individual stack frames separated
by pipe (``|``) characters:

