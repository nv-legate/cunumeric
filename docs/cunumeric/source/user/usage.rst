Usage
=====

Drop-in replacement
-------------------

Using cuNumeric as a replacement for NumPy is easy. Users only need
to replace:

.. code-block:: python

  import numpy as np

with

.. code-block:: python

  import cunumeric as np

Running cuNumeric programs
--------------------------

Using ``legate``
~~~~~~~~~~~~~~~~

The preferred way to run cuNumeric programs is with the ``legate`` driver:

.. code-block:: sh

  legate --cpus 2 --gpus 2 script.py <script options>

All ``legate`` command line configuration options must come *before* the script
that is be executed. Any options after the script are passed on to the script.

It is also possible to pass the ``legate`` command line configuration options
using the ``LEGATE_CONFIG`` environment variable:

.. code-block:: sh

  LEGATE_CONFIG="--cpus 2 --gpus 2" legate script.py <script options>

Using standard Python
~~~~~~~~~~~~~~~~~~~~~

It is also possible to run cuNumeric programs using the standard Python
interpreter, with some limitations:

.. code-block:: sh

  LEGATE_CONFIG="--cpus 2 --gpus 2" python script.py <script options>

When running programs with this method, configuration options may only be
passed via the ``LEGATE_CONFIG`` environment variable as shown above.

Additionally, several ``legate`` command line configuration options are not
available when running programs this way. See the output of ``legate --help``
for more details.

Configuration options
---------------------

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
``cunumeric`` runtime options (e.g. for :ref:`measuring api coverage`) may
also be used:

.. code-block:: sh

    $ CUNUMERIC_REPORT_COVERAGE=1 LEGATE_CONFIG="--cpus 4"  lgpatch test.py -patch numpy
    cuNumeric API coverage: 4/4 (100.0%)

