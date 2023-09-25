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

See the :ref:`config` section :ref:`config_legate` for more information.

Additionally, any Legion and Realm arguments may also be passed via the
``LEGION_DEFAULT_ARGS`` environment variable:

.. code-block:: sh

  LEGION_DEFAULT_ARGS="-lg:sched 100 -ll:show_rsrv" legate script.py <script options>

Using standard Python
~~~~~~~~~~~~~~~~~~~~~

It is also possible to run cuNumeric programs using the standard Python
interpreter, with some limitations:

.. code-block:: sh

  LEGATE_CONFIG="--cpus 2 --gpus 2" python script.py <script options>

When running programs with this method, Legate configuration options may only
be passed via the ``LEGATE_CONFIG`` environment variable as shown above.

Additionally, any Legion and Realm arguments must also be passed via the
``LEGION_DEFAULT_ARGS`` environment variable:

.. code-block:: sh

  LEGION_DEFAULT_ARGS="-lg:sched 100 -ll:show_rsrv" python script.py <script options>

.. note::

  Usage of standard Python is intended as a quick on-ramp for users to try
  out cuNumeric more easily. Several ``legate`` command line configuration
  options, especially for  multi-node execution, are not available when
  running programs with standard Python. See the output of ``legate --help``
  for more details.

Multi-node execution
--------------------

Using ``legate``
~~~~~~~~~~~~~~~~

Cunumeric programs can be run in parallel by using the ``--nodes`` option to
the ``legate`` driver, followed by the number of nodes to be used.
When running on 2+ nodes, a task launcher must be specified.

Legate currently supports using ``mpirun``, ``srun``, and ``jsrun`` as task
launchers for multi-node execution via the ``--launcher`` command like
arguments:

.. code-block::

  legate --launcher srun --nodes 2 script.py <script options>

See the :ref:`config` section :ref:`config_multi_node` for more
configuration options.

Using a manual task manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

  mpirun -np N legate script.py <script options>

It is also possible to use "standard python" in place of the ``legate`` driver.

Running Numpy programs without changes
--------------------------------------

The ``lgpatch`` script (in the same location as the ``legate`` executable) can
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

