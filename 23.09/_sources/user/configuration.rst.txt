.. _config:

Configuration
=============

The underlying Legate runtime has many options for configuring the details of
execution.

How to specify configuration
----------------------------

.. _config_legate:

Legate driver
~~~~~~~~~~~~~

When using the ``legate`` driver, it is possible to pass most configuration
options via the command line, as long as they appear *before* the script to
run:

.. code-block:: sh

  legate <legate options> script.py <script options>


For example to configure the CPUs and system memory per rank, you could
execute:

.. code-block:: sh

  legate --cpus 2 --sysmem 8000 script.py


It is also possible to pass these configuration options using the
``LEGATE_CONFIG`` environment variable as shown below:

.. code-block:: sh

  LEGATE_CONFIG="--cpus 2 --sysmem 8000" legate script.py

.. note::

  Options provided via ``LEGATE_CONFIG`` have lower precedence than arguments
  passed directly on the command line.

Standard Python
~~~~~~~~~~~~~~~

When invoking scripts using the standard Python interpreter, rather that the
``legate`` driver, then configuration options must be passed using the
environment variable:

.. code-block:: sh

  LEGATE_CONFIG="--cpus 2 --sysmem 8000" python script.py

Configuration scenarios
-----------------------

The sections below describe some of the common ``legate`` configuration
options and when you might use them. For a complete list of options, exectute
the command ``legate --help``.

Resource allocation
~~~~~~~~~~~~~~~~~~~

--cpus CPUS
  Number of CPUs to use per rank (default: 4)

--gpus GPUS
  Number of OpenMP groups to use per rank (default: 0)

--utility UTILITY
  Number of Utility processors per rank to request for meta-work (default: 2)

--sysmem SYSMEM
  Amount of DRAM memory per rank (in MBs) (default: 4000)

--fbmem FBMEM
  Amount of framebuffer memory per GPU (in MBs) (default: 4000)

.. _config_multi_node:

Multi-node execution
~~~~~~~~~~~~~~~~~~~~

When using the ``legate`` driver, the configuration options below can be used
to launch ``cunumeric`` code for multi-node execution.

--nodes NODES
  Number of nodes to use (default: 1)

--ranks-per-node RANKS_PER_NODE
  Number of ranks (processes running copies of the program) to launch per node.
  The default (1 rank per node) will typically result in the best performance.
  (default: 1)

--launcher LAUNCHER
  *{mpirun,jsrun,srun,none}*
  launcher program to use (set to "none" for local runs, or if the launch has
  already happened by the time legate is invoked), [legate-only, not supported
  with standard Python invocation] (default: none)

--launcher-extra LAUNCHER_EXTRA
  Additional argument to pass to the launcher (can appear more than once).
  Multiple arguments may be provided together in a quoted string (arguments
  with spaces inside must be additionally quoted), [legate-only, not supported
  with standard Python invocation] (default: [])
