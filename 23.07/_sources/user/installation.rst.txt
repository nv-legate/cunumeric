Installation
============

cuNumeric is available `from conda`_:

.. code-block:: sh

  conda install -c nvidia -c conda-forge -c legate cunumeric

Only linux-64 packages are available at the moment.

The default package contains GPU support, and is compatible with CUDA >= 11.4
(CUDA driver version >= r470), and Volta or later GPU architectures. There are
also CPU-only packages available, and will be automatically selected by
``conda`` when installing on a machine without GPUs.

See :ref:`building cunumeric from source` for instructions on building cuNumeric manually.

.. _from conda: https://anaconda.org/legate/cunumeric