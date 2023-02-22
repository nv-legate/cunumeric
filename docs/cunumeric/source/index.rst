:html_theme.sidebar_secondary.remove:

Welcome to cuNumeric's documentation!
=====================================

cuNumeric is a `Legate`_ library that aims to provide a distributed and
accelerated drop-in replacement for the `NumPy API`_ on top of the `Legion`_
runtime.

Using cuNumeric you do things like run the final example of the
`Python CFD course`_ completely unmodified on 2048 A100 GPUs in a
`DGX SuperPOD`_ and achieve good weak scaling.

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  user/index
  comparison/index
  api/index
  developer/index

.. toctree::
  :maxdepth: 1

  versions


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`

.. _DGX SuperPOD: https://github.com/barbagroup/CFDPython/blob/master/lessons/15_Step_12.ipynb
.. _Legate: https://github.com/nv-legate/legate.core
.. _Legion: https://legion.stanford.edu/
.. _Numpy API: https://numpy.org/doc/stable/reference/
.. _Python CFD course: https://github.com/barbagroup/CFDPython/blob/master/lessons/15_Step_12.ipynb