Array manipulation routines
===========================

.. currentmodule:: cunumeric

Basic operations
----------------

.. autosummary::
   :toctree: generated/

   ndim
   shape


Changing array shape
--------------------

.. autosummary::
   :toctree: generated/

   reshape
   ravel

Transpose-like operations
-------------------------

.. autosummary::
   :toctree: generated/

   moveaxis
   swapaxes
   transpose

See also :attr:`cunumeric.ndarray.T` property.

Changing number of dimensions
-----------------------------

.. autosummary::
   :toctree: generated/

   atleast_1d
   atleast_2d
   atleast_3d
   broadcast_arrays
   broadcast_shapes
   broadcast_to
   squeeze

Changing kind of array
----------------------

.. autosummary::
   :toctree: generated/

   asarray


Joining arrays
--------------

.. autosummary::
   :toctree: generated/

   append
   concatenate
   stack
   block
   vstack
   hstack
   dstack
   column_stack
   row_stack


Splitting arrays
----------------

.. autosummary::
   :toctree: generated/

   split
   array_split
   dsplit
   hsplit
   vsplit


Tiling arrays
-------------

.. autosummary::
   :toctree: generated/

   tile


Rearranging elements
--------------------

.. autosummary::
   :toctree: generated/

   flip
   fliplr
   flipud
