<!--
Copyright 2021 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-->

# Legate NumPy 

Legate NumPy is a [Legate](https://github.com/nv-legate/legate.core) library
that aims to provide a distributed and accelerated drop-in replacement for the 
[NumPy API](https://numpy.org/doc/stable/reference/) on top of the 
[Legion](https://legion.stanford.edu) runtime. Using Legate NumPy you do things like run 
[the final example of the Python CFD course](https://github.com/barbagroup/CFDPython/blob/master/lessons/15_Step_12.ipynb)
completely unmodified on 2048 A100 GPUs in a [DGX SuperPOD](https://github.com/barbagroup/CFDPython/blob/master/lessons/15_Step_12.ipynb) and achieve good weak scaling.

<img src="docs/figures/cfd-demo.png" alt="drawing" width="500"/>

Legate NumPy works best for programs that have very large arrays of data 
that cannot fit in the memory of a single GPU or a single node and need 
to span multiple nodes and GPUs. While our implementation of the current 
NumPy API is still incomplete, programs that use unimplemented features 
will still work (assuming enough memory) by falling back to the 
canonical NumPy implementation.

1. [Dependencies](#dependencies)
2. [Usage and Execution](#usage-and-execution)
3. [Supported and Planned Features](#supported-and-planned-features)
4. [Supported Types and Dimensions](#supported-types-and-dimensions)
5. [Future Directions](#future-directions)
6. [Known Bugs](#known-bugs)

## Dependencies

Users must have a working installation of the 
[Legate Core](https://github.com/nv-legate/legate.core)
library prior to installing Legate NumPy. 

Legate NumPy requires Python >= 3.6. We provide a 
[conda environment file](conda/legate_numpy_dev.yml) that
installs all needed dependencies in one step. Use the following command to
create a conda environment with it:
```
conda env create -n legate -f conda/legate_numpy_dev.yml
```

### Installation

Installation of Legate NumPy is done with either `setup.py` for simple 
uses cases or `install.py` for more advanced use cases. The most common 
installation command is:

```
python setup.py --with-core <path-to-legate-core-installation>
```

This will build Legate NumPy against the Legate Core installation and then 
install Legate NumPy into the same location. Users can also install Legate NumPy 
into an alternative location with the canonical `--prefix` flag as well.

```
python setup.py --prefix <install-dir> --with-core <path-to-legate-core-installation>
```

Note that after the first invocation of `setup.py` this repository will remember 
which Legate Core installation to use and the `--with-core` option can be 
omitted unless the user wants to change it.

Advanced users can also invoke `install.py --help` to see options for 
configuring Legate NumPy by invoking the `install.py` script directly.

Of particular interest to Legate NumPy users will likely be the option for
specifying an installation of [OpenBLAS](https://www.openblas.net/) to use.
If you already have an installation of OpenBLAS on your machine you can 
inform the `install.py` script about its location using the `--with-openblas` flag:
```
python setup.py --with-openblas /path/to/open/blas/
```

## Usage and Execution

Using Legate NumPy as a replacement for NumPy is easy. Users only need
to replace:

```
import numpy as np
```

with:

```
import legate.numpy as np
```

These programs can then be run by the Legate driver script described in the
[Legate Core](https://github.com/nv-legate/legate.core) documentation.

```
legate legate_numpy_program.py
```

For execution with multiple nodes (assuming Legate Core is installed with GASNet support)
users can supply the `--nodes` flag. For execution with GPUs, users can use the 
`--gpus` flags to specify the number of GPUs to use per node. We encourage all users
to familiarize themselves with these resource flags as described in the Legate Core
documentation or simply by passing `--help` to the `legate` driver script.

## Supported and Planned Features

Legate NumPy is currently a work in progress and we are gradually adding support for 
additional NumPy operators. Unsupported NumPy operations will provide a
warning that we are falling back to canonical NumPy. Please report unimplemented
features that are necessary for attaining good performance so that we can triage
them and prioritize implementation appropriately. The more users that report an 
unimplemented feature, the more we will prioritize it. Please include a pointer
to your code if possible too so we can see how you are using the feature in context.

## Supported Types and Dimensions

Legate NumPy currently supports the following NumPy types: `float16`, `float32`, `float64`,
`int16`, `int32`, `int64`, `uint16`, `uint32`, `uint64`, `bool`, `complex64`, and `complex128`. 
Legate currently also only works on up to 3D arrays at the moment. We're currently working
on support for N-D arrays. If you have a need for arrays with more than three
dimensions please let us know about it.

## Future Directions

There are three primary directions that we plan to investigate 
with Legate NumPy going forward: 

* More features: we plan to identify a few key lighthouse applications
  and use the demands of these applications to drive the addition of 
  new features to Legate NumPy.
* We plan to add support for sharded file I/O for loading and
  storing large data sets that could never be loaded on a single node.
  Initially this will begin with native support for [h5py](https://www.h5py.org/)
  but will grow to accommodate other formats needed by our lighthouse
  applications.
* Strong scaling: while Legate NumPy is currently implemented in a way that
  enables weak scaling of codes on larger data sets, we would also like
  to make it possible to strong-scale Legate applications for a single
  problem size. This will require leveraging some of the more advanced
  features of Legion from inside the Python interpreter.

We are open to comments, suggestions, and ideas.

## Known Bugs

 * Legate NumPy can exercise a bug in OpenBLAS when it is run with
   [multiple OpenMP processors](https://github.com/xianyi/OpenBLAS/issues/2146)
 * On Mac OSX, Legate NumPy can trigger a bug in Apple's implementation of libc++.
   The [bug](https://bugs.llvm.org/show_bug.cgi?id=43764) has since been fixed but
   likely will not show up on most Apple machines for quite some time. You may have
   to manually patch your implementation of libc++. If you have trouble doing this
   please contact us and we will be able to help you.
