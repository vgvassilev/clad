.. Clad documentation master file, created by
   sphinx-quickstart on Fri Sep 17 10:48:01 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Clad: Automatic differentiation plugin for C++
===============================================

Release v\ |version|.

Overview
------------

Clad enables
`automatic differentiation (AD) <https://en.wikipedia.org/wiki/Automatic_differentiation>`_
for C++. It is based on LLVM compiler infrastructure and is a plugin for
`Clang compiler <http://clang.llvm.org/>`_. Clad is based on source code
transformation. Given C++ source code of a mathematical function, it can
automatically generate C++ code for computing derivatives of the function.

.. todo::

   Add section that describes complete set of supported language features.

Clad supports a large set of C++ features including control flow statements and
function calls. Please visit (add hyperlink here) to know more about the
support of language features. It supports reverse-mode AD (a.k.a backpropagation)
as well as forward-mode AD. It also facilitates computation of hessian matrix and
jacobian matrix of any arbitrary function.

Automatic differentiation solves all the usual problems of numerical
differentiation such as precision loss and symbolic differentiation such as
limitations to closed form expressions and expression swell.
If you are just getting started with clad, then please
checkout :doc:`Using Clad <user/UsingClad>` and
:doc:`Tutorials <user/tutorials>`.

----------------------

Clad example use:

.. literalinclude:: ../../../test/Documentation/Guide/Overview.cpp
   :language: cpp
   :start-after: docs-begin-overview
   :end-before: docs-end-overview

Features
-----------

- Requires little to no code modification for computing derivatives of existing codebase.
- Features both reverse mode AD (backpropagation) and forward mode AD.
- Computes derivatives of functions, member functions, functors and lambda expressions.
- Supports large subset of C++ including if statements, for, while loops and so
  much more; it is actively being developed with the goal of supporting all of
  C++ syntax.
- Provides direct functions for computation of Hessian matrix and Jacobian matrix.
- Supports array differentiation, that is, it can differentiate either with
  respect to whole arrays or particular indices of the array.
- Features numerical differentiation support, to be used as a fallback where
  automatic differentiation is not feasible.


.. comment


   .. todo::

      Add more features such as error estimation, custom derivatives, class type support etc.

The User Guide
---------------

.. toctree::
   :maxdepth: 2

   user/InstallationAndUsage
   user/UsingClad
   user/CoreConcepts
   user/reference
   user/CustomDerivatives.rst
   user/tutorials
   user/UsingEnzymeWithinClad
   user/UsingVectorMode.rst
   user/UsingImmediateMode
   user/UsingCladOnCUDACode
   user/FAQ
   user/DevelopersDocumentation
   user/IntroductionToClangForCladContributors
   user/FloatingPointErrorEstimation

Citing Clad
-------------

If Clad contributed to your work, please cite the paper describing it:

.. code-block:: bibtex

   % 16th International workshop on Advanced Computing and Analysis Techniques
   % in physics research (ACAT), 1-5 September, 2014, Prague, The Czech Republic
   @inproceedings{Vassilev_Clad,
     author = {Vassilev,V. and Vassilev,M. and Penev,A. and Moneta,L. and Ilieva,V.},
     title = {{Clad -- Automatic Differentiation Using Clang and LLVM}},
     journal = {Journal of Physics: Conference Series},
     year = 2015,
     month = {may},
     volume = {608},
     number = {1},
     pages = {012055},
     doi = {10.1088/1742-6596/608/1/012055},
     url = {https://iopscience.iop.org/article/10.1088/1742-6596/608/1/012055/pdf},
     publisher = {{IOP} Publishing}
   }

Founders
---------

Clad was founded by Vassil Vassilev, as part of his research interests and
vision. He holds the exclusive copyright and other related rights, described in
`Copyright.txt <https://github.com/vgvassilev/clad/blob/master/Copyright.txt>`__.

License
--------

Clad is an open source project, licensed under the GNU Lesser General Public
License. A module under a different license says so in the ``License.txt`` of
its own source folder. See
`License.txt <https://github.com/vgvassilev/clad/blob/master/License.txt>`__
for the full text.
