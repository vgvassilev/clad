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
differentiation (precision loss) and symbolic differentiation 
(inefficient code produced). If you are just getting started with clad, then please 
checkout :doc:`Using Clad <user/UsingClad>` and
:doc:`Tutorials <user/tutorials>`.

----------------------

Clad example use::

   #include "clad/Differentiator/Differentiator.h"
   #include <iostream>

   double f(double x, double y) { return x * y; }

   int main() {
     auto f_dx = clad::differentiate(f, "x");
     // computes derivative of 'f' when (x, y) = (3, 4) and prints it.
     std::cout << f_dx.execute(3, 4) << std::endl; // prints: 4
     f_dx.dump(); // prints:
     /* double f_darg0(double x, double y) {
         double _d_x = 1; double _d_y = 0;
          return _d_x * y + x * _d_y;
        } */
   }

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
   user/tutorials
   user/UsingEnzymeWithinClad
   user/UsingVectorMode.rst
   user/FAQ
   user/DevelopersDocumentation
   user/IntroductionToClangForCladContributors
   user/FloatingPointErrorEstimation
   
Citing Clad
-------------

Founders
---------

License
--------
