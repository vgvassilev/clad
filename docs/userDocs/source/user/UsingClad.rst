Using Clad
***********

This section briefly describes all the key functionalities offered by Clad.
If you are just getting started with Clad, then this is the best place to start.
You may want to skim some sections on the first read. 

In case you haven't installed Clad already, then please do before proceeding 
with this guide. Visit :doc:`Clad installation and usage <InstallationAndUsage>` 
to know more about installing clad.

Let's get started.

Automatic Differentiation
===========================

.. todo::
   
   Briefly explain automatic differentiation.

Differentiating a function
----------------------------


Forward Mode Automatic Differentiation
----------------------------------------

Reverse Mode Automatic Differentiation
----------------------------------------

Hessian Computation
----------------------

Jacobian Computation
----------------------

The Jacobian matrix is the generalization of the gradient for vector-valued functions of several variables.

Clad can compute the 
 `jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_ of a
 function through the ``clad::jacobian`` interface.

 .. figure:: ../_static/jacobian-matrix.png
   :width: 400
   :align: center
   :alt: Jacobian matrix image taken from Wikipedia

   Jacobian matrix of a function with x\ :sub:`n`\ parameters:  
   (x\ :sub:`1`\ , x\ :sub:`2`\ , ..., x\ :sub:`n`\ ).


 A self-explanatory example that demonstrates the usage of ``clad::jacobian``::

   #include "clad/Differentiator/Differentiator.h"

   void fn_jacobian(double i, double j, double *res) {
      res[0] = i*i;
      res[1] = j*j;
      res[2] = i*j;
   }

   int main() {
     // Generates all first-order partial derivatives columns of a Jacobian matrix
     // and stores CallExprs to them inside a single function 
     auto jacobian = clad::jacobian(fn_jacobian);

     // Creates an empty matrix to store the Jacobian in
     // Must have enough space, 2 columns (independent variables) times 3 rows (2*3=6)
     double matrix[6];

     // Prints the generated Hessian function
     jacobian.dump();

     // Substitutes these values into the Jacobian function and pipes the result
     // into the matrix variable.
     double res[3] = {0, 0, 0};
     jacobian.execute(3, 5, res, matrix);
   }

 Few important things to note through this example:

 - ``clad::jacobian`` supports differentiating w.r.t multiple paramters.

 - The array that will store the computed jacobian matrix needs to be passed as the 
   last argument to ``CladFunction::execute`` call. The array size 
   needs to be greater or equal to the size required to store the jacobian matrix. 
   Passing an array of a smaller size will result in undefined behaviour.

Array Support 
----------------

Functor Support
-----------------

Differentiable Class Types
----------------------------

Custom Derivatives
---------------------

Numerical Differentiation Fallback
====================================


Error Estimation
======================

Debug functionalities
======================


