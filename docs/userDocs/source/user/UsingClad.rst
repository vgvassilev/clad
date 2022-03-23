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
Clad currently supports differentiating arrays for forward, reverse, hessian and error estimation modes. The interface
for these vary a bit.

Forward mode: The interface requires the user to provide the exact index of the array for which the function is to
be differentiated. The interface of the diff function remains the same as before. An example is given below::

    #include "clad/Differentiator/Differentiator.h"

    double f (double arr[4]) { return arr[0] * arr[1] * arr[2] * arr[3]; }

    int main() {
        // Differentiating the function f w.r.t arr[1] :
        auto f_diff = clad::differentiate(f, "arr[1]");

        double arr[4] = {1, 2, 3, 4};
        // Pass the input to f to the execute function
        // The output is stored in a variable with the same type as the return type of f
        double f_dx = f_diff.execute(arr);

        printf("df/darr[2] = %g\n", f_dx);
    }

Reverse mode: The interface doesn't require any specific index to be mentioned. The interface of the diff function
requires you to pass `clad::array_ref<T>` for the independent variables after you pass the inputs to the original
function. The `T` here is the return type of the original function. The example below will explain it better::

    #include "clad/Differentiator/Differentiator.h"

    double g(double x, double arr[2]) { return x * arr[0] + x * arr[1]; }

    int main() {
        // Differentiating g w.r.t all the input variables (x, arr)
        auto g_grad = clad::gradient(g);

        double x = 2, arr[2] = { 1, 2 };
        // Create memory for the output of differentiation
        double dx = 0, darr[2] = { 0 };

        // Create an clad::array_ref out of darr, no need to create an
        // array_ref for dx we can just pass the pointer to dx
        clad::array_ref<double> darr_ref(darr, 2);

        // The inputs to the original function g (i.e x and arr) are passed
        // followed by the variables to store the output (i.e dx and darr)
        g_grad.execute(x, arr, &dx, darr_ref);

        printf("dg/dx = %g \ndg/darr = { %g, %g } \n", dx, darr[0], darr[1]);
    }

Hessian Mode: The interface requires the indexes of the array being differentiated to be mentioned explicitly even if
you are trying to differentiate w.r.t the whole array. The interface of the diff function requires you to pass an
`clad::array_ref<T>` after passing the inputs to the original function. The `T` is the return type of the original
function and the size of the `clad::array_ref` should be at least the square of the number of independent variables
(each index of an array is counted as one independent variable). Example::

    #include "clad/Differentiator/Differentiator.h"

    double h(double x, double arr[3]) { return x * arr[0] * arr[1] * arr[2]; }

    int main() {
        // Differentiating h w.r.t all the input variables (x, arr)
        // Note that the array and the indexes are explicitly mentioned even though all the indexes (0, 1 and 2)
        // are being differentiated
        auto h_hess = clad::hessian(h, "x, arr[0:2]");

        double x = 2, arr[3] = { 1, 2, 3 };

        // Create memory for the hessian matrix
        // The minimum required size of the matrix is the square of the
        // number of independent variables
        // Since there are 3 indexes of the array and a scalar variable
        // the total number of independent variables are 4
        double mat[16];

        // Create a clad::array_ref for the matrix
        clad::array_ref<double> mat_ref(mat, 16);

        // The inputs to the original function h (i.e x and arr) are passed
        // followed by the output matrix
        h_hess.execute(x, arr, mat_ref);

        printf("hessian matrix: \n"
               "{ %g, %g, %g, %g\n"
               "  %g, %g, %g, %g\n"
               "  %g, %g, %g, %g\n"
               "  %g, %g, %g, %g }\n",
                mat[0], mat[1], mat[2], mat[3],
                mat[4], mat[5], mat[6], mat[7],
                mat[8], mat[9], mat[10], mat[11],
                mat[12], mat[13], mat[14], mat[15]);
    }

Error estimation: This interface is the same as with reverse mode.

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


