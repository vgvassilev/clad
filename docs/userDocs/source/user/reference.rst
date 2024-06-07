API reference
======================


.. cpp:class:: CladFunction

   Provides an interface to easily access, call and print the differentiated
   function.

   .. todo::

      Add class member documentation.

------------------

   .. _api_reference_clad_differentiate:

   .. cpp:function:: template<class Fn>\
                  CladFunction differentiate(Fn fn, const char* args)


   In very brief, this function differentiate functions using the forward mode
   automatic differentiation.

   More specifically, this function performs partial differentiation of the
   provided function (``fn``) using the forward mode automatic differentiation
   with respect to parameter specified in ``args``. Template parameter ``N``
   denotes the derivative order.

   Please refer this to know more about the forward mode automatic differentiation.
   For now it is enough to know that forward mode automatic differentiation (AD)
   is more efficient than the reverse mode automatic differentiation when the
   number of output parameters of the function are greater than the number of
   input paramters of the function.

   ::

      #include "clad/Differentiator/Differentiator.h"

      double func(double x, double y) { return x * x * y + y * y; }

      int main(){

         // fn_dx is of type CladFunction, which is a tiny wrapper
         // over the derived function pointer.
         // It will differentiate 'func' w.r.t 'x'.
         auto fn_dx = clad::differentiate(func, "x");

         // Using CladFunction::execute method when (x, y) = (5, 3)
         double func1stOrderDerivative = fn_dx.execute(5,3);
         printf("Result is %d\n", func1stOrderDerivative); //Result is 30

      }

   .. cpp:function:: template<class Fn>\
                  CladFunction gradient(Fn fn, const char* args)

   In very brief, this function differentiate functions using the reverse mode
   automatic differentiation.

   More specifically, this function performs partial differentiation of the provided
   function (``fn``) using the reverse mode automatic differentiation with respect
   to all the parameters specified in ``args``.

   Please refer this to know more about the reverse mode automatic differentiation.
   For now it is enough to know that generally reverse mode AD is more efficient
   than the forward mode AD when there are multiple input paramters.

   ::

      #include "clad/Differentiator/Differentiator.h"

      double func (double i, double j) {
         return 5*i*i + 2*j;
      }
      int main() {
         auto fn_grad = clad::gradient(func);
         double d_i = 0, d_j = 0;
         fn_grad.execute(3, 5, &d_i, &d_j);
         printf("Result is %g , %g \n", d_i, d_j); //Result is 30 , 2
      }

   .. cpp:function:: template<class Fn>\
                  CladFunction hessian(Fn fn, const char* args)

   This function generates a function that can be used to compute
   `hessian matrix <https://en.wikipedia.org/wiki/Hessian_matrix>`_
   of the provided function (``fn``) with respect to all the arguments
   specified in ``args``.

   ::

      #include "clad/Differentiator/Differentiator.h"
      double func(double i, double j) {
         double a = i * j;
         double b = 4 * a;
         return b * i;
      }
      int main() {

        auto fn_hesn = clad::hessian(func);

        // Creates an empty matrix to store the Hessian in
        double matrix[4] = {0};

        // Clad requires array size information as well
        fn_hesn.execute(8, 2, matrix);

        // Result is 16, 64, 64,0
        printf("Result is %g, %g, %g,%g \n", matrix[0], matrix[1],
               matrix[2], matrix[3]);
      }

   .. cpp:function:: template<class Fn>\
                  CladFunction jacobian(Fn fn, const char* args)

   This function generates a function that can be used to compute
   `jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_
   of the provided function (``fn``) with respect to all
   the arguments specified in ``args``. If no explicit ``args`` argument is specified,
   then jacbian matrix is computed with respect to all the input parameters.
   For a function with 3 input parameters and an output array of size 4,
   the jacobian matrix will contain 12 elements.

    ::

      #include "clad/Differentiator/Differentiator.h"
      void func(double i, double j, double result[]) {
        result[0] = i * i * j;
        result[1] = j * j * i;
        result[2] = j * i;
      }
      int main() {

        auto fn_jcbn = clad::jacobian(func);

        // Creates an empty matrix to store the Jacobian in
        double matrix[6] = {0};
        double res[3] = {0};

        fn_jcbn.execute(8, 2, res, matrix);

        //Result is 48, 64, 4, 32, 2, 8
        printf("Result is %g, %g, %g, %g, %g, %g \n", matrix[0], matrix[1],
               matrix[2], matrix[3], matrix[4], matrix[5]);
      }

------------------

.. todo::

   Add numerical differentiation and error estimation framework API reference.