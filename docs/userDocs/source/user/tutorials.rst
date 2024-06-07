Tutorials
----------
   
Clad is an open source clang plugin which supports automatic differentiation of 
mathematical functions in C++. Currently Clad supports four modes for automatic 
differentiation namely forward, reverse, Hessian, Jacobian.

**The Forward mode** 

Clad supports forward mode automatic differentiation through the `clad::differentiate`
API call.

.. code-block:: cpp 

 #include <iostream>
 #include "clad/Differentiator/Differentiator.h"

 double func(int x) { return x * x; }

 int main() {
   /*Calling clad::differentiate to get the forward mode derivative of
   the given mathematical function*/
   auto d_func = clad::differentiate(func, "x");
   // execute the generated derivative function.
   std::cout << d_func.execute(/*x =*/3) << std::endl;
   // Dump the generated derivative code to std output.
   d_func.dump();
 }

Here we are differentiating a function `func` which takes an input `x` and 
returns a scaler value `x * x`.`.dump()` method is used to get a dump of generated 
derivative function to the standard output.

**The Reverse Mode** 
 
Clad also supports reverse mode automatic differentiation, through the `clad::gradient` 
API call.

.. code-block:: cpp  

 #include <iostream>
 #include "clad/Differentiator/Differentiator.h"

 double f(double x, double y, double z) { return x * y * z; }

 int main() {
   auto d_f = clad::gradient(f, "x, y");
   double dx = 0, dy = 0;
   d_f.execute(/*x=*/2, /*y=*/3, /*z=*/4, &dx, &dy);
   std::cout << "dx : " << dx << "dy :" << dy << std::endl;
 }

In the above example we are differentiating w.r.t `x and y` we can also 
differentiate w.r.t to single argument i.e. either `x` or `y` as `clad::gradient(f, "x")` 
not writing any argument i.e. `clad::gradient(f)` will result in differentiation 
of the function w.r.t to each input. 


**The Hessian Mode**

Clad can also produce an hessian matrix through the `clad::hessian` API call.
It returns the hessian matrix as a flattened vector in row major format.

.. code-block:: cpp

 #include <iostream>
 #include "clad/Differentiator/Differentiator.h"

 double f(double x, double y, double z) { return x * y * z; }

 // Function with array input

 double f_arr(double x, double y, double z[2]) { return x * y * z[0] * z[1]; }

 int main() {
   // Workflow similar to clad::gradient for non-array input arguments.
   auto f_hess = clad::hessian(f, "x, y");
   double matrix_f[9] = {0};
   f_hess.execute(3, 4, 5, matrix_f);
   std::cout << "[" << matrix_f[0] << ", " << matrix_f[1]
             << matrix_f[2] << "\n"
             << matrix_f[3] << ", " << matrix_f[4] << matrix_f[5]
             << "\n"
             << matrix_f[6] << ", " << matrix_f[7] << matrix_f[8]
             << "]"
             << "\n";
 }

When arrays are involved we need to specify the array index that needs to be 
differentiated. For example if we want to differentiate w.r.t to the first two 
elements of the array along with `x` and `y` we will write `clad::hessian(f_arr, z[0:1])` 
for the above example rest of the steps for execution are similar to reverse mode.
Here the array variable stores the hessian matrix.


**The Jacobian Mode**

Clad can produce Jacobian of a function using its reverse mode. It returns the 
jacobian matrix as a flattened vector with elements arranged in row-major format.

.. code-block:: cpp

 #include <iostream>
 #include "clad/Differentiator/Differentiator.h"

 void f(double x, double y, double z, double* output) {
   output[0] = x * y;
   output[1] = y * y * x;
   output[2] = 6 * x * y * z;
 }

 int main() {
   auto f_jac = clad::jacobian(f);

   double jac[9] = {0};
   double output[3] = {0};
   f_jac.execute(3, 4, 5, output, jac);
   std::cout << jac[0] << " " << jac[1] << std::endl
             << jac[2] << " " << jac[3] << std::endl
             << jac[4] << " " << jac[5] << std::endl
             << jac[6] << " " << jac[7] << std::endl
             << jac[8] << std::endl;
 }

The jacobian matrix size should be equal to `no. of independent variables times 
the number of outputs in the original function` in the above example it would be
an array of size 3x3 = 9.

**Error Estimation API**

Clad is capable of annotating a given function with floating point error estimation
code using reverse mode AD.

.. code-block::  cpp

 #include <iostream>
 #include "clad/Differentiator/Differentiator.h"

 double func(double x, double y) { return x * y; }

 int main() {

   auto dfunc_error = clad::estimate_error(func);
   // Used to print generated code to standard output.
   dfunc_error.dump();
   double x, y, d_x, d_y, final_error = 0;
   // Call execute
   dfunc_error.execute(x, y, &d_x, &d_y, final_error);

   std::cout << final_error;
 }

The function signature is similar to `clad::gradient` except we need to add an 
extra argument of type `double&` which is used to store the total floating point
error.
