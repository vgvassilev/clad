Tutorials
----------

Clad is an open source clang plugin which supports automatic differentiation of
mathematical functions in C++. This page walks through one small example per
mode.

Every example below is a file in clad's test suite, included here verbatim. The
``// prints:`` comments are what the suite checks the program writes, so the
code and the numbers beside it are kept true by the build.

**The Forward mode**

Clad supports forward mode automatic differentiation through the `clad::differentiate`
API call. It differentiates with respect to one parameter, named in the second
argument, and the generated function returns the derivative.

.. literalinclude:: ../../../../test/Documentation/Tutorials/ForwardMode.cpp
   :language: cpp
   :start-after: docs-begin-forward-mode
   :end-before: docs-end-forward-mode

`.dump()` prints the derivative clad generated, which is often the quickest way
to see what it did.

**The Reverse Mode**

Clad also supports reverse mode automatic differentiation, through the `clad::gradient`
API call. One call computes the derivatives with respect to every parameter
named in `args`, and writes each one through a pointer the caller supplies.

.. literalinclude:: ../../../../test/Documentation/Tutorials/ReverseMode.cpp
   :language: cpp
   :start-after: docs-begin-reverse-mode
   :end-before: docs-end-reverse-mode

The example differentiates with respect to `x` and `y`. Naming one parameter,
as in `clad::gradient(f, "x")`, differentiates with respect to that one; naming
none, as in `clad::gradient(f)`, differentiates with respect to all of them.

**The Hessian Mode**

Clad can also produce a hessian matrix through the `clad::hessian` API call.
It returns the matrix as a flattened array in row major order, so `n`
independent variables need `n * n` elements.

.. literalinclude:: ../../../../test/Documentation/Tutorials/Hessian.cpp
   :language: cpp
   :start-after: docs-begin-hessian
   :end-before: docs-end-hessian

When an array is involved, say which elements to differentiate with respect to:
for `double f_arr(double x, double y, double z[2])`, the call
`clad::hessian(f_arr, "x, y, z[0:1]")` uses four independent variables and so
needs sixteen elements.

**The Jacobian Mode**

Clad can produce the jacobian of a function using its vectorized forward mode.
It returns
the jacobian as a `clad::matrix` for every pointer or array parameter.

.. literalinclude:: ../../../../test/Documentation/Tutorials/Jacobian.cpp
   :language: cpp
   :start-after: docs-begin-jacobian
   :end-before: docs-end-jacobian

The matrix has one row per element of the output and one column per independent
scalar. Here the output has three elements, and the independent scalars are
`x`, `y`, `z` and the three elements of `output` itself, which is why it is
3 x 6. The last three columns come out zero here, because `output` does not
depend on its own previous contents.

**Error Estimation API**

Clad is capable of annotating a given function with floating point error
estimation code using reverse mode AD.

.. literalinclude:: ../../../../test/Documentation/Tutorials/ErrorEstimation.cpp
   :language: cpp
   :start-after: docs-begin-error-estimation
   :end-before: docs-end-error-estimation

The signature is the one `clad::gradient` would generate, with one extra
argument of type `double&` at the end, which receives the total floating point
error.
