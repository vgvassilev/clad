Using Vector Mode for Differentiation
**************************************

.. note::
   This feature is still under development and may result in unexpected
   behavior. Please report any issues you find.

For forward mode AD, the restriction is that the function can be only be
differentiated with respect to a single input variable. However, in many cases,
it is desirable to differentiate a function with respect to multiple input
variables. One way to do this is to use a vectorized version of forward mode AD.

Without vector mode, for computing derivative of a function with n-dimensional
input - forward mode requires n forward passes, i.e. one for each input
variable. In vector mode, all these computations are batched together and
computed in a single forward pass and the function is differentiated with
respect to multiple input variables. This can help in reducing the overhead of
computing an expensive operation in multiple forward passes. This can also help
in utilizing the vectorization capabilities of the hardware.

The output of the function is a vector of partial derivatives with respect to
each input variable.

Currently, Clad only supports vectorized version of forward mode AD.

Asking Clad to differentiate using Vector mode
================================================

The following code snippet shows how one can request Clad to use vector mode for
differentiation:

.. literalinclude:: ../../../../test/Documentation/Guide/VectorMode.cpp
   :language: cpp
   :start-after: docs-begin-vector-mode
   :end-before: docs-end-vector-mode

Thus, the calling convention is to use
``clad::differentiate<clad::opts::vector_mode>(...)`` instead of the usual
calling convention, ``clad::differentiate(...)``.

Extent of support for Vector Mode within Clad
================================================

Currently only forward mode AD (using ``clad::differentiate``) has support for
vector mode with features being added incrementally for various cases. For more
ideas on the type of functions supported, please have a look at
``test/ForwardMode/VectorMode.C`` for examples that can be differentiated within
Clad.