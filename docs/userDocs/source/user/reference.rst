API reference
======================


.. cpp:class:: CladFunction

   Provides an interface to easily access, call and print the differentiated
   function.

   Every entry point below -- :cpp:func:`differentiate`, :cpp:func:`gradient`,
   :cpp:func:`hessian`, :cpp:func:`jacobian` and :cpp:func:`estimate_error` --
   returns one of these. It is a small wrapper around a pointer to the
   generated derivative, which Clad fills in while compiling the call.

   .. cpp:function:: template<class ...Args> return_type_t<F> execute(Args&&... args) const

      Calls the generated derivative. The arguments are the ones the original
      function takes, followed by the ones the derivative adds -- a pointer or
      reference per differentiated parameter in reverse mode, the result matrix
      in Hessian and Jacobian mode. Each mode's section above shows the shape.

      For the derivative of a member function, the object to call it on comes
      first, unless :cpp:func:`setObject` has already supplied one.

   .. cpp:function:: template<class ...Args> auto operator()(Args&&... args) const

      Same as :cpp:func:`execute`, so a ``CladFunction`` can be passed wherever
      a callable is expected.

   .. cpp:function:: const char* getCode() const

      Returns the source code of the generated derivative, which Clad stores in
      the object as a string literal while compiling.

   .. cpp:function:: void dump() const

      Prints :cpp:func:`getCode` to standard output.

   .. cpp:function:: CladFunctionType getFunctionPtr() const

      Returns a pointer to the generated derivative, for code that needs the
      plain function pointer rather than the wrapper.

   .. cpp:function:: void setObject(FunctorType* functor)
                     void setObject(FunctorType& functor)

      Remembers an object for the derivative of a member function or a functor,
      so later :cpp:func:`execute` calls need not pass one.

   .. cpp:function:: void clearObject()

      Forgets the object set by :cpp:func:`setObject`.

   .. cpp:function:: template<class ...Args> return_type_t<F> execute_kernel(dim3 grid, dim3 block, Args&&... args)

      Launches the derivative of a CUDA kernel with the given grid and block
      dimensions. Only available when compiling CUDA code, and the only way to
      call the derivative of a ``__global__`` function --
      :cpp:func:`execute` refuses it. See
      :doc:`Using Clad on CUDA code <UsingCladOnCUDACode>`.

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
   input parameters of the function.

   .. literalinclude:: ../../../../test/Documentation/Reference/Differentiate.cpp
      :language: cpp
      :start-after: docs-begin-differentiate
      :end-before: docs-end-differentiate

   .. cpp:function:: template<class Fn>\
                  CladFunction gradient(Fn fn, const char* args)

   In very brief, this function differentiate functions using the reverse mode
   automatic differentiation.

   More specifically, this function performs partial differentiation of the provided
   function (``fn``) using the reverse mode automatic differentiation with respect
   to all the parameters specified in ``args``.

   Please refer this to know more about the reverse mode automatic differentiation.
   For now it is enough to know that generally reverse mode AD is more efficient
   than the forward mode AD when there are multiple input parameters.

   .. literalinclude:: ../../../../test/Documentation/Reference/Gradient.cpp
      :language: cpp
      :start-after: docs-begin-gradient
      :end-before: docs-end-gradient

   .. cpp:function:: template<class Fn>\
                  CladFunction hessian(Fn fn, const char* args)

   This function generates a function that can be used to compute
   `hessian matrix <https://en.wikipedia.org/wiki/Hessian_matrix>`_
   of the provided function (``fn``) with respect to all the arguments
   specified in ``args``.

   .. literalinclude:: ../../../../test/Documentation/Reference/Hessian.cpp
      :language: cpp
      :start-after: docs-begin-hessian
      :end-before: docs-end-hessian

   .. cpp:function:: template<class Fn>\
                  CladFunction jacobian(Fn fn, const char* args)

   This function generates a function that can be used to compute
   `jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_
   of the provided function (``fn``) with respect to all
   the arguments specified in ``args``. If no explicit ``args`` argument is
   specified, then the jacobian is computed with respect to all the input
   parameters. The matrix has one row per element of the output and one column
   per independent scalar, counting the elements of the output array itself. For
   two scalar parameters and an output array of three elements that is 3 x 5.

    .. literalinclude:: ../../../../test/Documentation/Reference/Jacobian.cpp
       :language: cpp
       :start-after: docs-begin-jacobian
       :end-before: docs-end-jacobian

   .. cpp:function:: template<class Fn>\
                  CladFunction estimate_error(Fn fn, const char* args)

   This function generates a function that computes the gradient of ``fn`` and,
   along the way, an estimate of the floating-point error committed while
   evaluating it. The estimate comes from the same reverse sweep: the adjoint of
   a value says how strongly the result depends on it, so multiplying it by the
   rounding error of that value and summing over the program gives the error in
   the result.

   The generated function has the signature ``clad::gradient(fn)`` would
   produce, with one more parameter at the end, of type ``double&``, which
   receives the total estimated error.

   .. literalinclude:: ../../../../test/Documentation/Reference/EstimateError.cpp
      :language: cpp
      :start-after: docs-begin-estimate-error
      :end-before: docs-end-estimate-error

   By default the error of each value is estimated with a Taylor approximation
   model. A different model can be supplied instead; see
   :doc:`Floating point error estimation <FloatingPointErrorEstimation>`.

------------------

.. todo::

   Add the numerical differentiation API reference.
