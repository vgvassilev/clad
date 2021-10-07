Clad API reference
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

   .. todo::

      Add examples.

.. _api_reference_clad_gradient:

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

   .. todo::

      Add examples.

.. cpp:function:: template<class Fn>\
                  CladFunction hessian(Fn fn, const char* args)

   This function generates a function that can be used to compute
   `hessian matrix <https://en.wikipedia.org/wiki/Hessian_matrix>`_
   of the provided function (``fn``) with respect to all the arguments specified in
   ``args``.
  
   .. todo::

      Add examples.
  
.. cpp:function:: template<class Fn>\
                  CladFunction jacobian(Fn fn, const char* args)

   This function generates a function that can be used to compute
   `jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_ 
   of the provided function (``fn``) with respect to all
   the arguments specified in ``args``.

    .. todo::

       Add examples.

.. todo::

   Add numerical differentiation and error estimation framework API reference.