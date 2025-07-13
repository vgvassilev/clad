Custom derivatives
*******************

.. contents::

What are custom derivatives?
============================

Clad allows users to specify custom derivatives for functions. It is useful when a more
efficient or numerically stable expression for derivatives is known, or when Clad is unable
to differentiate a function. Clad is unable to differentiate a function if its definition is
in a library and source code is not available, or when the function code contains a C++ feature
that Clad does not support yet.

Custom derivatives are defined as C++ functions in clad-specific namespaces. Whenever Clad
needs to differentiate a function, it will first look if a custom derivative for the function
is available. If so, Clad will use the custom derivative instead of differntiating the function
using AD.

Clad supports 3 types of custom derivatives: :code:`pushforward`, :code:`pullback` and :code:`reverse_forw`.
Each type has a distinct use-case. Clad internally differentiates functions using the
same types as we have for custom derivatives. For example, if a Clad needs to differentiate
a function :code:`foo` that has a custom derivative :code:`foo_pushforward` defined, then it will use
:code:`foo_pushforward` to compute the derivative of :code:`foo`. Otherwise, Clad will try to automatically
generate :code:`foo_pushforward`. The outside observable behavior of :code:`foo_pushforward` should be the
same in both the cases. Clad does not make any efforts to ensure that the custom derivatives have the correct
behavior. It is the user's responsibility to ensure that the custom derivatives are correct.

You need to define :code:`pushforward` custom derivative if you want the derivative to be used
by the Clad forward mode AD (:code:`clad::differentiate`). The :code:`pullback` custom derivatives is
used by the Clad reverse mode AD (:code:`clad::gradient`). :code:`reverse_forw` is a weird custom derivative
type because it is not meant to differentiate anything. It is used by the Clad reverse mode AD
to determine the ajoint of a function's return value for functions which returns a reference or
a pointer type. This case will be explained in more detail later.

Where to define custom derivatives?
====================================

The custom derivatives for free functions needs to be defined under
:code:`clad::custom_derivatives` and for class functions (both static and non-static) under
:code:`clad::custom_derivatives::class_functions`.

If a free function is defined in a namespace :code:`A::B::C`, then the custom derivative for
the function must be defined in the same namespace sequence under :code:`clad::custom_derivatives`,
i.e, :code:`clad::custom_derivatives::A::B::C`. The custom derivatives under
:code:`clad::custom_derivatives::class_functions` do not follow this rule. The custom derivatives
for the class functions must all be defined directly in :code:`clad::custom_derivatives::class_functions`
regardless of the class's namespace.

Pushforward custom derivatives
===============================

The :ref:`pushforward <PushforwardFunctions>` custom derivative is used by the Clad forward mode AD.
Pushforward functions *pushes* sensitivities of the inputs to the sensitivities of the outputs.
Put simply, it computes partial derivative of function's output with respect to some independent
variable. This independent variable does not necessarily have to be the function's input.
This functionality can be easily understood with the help of an example, so let's set aside
the mathematics jargon.

Let's say we want to provide pushforward custom derivative for the function :code:`foo`::

  double foo(double u, double v) {
    return u * v;
  }

Then the pushforward custom derivative for the function :code:`foo` must compute the
partial derivative of the function's output with respect to the independent variable using the
values and the partial derivatives of the inputs. For example::

  u = x;
  v = 2 * x;
  y = foo(u, v);

If we are differentiating the above code with respect to :code:`x`, then the :code:`foo`
pushforward should compute the partial derivative of the :code:`foo`'s output (that is, :code:`y`)
with respect to :code:`x` using the values of :code:`u` and :code:`v` and
their partial derivatives with respect to :code:`x`.

The story does not end here. The :code:`pushforward` function is also required to compute the
primal value, that is, the result of the call :code:`foo(u, v)`. This is essential for the
forward mode AD to work correctly when a function take reference or pointer arguments.
It is also beneficial for generating more efficient code.

Now we are ready to see the :code:`pushforward` custom derivative of :code:`foo`::

  namespace clad {
  namespace custom_derivatives {

  clad::ValueAndPushforward<double, double>
  foo_pushforward(double u, double v, double du, double dv) {
    double y = foo(u, v); // compute the primal value
    double dy = v * du + u * dv; // compute the output derivative
    return {y, dy};
  }

  } // namespace custom_derivatives
  } // namespace clad

In the :code:`foo_pushforward` function, :code:`du` and :code:`dv` are :math:`\partial u / \partial x`
and :math:`\partial v / \partial x` respectively, where :code:`x` is the independent variable
with respect to which we are differentiating.

Some important things to note here:

- The :code:`pushforward` custom derivative function name must be :code:`<function_name>_pushforward`.

- The :code:`pushforward` custom derivative function must take the same number of arguments as the
  original function, followed by the partial derivatives of the inputs. The order of the arguments
  must be the same as in the original function.

- The :code:`pushforward` custom derivative function must return a
  :code:`clad::ValueAndPushforward` object. This object contains the primal value
  and the output derivative.

Pullback custom derivatives
============================

The :ref:`pullback <PullbackFunctions>` custom derivative is used by the Clad reverse mode AD.
Pullback function *pulls* sensitivities of outputs to the sensitivities of inputs.
Put simply, it computes the contributions to the partial derivatives of some output with respect
to the function's inputs. This output variable does not necessarily have to be the function's output.
Let's take the same example as before to understand the pullback custom derivative::

  double foo(double u, double v) {
    return u * v;
  }

The pullback custom derivative for the function :code:`foo` must compute the contributions to the
partial derivatives of some output variable with respect to the function's input variables using the
output sensitivities. For example::

  r = foo(u, v);
  y = r;
  return y;

If :code:`y` is the final output of the code getting differentiated, then the
:code:`foo` pullback should compute the contributions to the partial derivatives of
:code:`y` with respect to :code:`u` and :code:`v`. Please note that the output variable is
:code:`y`, which is not the function's output.

Now we are ready to see the pullback custom derivative of :code:`foo`::

  namespace clad {
  namespace custom_derivatives {

  void foo_pullback(double u, double v, double dr, double *du, double *dv) {
    *du += v * dr;
    *dv += u * dr;
  }

  } // namespace custom_derivatives
  } // namespace clad

:code:`r` is the :code:`foo`s output and :code:`y` is the final output
of the code getting differentiated. :code:`dr` is the partial derivative
of the output variable with respect to the function's output, that is,
:math:`\partial y/ \partial r`, and :code:`du` and :code:`dv` are the
partial derivatives :math:`\partial y / \partial u` and :math:`\partial y / \partial v`
respectively.

Some important things to note here:
- The pullback custom derivative function name must be :code:`<function_name>_pullback`.

- The pullback custom derivative function must take the same number of arguments as the
  original function, followed by the partial derivative of the function's output, which is
  then followed by the partial derivatives of the functions' arguments. The order of the
  arguments must be the same as in the original function.

Reverse-forward custom derivatives
====================================

This is an advance section. Please feel free to skip it if it is your first read of this document.

The reverse-forward custom derivative is used by the Clad reverse mode AD to determine
the adjoint of a function's return value for functions which returns a reference or a
pointer type. Adjoint of a variable :code:`u` is the partial derivative of the output variable
with respect to :code:`u`. Let's understand why reverse-forward functions are needed with
the help of an example::

  double &foo(double &u, double &v) {
    if (u > v)
      return u;
    return v;
  }

  double fn(double u, double v) {
    double &r = foo(u, v);
    return r;
  }

In the above example, the :code:`foo(u, v)` output and :code:`double &r` refers to the
same variable, hence they should have the same adjoint. That is, if :code:`foo(u, v)` returns
:code:`u`, then :code:`r` is an alias for :code:`u` and :code:`dr` must be an alias
for :code:`du`. However, there is no purely static analysis mechanism possible for Clad to
determine the return value of a function call because a function call result depends on the
runtime values. So the question becomes how to correctly set the adjoint :code:`dr` to either
:code:`du` or :code:`dv` in the derivative function?

Reverse-forward function is used to solve this problem. The reverse-forward function modifies
the primal function, :code:`foo` in our case, to return both the primal value and the adjoint.
With both the primal value and the adjoint being returned, Clad can correctly set both the :code:`r`
and :code:`dr`. Note that this method can work because the reverse-forward function computes the
adjoint variable at runtime instead of the compile-time.

Now we are ready to see the reverse-forward custom derivative of :code:`foo`::

  namespace clad {
  namespace custom_derivatives {

  clad::ValueAndAdjoint<double &, double &>
  foo_reverse_forw(double &u, double &v, double &du, double &dv) {
    if (u > v) {
      return {u, du}; // primal value and adjoint
    }
    return {v, dv}; // primal value and adjoint
  }

  } // namespace custom_derivatives
  } // namespace clad

Here :code:`du` and :code:`dv` are the adjoints of the function arguments.

Some important things to note here:
- The reverse-forward custom derivative function name must be :code:`<function_name>_reverse_forw`.

- The reverse-forward custom derivative function must take the same number of arguments as the
  original function, followed by the adjoints of the function's arguments. The adjoint of a
  function argument has the same type as the function argument after removing the :code:`const`
  qualifier.  The order of the arguments must be the same as in the original function.

- The reverse-forward custom derivative function must return a :code:`clad::ValueAndAdjoint`
  object. This object contains both the primal value and the adjoint.

Member functions custom derivatives
=====================================

Differentiating member functions is similar to differentiating free functions.
The only differences are:

- The member functions custom derivatives must be defined
  in :code:`clad::custom_derivatives::class_functions` namespace instead
  of :code:`clad::custom_derivatives<::namespace::sequence::of::free::function>`.

- The :code:`this` pointer must be accounted for in the custom derivative.

An example will make things clear::


  class A {
  public:
    // ...
    // ...

    double foo(double u, double v) {
      return u * val1 + v * val2;
    }
  };

  namespace clad {
  namespace custom_derivatives {
  namespace class_functions {
    // pushforward custom derivative
    clad::ValueAndPushforward<double, double>
    foo_pushforward(A *a, double u, double v, A *da, double du, double dv) {
      double y = a->foo(u, v); // compute the primal value
      // compute the derivative
      double dy = u * da->val1 + du * a->val1 + v * da->val2 + dv * a->val2;
      return {y, dy};
    }

    // pullback custom derivative
    void foo_pullback(A *a, double u, double v, double dr, A *da, double *du, double *dv) {
      *du += dr * a->val1;
      da->val1 += dr * u;
      *dv += dr * a->val2;
      da->val2 += dr * v;
    }
  } // namespace class_functions
  } // namespace custom_derivatives
  } // namespace clad

Constructor custom derivatives
=================================

Constructor custom derivatives are essential when we want to differentiate codes
involving class objects. Constructors are simlar to member functions, except that
they can initialize members. Initialization and assignment are very different things in C++.
Some types such as :code:`const`, reference types, ..., must be initialized. The
initialization aspect make the constructor differentiation a little more complex than
good old member functions.

Constructor pushforward custom derivative
------------------------------------------

Constructor pushforward functions differ from ordinary pushforward
functions in two important ways:

- Constructor pushforward functions initialize the primal class object
  and the corresponding derivative object. Ordinary member function
  pushforwards takes an already-existing primal class object and the
  corresponding derivative object as inputs.

- Constructor pushforward functions return a value even though
  constructor do not return anything. Constructor pushforward functions
  return initialized primal object and the derivative object. These are
  then used to initialize primal object and the derivative in the
  derivative function code.

Now let's see constructor pushforward custom derivative in-action::

  class Coordinates {
    public:
    Coordinates(double px, double py, double pz) :
      x(px), y(py), z(pz) {}

    public:
    double x, y, z;
  };

  namespace clad {
  namespace custom_derivatives {
  namespace class_functions {
  // custom constructor pushforward function
  clad::ValueAndPushforward<::Coordinates, ::Coordinates>
  constructor_pushforward(clad::ConstructorPushforwardTag<::Coordinates>, double x, double y,
                          double z, double d_x, double d_y, double d_z) {
    return {::Coordinates(x, y, z), ::Coordinates(d_x, d_y, d_z) };
  }
  } // namespace class_functions
  } // namespace custom_derivatives
  } // namespace clad

:code:`clad::ConstructorPushforwardTag<::Coordinates>` is used to identify the
class for which the constructor pushforward is defined. The member function
custom derivatives do not require this tag because the custom derivative function
takes the class object as the first argument, which is sufficient to identify
the class.

constructor pullback custom derivative
----------------------------------------

Constructor pullback custom derivatives are more similar to the ordinary pullback
functions. Constructor pullback functions do not have the same problem as of constructor
pushforward functions of initializing the primal object and the derivative object. After all,
by the time the constructor pullback is called, both the primal object and the adjoint object
are already initialized.

One important difference between a construct pullback  and an ordinary member function
pullback is that the member function pullback takes the associated class object as an argument,
whereas the constructor pullback does not. This is because the constructor pullback
does not have a need for the class object to compute the pullback. Think of it another way,
when the constructor is called, at that time the class object does not exist. Hence there is no
need the class object to compute the constructor pullback.

Let's see the constructor pullback custom derivative in-action using the
same :code:`Coordinates` class ::

  class Coordinates {
    Coordinates(double px, double py, double pz) :
      x(px), y(py), z(pz) {}

    public:
    double x, y, z;
  }

  namespace clad {
  namespace custom_derivatives {
  namespace class_functions {
  void constructor_pullback(double x, double y, double z, ::Coordinates *d_coordinates,
      double *d_x, double *d_y, double *d_z) {
    *d_x += d_coordinates->x;
    d_coordinates->x = 0;
    *d_y += d_coordinates->y;
    d_coordinates->y = 0;
    *d_z += d_coordinates->z;
    d_coordinates->z = 0;
  }
  } // namespace class_functions
  } // namespace custom_derivatives
  } // namespace clad

Note that the constructor pullback does not need anything such as
:code:`clad::ConstructorPushforwardTag<::Coordinates>`. It is because
the constructor pullback takes :code:`d_coordinates` as an argument, which can be
used to identify the class for which the constructor pullback is defined.
