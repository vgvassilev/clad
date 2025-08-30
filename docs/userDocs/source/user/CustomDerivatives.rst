Custom derivatives
*******************

Clad allows users to provide derivatives for functions. This feature, known as custom derivatives,
is useful in a variety of cases. This guide describes all that you need to know about custom
derivatives: what they are, why you should care about them, and how to use the functionality
to its fullest.

Let's get started.

.. contents::

What are custom derivatives?
=============================

Custom derivatives is a feature that lets users supply derivatives that
Clad can use during differentiation. We use the term *custom derivatives*
to refer to both this mechansim and to the user-provided derivatives themselves.

Custom derivatives are useful when a more efficient or numerically stable
expression for derivatives is known, or when Clad is unable to differentiate a function.
Clad is unable to differentiate a function if its definition is in a library and thus
source code is not available, or when the function code contains a C++ feature
that Clad does not support yet.

The custom derivative feature also enables hybrid AD approaches where Clad can work synergetically
with tools based on operator overloading, for example. Another use of custom derivatives is to
connect to third party libraries which need specific rules for differentiation such as
linear solvers. The mechanism is useful to connect differentiable code to code which does not
have differentiable properties, such as neural networks or other ML models.

Custom derivatives are defined as C++ functions in clad-specific namespaces. Whenever Clad
needs to differentiate a function, it will first look if a custom derivative for the function
is available. If so, Clad will use the custom derivative instead of differentiating the function
using AD.

Custom derivatives are of three flavours: `pushforward`_, `pullback`_ and
:code:`reverse_forw`. Each flavour has a distinct use-case. The :code:`pushforward`
custom derivatives are used by the Clad forward mode AD
(:code:`clad::differentiate`). The :code:`pullback` custom derivatives are used by the
Clad reverse mode AD (:code:`clad::gradient`). :code:`reverse_forw` is a weird custom derivative
type because it is not meant to differentiate anything. It is used by the Clad reverse mode AD
to determine the adjoint of a function's return value for functions which returns a reference or
a pointer type. This case will be explained in more detail later.

.. _pushforward: https://en.wikipedia.org/wiki/Pushforward_(differential)
.. _pullback: https://en.wikipedia.org/wiki/Pullback_(differential_geometry)


Clad internally automatically differentiates functions using these same flavours.
If Clad needs to differentiate a function :code:`fn` that has a custom derivative
:code:`fn_pushforward` defined, then it will use :code:`fn_pushforward` to compute
the derivative of :code:`fn`. Otherwise, Clad will attempt to automatically generate
:code:`fn_pushforward`. The externally observable behavior of :code:`fn_pushforward`
should be the same in both the cases. Put simply, :code:`fn_pushforward` should correctly compute
the derivative.

Clad does not make any efforts to ensure that a custom derivative has the correct
behavior. It is your responsibility to ensure that your custom derivatives are correct.


Where to define custom derivatives?
====================================

The custom derivatives for free functions needs to be defined under
:code:`clad::custom_derivatives` namespace and for class functions
(both static and non-static) under :code:`clad::custom_derivatives::class_functions`
namespace.

If a free function is defined in a namespace :code:`A::B::C`,
then the custom derivative for the function must be defined in the same namespace sequence
under :code:`clad::custom_derivatives`, that is, :code:`clad::custom_derivatives::A::B::C`.
The custom derivatives under :code:`clad::custom_derivatives::class_functions` do not
follow this rule. The custom derivatives for the class functions must all be defined
directly in :code:`clad::custom_derivatives::class_functions` regardless of the class's namespace.

.. note::

  Non-templated free functions defined in a header file need to be marked :code:`inline`
  to avoid issues with symbol duplication just like any other C++ entity defined in a header file.

Pushforward custom derivatives
===============================

The :ref:`pushforward <PushforwardFunctions>` custom derivative is used by the Clad forward mode AD.
Pushforward functions *pushes* sensitivities of the inputs to the sensitivities of the outputs.
Put simply, it computes partial derivative of function's output with respect to some independent
variable. This independent variable does not necessarily have to be the function's input.
This functionality can be easily understood with the help of an example, so let's set aside
the mathematics jargon.

Let's say we want to provide pushforward custom derivative for the function :code:`fn`::

  double fn(double u, double v) {
    return u * v;
  }

Then the pushforward custom derivative for the function :code:`fn` must compute the
partial derivative of the function's output with respect to the independent variable using the
values and the partial derivatives of the inputs. For example::

  u = x;
  v = 2 * x;
  y = fn(u, v);

If we are differentiating the above code with respect to :code:`x`, then the :code:`fn`
pushforward should compute the partial derivative of the :code:`fn`'s output (that is, :code:`y`)
with respect to :code:`x` using the values of :code:`u` and :code:`v` and
their partial derivatives with respect to :code:`x`.

More formally, the function pushforward should compute the directional derivative of
function output (:code:`y`) at the point :code:`{u, v}` in the direction of :code:`{du, dv}`.

The story does not end here. The :code:`pushforward` function is also required to compute the
primal value, that is, the result of the call :code:`fn(u, v)`. This is essential for the
forward mode AD to work correctly when a function take reference or pointer arguments.
It is also beneficial for generating more efficient code.

Now we are ready to see the :code:`pushforward` custom derivative of :code:`fn`::

  namespace clad {
  namespace custom_derivatives {

  clad::ValueAndPushforward<double, double>
  fn_pushforward(double u, double v, double du, double dv) {
    double y = fn(u, v); // compute the primal value
    double dy = v * du + u * dv; // compute the output derivative
    return {y, dy};
  }

  } // namespace custom_derivatives
  } // namespace clad

In the :code:`fn_pushforward` function, :code:`du` and :code:`dv` are :math:`\partial u / \partial x`
and :math:`\partial v / \partial x` respectively, where :code:`x` is the independent variable
with respect to which we are differentiating.

Some important things to note here:

- The :code:`pushforward` custom derivative function name must be :code:`<function_name>_pushforward`.

- The :code:`pushforward` custom derivative function must take the same number of arguments as the
  original function, followed by the partial derivatives of the inputs. The order of the arguments
  must be the same as in the original function.

- The :code:`pushforward` custom derivative function must return a
  :code:`clad::ValueAndPushforward` object. This object contains both the primal value
  and the output derivative.

Pullback custom derivatives
============================

The :ref:`pullback <PullbackFunctions>` custom derivative is used by the Clad reverse mode AD.
Pullback function *pulls* sensitivities of outputs to the sensitivities of inputs.
Put simply, it computes the contributions to the partial derivatives of some output with respect
to the function's inputs. This output variable does not necessarily have to be the function's output.
Let's take the same example as before to understand the pullback custom derivative::

  double fn(double u, double v) {
    return u * v;
  }

The pullback custom derivative for the function :code:`fn` must compute the contributions to the
partial derivatives of some output variable with respect to the function's input variables using the
output sensitivities. For example::

  r = fn(u, v);
  y = r;
  return y;

If :code:`y` is the final output of the code getting differentiated, then the
:code:`fn` pullback should compute the contributions to the partial derivatives of
:code:`y` with respect to :code:`u` and :code:`v`. Please note that the output variable is
:code:`y`, which is not the function's output.

Now we are ready to see the pullback custom derivative of :code:`fn`::

  namespace clad {
  namespace custom_derivatives {

  void fn_pullback(double u, double v, double dr, double *du, double *dv) {
    *du += v * dr;
    *dv += u * dr;
  }

  } // namespace custom_derivatives
  } // namespace clad

:code:`r` is the :code:`fn`s output and :code:`y` is the final output
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

  double &g(double &u, double &v) {
    if (u > v)
      return u;
    return v;
  }

  double fn(double u, double v) {
    double &r = g(u, v);
    return r;
  }

In the above example, the :code:`g(u, v)` output and :code:`double &r` refers to the
same variable, hence they should have the same adjoint. That is, if :code:`g(u, v)` returns
:code:`u`, then :code:`r` is an alias for :code:`u` and :code:`dr` must be an alias
for :code:`du`. However, there is no purely static analysis mechanism possible for Clad to
determine the return value of a function call because a function call result depends on the
runtime values. So the question becomes how to correctly set the adjoint :code:`dr` to either
:code:`du` or :code:`dv` in the derivative function?

Reverse-forward function is used to solve this problem. The reverse-forward function modifies
the primal function, :code:`fn` in our case, to return both the primal value and the adjoint.
With both the primal value and the adjoint being returned, Clad can correctly set both the :code:`r`
and :code:`dr`. Note that this method can work because the reverse-forward function computes the
adjoint variable at runtime instead of the compile-time.

Now we are ready to see the reverse-forward custom derivative of :code:`fn`::

  namespace clad {
  namespace custom_derivatives {

  clad::ValueAndAdjoint<double &, double &>
  fn_reverse_forw(double &u, double &v, double &du, double &dv) {
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

    double fn(double u, double v) {
      return u * val1 + v * val2;
    }
  };

  namespace clad {
  namespace custom_derivatives {
  namespace class_functions {
    // pushforward custom derivative
    clad::ValueAndPushforward<double, double>
    fn_pushforward(A *a, double u, double v, A *da, double du, double dv) {
      double y = a->fn(u, v); // compute the primal value
      // compute the derivative
      double dy = u * da->val1 + du * a->val1 + v * da->val2 + dv * a->val2;
      return {y, dy};
    }

    // pullback custom derivative
    void fn_pullback(A *a, double u, double v, double dr, A *da, double *du, double *dv) {
      *du += dr * a->val1;
      da->val1 += dr * u;
      *dv += dr * a->val2;
      da->val2 += dr * v;
    }
  } // namespace class_functions
  } // namespace custom_derivatives
  } // namespace clad

.. note::

   If :code:`fn` is a :code:`const` member function, then the the primal
   object is taken as a :code:`const` parameter. For example, the signature
   of pushforward and pullback will be as follows for :code:`fn(...) const`::

    // pushforward custom derivative
    clad::ValueAndPushforward<double, double>
    fn_pushforward(const A *a, double u, double v, A *da, double du, double dv);

    // pullback custom derivative
    void fn_pullback(const A *a, double u, double v, double dr, A *da, double *du, double *dv)

  Please note that the derivative object stays non-:code:`const`.

Constructor custom derivatives
=================================

Constructor custom derivatives are essential when we want to differentiate codes
involving class objects. Constructors are simlar to member functions, except that
they can initialize members. Initialization and assignment are very different things in C++.
Some types such as :code:`const`, reference types, ..., must be initialized. The
initialization aspect make the constructor differentiation a little more complex than
the good old member functions.

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
  derivative function code. Note that this requires that the class
  type must be move-constructible.

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

Constructor pullback custom derivative
----------------------------------------

Constructor pullback custom derivatives are more similar to the ordinary pullback
functions. Constructor pullback functions do not have the same problem as of constructor
pushforward functions of initializing the primal object and the derivative object. After all,
by the time the constructor pullback is called, both the primal object and the adjoint object
are already initialized. The initialization must be done in the forward-pass of the
reverse-mode AD, and thus the responsibility of this lies on :code:`constructor_reverse_forw`.

One important difference between a construct pullback  and an ordinary member function
pullback is that the member function pullback takes the associated class object as an argument,
whereas the constructor pullback does not. This is because the constructor pullback
does not have a need for the class object to compute the pullback. Think of it another way,
when the constructor is called, at that time the class object does not exist. Hence there is no
need of the class object to compute the derivative.

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
