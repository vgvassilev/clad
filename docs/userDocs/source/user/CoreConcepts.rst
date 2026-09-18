Core concepts
*************

How Clad Works
=================

Clad is an open source clang plugin that enables automatic differentiation of
mathematical functions in C++. Clad which is based on the LLVM compiler
infrastructure does this by parsing and transforming the abstract syntax tree
(AST); it detects mathematical function and consequently inserts the derived
function in the AST.

Forward and Reverse Mode Automatic Differentiation
====================================================

Both modes propagate derivatives through the same elementary steps, in opposite
directions, and this page uses one notation for both. :math:`\dot{v}` is the
*tangent* of :math:`v`: its derivative with respect to the input being
differentiated. :math:`\bar{v}` is the *adjoint* of :math:`v`: the derivative
of the output with respect to :math:`v`. That is the standard notation of the
automatic differentiation literature; Griewank and Walther, listed under
:ref:`Further Reading <ad-further-reading>`, is the canonical reference for it.
Names in ``code font`` are C++ names -- the variables Clad generates, such as
``_d_x``, and the names in the example code -- not mathematical symbols.

Forward Mode Automatic Differentiation
----------------------------------------

In forward mode AD, one first fixes the independent variable with respect to
which differentiation is performed and computes the derivative of each
sub-expression recursively. Compared to reverse mode AD, forward
mode is natural and easy to implement as the flow of derivative information
coincides with the order of evaluation of the mathematical function.

Consider the equation

.. math::

   z = x y + \sin x

broken into the elementary steps a compiler sees:

.. math::

   a = x y \\
   b = \sin x \\
   z = a + b

Differentiating each step with respect to an arbitrary variable :math:`t` gives

.. math::

   \pdv{a}{t} = y \pdv{x}{t} + x \pdv{y}{t} \\
   \pdv{b}{t} = \cos x \pdv{x}{t} \\
   \pdv{z}{t} = \pdv{a}{t} + \pdv{b}{t}

which is the chain rule

.. math::

   \pdv{w}{t} = \pdv{w}{u_1} \pdv{u_1}{t} + \pdv{w}{u_2} \pdv{u_2}{t} + \cdots

applied step by step, with :math:`w` the output of a step and :math:`u_i` the
inputs it reads. In tangent notation the three steps read

.. math::

   \dot{a} = y \dot{x} + x \dot{y} \\
   \dot{b} = \cos x \dot{x} \\
   \dot{z} = \dot{a} + \dot{b}

Each step is a node of the computation graph, and a tangent travels with the
arrows, in the same order the program evaluates:

.. mermaid::

   flowchart TD
     X["x<br/>ẋ = 1"]
     Y["y<br/>ẏ = 0"]
     A["a = x y<br/>ȧ = y ẋ + x ẏ"]
     B["b = sin x<br/>ḃ = cos x ẋ"]
     Z["z = a + b<br/>ż = ȧ + ḃ"]

     X --> A
     Y --> A
     X --> B
     A --> Z
     B --> Z

Choosing :math:`t = x` means seeding :math:`\dot{x} = 1` and
:math:`\dot{y} = 0`, after which :math:`\dot{z}` holds :math:`\pdv{z}{x}`.
The derivative with respect to :math:`y` needs a second run with the seeds
swapped. That is the cost of forward mode: :math:`O(n)` sweeps for :math:`n`
inputs, which is the wrong way round for the gradient of a function with many
of them.


Reverse Mode Automatic Differentiation
----------------------------------------

In reverse mode AD, the dependent variable to be differentiated is fixed and the
derivative is computed with respect to each sub-expression recursively. Reverse
accumulation traverses the chain rule from outside to inside -- on the directed
acyclic graph of the function, from the output back to the inputs. A forward
pass evaluates the function and records what the reverse sweep will need; one
reverse sweep then gives every derivative. The recording is what reverse mode
pays for its speed: memory in proportion to the number of operations the
function executes, not to the size of its source.

Reverse mode reads the same chain rule in the other direction:

.. math::

   \pdv{s}{u} = \pdv{w_1}{u} \pdv{s}{w_1} + \pdv{w_2}{u} \pdv{s}{w_2}

Here :math:`u` is an input variable, :math:`w_i` are the step outputs that read
it directly, and :math:`s` is the final output being differentiated. The rule is
applied once per input rather than once per output. On the same three steps it
gives

.. math::

   \pdv{s}{b} = \pdv{s}{z} \\
   \pdv{s}{a} = \pdv{s}{z} \\
   \pdv{s}{y} = x \pdv{s}{a} \\
   \pdv{s}{x} = y \pdv{s}{a} + \cos x \pdv{s}{b}

or, in adjoint notation,

.. math::

   \bar{b} = \bar{z} \\
   \bar{a} = \bar{z} \\
   \bar{y} = x \bar{a} \\
   \bar{x} = y \bar{a} + \cos x \bar{b}

It is the same graph with the arrows turned round -- an adjoint travels against
the order of evaluation:

.. mermaid::

   flowchart BT
     Z["z<br/>z̄ = 1"]
     A["a<br/>ā = z̄"]
     B["b<br/>b̄ = z̄"]
     X["x<br/>x̄ = y ā + cos x b̄"]
     Y["y<br/>ȳ = x ā"]

     Z --> A
     Z --> B
     A --> X
     A --> Y
     B --> X

The sweep is seeded with :math:`\bar{z} = 1`, since :math:`s` is :math:`z`.

So one sweep gives the derivative with respect to every input at once. The cost
is the mirror of forward mode's: a second output needs a second sweep.

That order -- forward, then back -- survives into the generated code. For a
primal whose loop overwrites a value,

.. code-block:: cpp

   double f(double x) {
     double t = 1;
     for (int i = 0; i < 3; ++i)
       t *= x;
     return t;
   }

``clad::gradient(f)`` emits one function, ``f_grad``, that runs the primal
forward while recording what it will need, then walks back through it:

.. mermaid::

   flowchart TD
     SIG["void f_grad(double x, double *_d_x)<br/>the primal parameters, plus one pointer<br/>per differentiated parameter"]
     DECL["double _d_t = 0.;<br/>clad::tape&lt;double&gt; _t1;<br/>a zeroed adjoint per local, and the tapes"]
     FWD["forward sweep: the primal statements<br/>t *= x;"]
     TAPE["clad::push(_t1, t);<br/>only what to-be-recorded analysis<br/>says the reverse sweep reads"]
     SEED["_d_t += 1;<br/>the seed t̄ = 1"]
     REV["reverse sweep: the same statements, reversed<br/>t = clad::pop(_t1);<br/>_d_t += _r_d0 * x;"]
     OUT["*_d_x += t * _r_d0;<br/>x̄ accumulated into the caller's buffer"]

     SIG --> DECL --> FWD --> SEED --> REV --> OUT
     FWD -- "records" --> TAPE
     TAPE -- "restores" --> REV

Both sweeps are in the one function body, and the reverse sweep runs after the
forward one. ``_r_d0`` holds ``t``'s adjoint from before the assignment cleared
it, which is what lets the adjoints of ``t`` and ``x`` be updated from the same
step. What reaches the tape is decided by the to-be-recorded analysis rather
than by the chain rule: ``y = y + x * i`` in a loop stores nothing, because the
pullback of ``+`` reads neither operand, while ``y = y * x`` stores every
iteration's ``y``. Turning the analysis off with ``-fdisable-analysis=tbr``
stores every overwritten value.


Vectorized Forward Mode Automatic Differentiation
===================================================

Vectorized Forward Mode Automatic Differentiation is a computational technique
that combines two powerful concepts: vectorization and forward mode automatic
differentiation. This approach is used to efficiently compute derivatives of
functions with respect to multiple input variables by taking advantage of both
parallel processing capabilities and the structure of the computation graph.

Working
--------

For computing the gradient of a function with an :math:`n`-dimensional input,
forward mode requires :math:`n` forward passes.

Vector mode does it in a single forward pass. Instead of accumulating one
scalar derivative per node, it maintains a gradient vector at each node, so the
derivatives with respect to all three scalar inputs are carried together rather
than one pass at a time.

At each node, we maintain a vector, storing the complete gradient of that node's
output w.r.t.. all the input parameters. All operations are now vector operations,
for example, applying the sum rule will result in the addition of vectors.
Initialization for input nodes are done using one-hot vectors.

For :math:`f(x, y, z) = x + y + z` the three inputs are seeded with one-hot
vectors, the sum rule adds the vectors at the node, and the output carries the
whole gradient:

.. mermaid::

   flowchart TD
     X["x<br/>∇x = (1, 0, 0)"]
     Y["y<br/>∇y = (0, 1, 0)"]
     Z["z<br/>∇z = (0, 0, 1)"]
     ADD(("+"))
     F["f(x, y, z)<br/>∇f = ∇x + ∇y + ∇z = (1, 1, 1)"]

     X --> ADD
     Y --> ADD
     Z --> ADD
     ADD --> F

Benefits
----------

We know that each node requires computing a vector, which requires more memory
and more time, which adds to these memory allocation calls. This must be offset
by some improvement in computing efficiency.

This can prevent the recomputation of some expensive functions, which would have
executed in a non-vectorized version due to multiple forward passes. This approach
can take advantage of the hardware's vectorization and parallelization capabilities
using SIMD techniques.

.. _derived-function-types:

Derived Function Types and Derivative Types
=============================================

Each entry point asks for a different shape of derivative, so each generates a
function with its own name and signature. For ``double f(double x, double y)``
Clad produces:

``clad::differentiate(f, "x")``
  ``double f_darg0(double x, double y)`` -- the same signature as ``f``,
  returning the derivative instead of the value.

``clad::gradient(f)``
  ``void f_grad(double x, double y, double *_d_x, double *_d_y)`` -- one
  extra parameter per differentiated parameter. It returns nothing and writes
  through those pointers.

``clad::hessian(f)``
  ``void f_hessian(double x, double y, double *hessianMatrix)`` -- the matrix
  flattened in row major order, so :math:`n` independent variables need
  :math:`n^2` elements.

``clad::jacobian(g)``
  ``void g_jac(..., clad::matrix<double> *_d_vector_out)`` -- one matrix per
  pointer or array parameter that the function writes to.

``clad::estimate_error(f)``
  what ``clad::gradient`` would generate, with one more parameter at the end,
  ``double &_final_error``.

The derivative of a value has the same type as the value. A ``double`` has a
``double`` derivative, an array of three doubles has an array of three
doubles, and a user-defined type has a derivative of that same type. This is
why a gradient asks for a pointer to the parameter's own type rather than to
some separate derivative type, and why the derivative of an array parameter
whose length is only known at run time is passed as a ``clad::array_ref``:
Clad has to be told how far it may write.

Reverse mode accumulates into these parameters rather than assigning to them,
so the caller allocates them and sets them to zero. Calling a gradient twice
with the same buffer adds the second result to the first.

Custom Derivatives
====================

Clad differentiates source code, so it needs the body of every function it
reaches. Sometimes there is no body to reach -- the function is in a library,
or it is a compiler builtin -- and sometimes there is one but it should not be
differentiated: it may be an iterative approximation whose derivative is
better computed in closed form, or code whose derivative is known to be more
numerically stable when written by hand.

A custom derivative is how you tell Clad what the derivative of such a
function is. You write a function whose name is the original's with a suffix,
put it in ``clad::custom_derivatives``, and Clad calls it instead of
differentiating the body. Which suffix depends on what the call site needs: a
pushforward for forward mode, a pullback for reverse mode, and a
reverse-forward function for a call in reverse mode whose result is used
before the reverse sweep reaches it. The three are described below and in
:doc:`Custom derivatives <CustomDerivatives>`, which also covers member
functions and constructors.

The lookup is by name. A custom derivative whose signature does not match what
Clad expects is reported as an error naming the expected signature, but one
whose *name* is wrong is simply not found: Clad differentiates the function
itself and never mentions that your function exists.

Pushforward and Pullback functions
===================================

.. _PushforwardFunctions:

Pushforward functions
-------------------------

A pushforward computes the tangents of the outputs from the input values and
the tangents of the inputs.
Intuitively, pushforward functions propagates the derivatives forward. Pushforward
functions are constructed by applying the core principles of the forward mode
automatic differentiation.

As a user, you need to understand how pushforward function mechanism works so that you
can define custom derivative pushforward functions as require and can thus, unlock full
potential of Clad.

Mathematically, for

.. math::

   y = \operatorname{fn}(u) = \sin u

the pushforward is the rule

.. math::

   \dot{y} = \cos u \, \dot{u}

and ``fn_pushforward`` is the function Clad calls to apply it, given :math:`u`
and :math:`\dot{u}`. Here :math:`x` is the independent variable the derivative
is taken with respect to, so :math:`\dot{u} = \pdv{u}{x}`.

As a concrete example, Clad ships the pushforward of `std::sin` in
``clad/Differentiator/BuiltinDerivatives.h``::

  namespace clad {
  namespace custom_derivatives {
  namespace std {

  template <typename T, typename dT>
  ValueAndPushforward<T, dT> sin_pushforward(T x, dT d_x) {
    return {::std::sin(x), ::std::cos(x) * d_x};
  }

  }}}

A pushforward returns both the value and the derivative, in a
``clad::ValueAndPushforward``. That is what lets the derived function get both
from one call, instead of calling the original function a second time.

In the forward mode automatic differentiation, we need to compute derivative
of each expression in the original source program. Pushforward functions allow to
effectively compute and process derivatives of function call expressions. For example::

  y = fn(u, v);

In the derived function, this statement will be transformed to::

  clad::ValueAndPushforward<double, double> _t0 =
      fn_pushforward(u, v, _d_u, _d_v);
  _d_y = _t0.pushforward;
  y = _t0.value;

The generated names carry the tangents: ``_d_u`` holds :math:`\dot{u}`,
``_d_v`` holds :math:`\dot{v}`, and ``_d_y`` receives :math:`\dot{y}`, with
:math:`x` the independent variable.

From here onwards, in a pushforward or in a scalar forward-mode derived
function, ``_d_someVar`` holds the tangent of ``someVar``.

Pushforward functions are generated on demand. That is, if Clad needs to
differentiate a function call expression,
then only it will generate the pushforward of the corresponding function declaration.

For a function::

  double fn1(float i, double& j, long double k) { }

the prototype of the corresponding pushforward function will be as follows::

  clad::ValueAndPushforward<double, double>
  fn1_pushforward(float i, double& j, long double k,
                  float _d_i, double& _d_j, long double _d_k);

Please note the following specification of the pushforward functions:

- The name of the pushforward function is the name of the source function
  followed by ``_pushforward``.
- Return type of the pushforward function is ``clad::ValueAndPushforward<T, dT>``,
  where ``T`` is the return type of the source function and ``dT`` the type of
  its derivative. A function returning ``void`` has no value to pair with a
  derivative, so its pushforward returns ``void`` too.
- Parameter list of the pushforward function is the parameter list of the original function followed by
  a derivative parameter for each of those parameters, in the same order. Clad
  adds one for every parameter, including the ones no derivative flows through.

All of these specifications must be exactly satisfied when creating a custom
derivative pushforward function, and the name is the one to check first. Clad
looks a custom derivative up by name. If it finds one and the signature does
not match, it says so and names the signature it expected. If the name is
wrong, there is nothing to find and nothing to report: Clad differentiates the
function itself, and a misspelled custom derivative is simply never called.
:doc:`Custom derivatives <CustomDerivatives>` describes how to write them.

.. _PullbackFunctions:

Pullback functions
--------------------

A pullback is the reverse-mode counterpart of a pushforward. Where a
pushforward carries a derivative forward through a call, from the inputs to
the result, a pullback carries one backward, from the result to the inputs.

For a function ``double fn(double u, double v)`` the pullback is::

  void fn_pullback(double u, double v, double _d_y,
                   double *_d_u, double *_d_v);

It takes the original parameters, then the adjoint of the result -- how much
the final output depends on what this call returned -- then a pointer for each
parameter's adjoint. It computes nothing to return: it adds each parameter's
contribution into the adjoint it was handed.

Adding rather than assigning is the contract, not a detail. The same variable
can reach several calls, and each call owes it a share of the derivative; a
pullback that assigns silently discards the shares written before it. Clad
passes fresh zeroed temporaries for arguments taken by value, so the
difference does not show there, but for an argument taken by reference Clad
hands the pullback the caller's own adjoint, and assigning to it loses
whatever had accumulated.

A pullback does not return the primal value, because in reverse mode the value
was computed on the way in and is available already. A call whose result is
needed by the reverse sweep itself -- one returning a reference, for instance
-- is handled by a reverse-forward function instead, which returns the value
and its adjoint together as a ``clad::ValueAndAdjoint``.

Differentiable Class Types
==============================

Clad differentiates a user-defined type member by member. The derivative of an
object is another object of the same type, whose members hold the derivatives
of the corresponding members, which follows from the rule that a derivative
has the type of the value it belongs to. A gradient with respect to a
parameter of type ``Coordinates`` therefore takes a ``Coordinates *``, so a
caller passes ``&d_p`` and afterwards reads the derivative with respect to
``p.x`` as ``d_p.x``.

Two things need saying for this to work. The first is how an adjoint object
starts: it must be zero, and what zero means for a type is the type's own
business, so Clad value-initialises by default and lets a type say otherwise
through ``clad::zero_init`` or ``clad::zero_like``. The second is what a
constructor contributes, since a constructor is where a member first gets its
value from the arguments; Clad generates or looks up
``constructor_pushforward`` and ``constructor_pullback`` for that, and
:doc:`Custom derivatives <CustomDerivatives>` shows how to write them.

Members that are references or pointers are where this model is currently
weakest: a reference member and the variable it binds to are the same object,
so a derivative can reach it by two paths and be counted twice. See
`issue #2082 <https://github.com/vgvassilev/clad/issues/2082>`__.

.. _numerical-differentiation:

Numerical Differentiation
============================

Clad currently provides two interfaces packaged in a single template header file
that allows users to easily use numerical differentiation standalone. The two
interfaces and their usages are mentioned as follows:

* `forward_central_difference`

The numerical differentiation function that differentiates a multi-argument
function with respect to a single argument only. The position of the argument
is specified by the user or Clad. This interface is mainly used in Clad's
forward mode for call expressions with single arguments. However, it can also
easily be extended for jacobian-vector products. The signature of this
method is as follows::

  template < typename F, typename T, typename... Args>
    precision forward_central_difference(F f, T arg, std::size_t n, bool printErrors, Args&&... args){
        // Clad has enough type generality that it can accept
  	// functions with a variety of input types.
  	// Here:
  	// f(args...) - is the target function.
  	// n - is the position of the parameter with respect to which the derivative is calculated.
  	// printErrors - a flag to enable printing of error estimates.
  }

* `central_difference`

The numerical differentiation function that differentiates a multi-argument
function with respect to all the input arguments. This function returns the
partial derivative of the function with respect to every input, and as such
is used in Clad's reverse mode. The signature of the method is as follows::

  template <typename F, std::size_t... Ints, typename GradType,
              typename... Args>
    void central_difference(F f, GradType& _grad, bool printErrors, Args&&... args) {
  	// Similar to the above method, here:
  	// f(args...) - is the target function.
  	// grad - is a 2D data structure to store all our derivatives as grad[paramPosition][indexPosition]
  	// printErrors - a flag to enable printing of error estimates.
  }

The above uses functions from the standard math library and so is required
to link against the same. To avoid this (and disable numerical differentiation)
use `-DCLAD_NO_NUM_DIFF` at the target program's compile time.


Implementation Details
-------------------------

Clad uses the five-point stencil method to calculate numerical derivatives. Here,
the target function is executed at least 4 times for each input parameter. Since the
number of parameters can be different across multiple candidate functions, we use an
add-on function to correctly select the parameter whose derivative is to be calculated.
The function is described as follows::

  // This function enables 'selecting' the correct parameter to update.
  // Without this function, Clad will not be able to figure out which x should be updated to x ± h.
  template <typename T>
  T updateIndexParamValue(T arg, std::size_t idx, std::size_t currIdx, int multiplier, precision& h_val,...) {
      if (idx == currIdx) {
  	    // selects the correct ith term.
  	    // assigns it an h_val (h)
  	    // and returns arg + multiplier * h_val.
      }
      return arg;
    }

Here, Idx is the current parameter and currIdx is the parameter to differentiate with
respect to in that pass. If the indices do not match, the argument is returned unchanged.

This function is then applied to all the arguments and is forwarded to the target function `f`::

  fxh = f(updateIndexParamValue(args, indexSeq/*integer index sequence for the parameter pack,
  				Args allows giving an index to each parameter in the pack.*/,
  				i /*index to be differentiated wrt*/,
  				/*±1*/,
  				h/*this is returned*/,
  				/*other params omitted for brevity*/)...);

The above line results in the calculation of `f(..., xi ± h, ...)`. Finally the whole algorithm
for calculating the gradient of a function (numerically) is as follows::

  for each i in args, do:

    fx1 := f(updateIndexParamValue(args, idexSeq, i, 1, h, /*other params*/)...)

    fx2 := f(updateIndexParamValue(args, idexSeq, i, -1, h, /*other params*/)...)

    grad[i][0] := (fx1 - fx2)/(2 * h)

  end for


Currently Supported Use Cases
--------------------------------

* Differentiating multi-arg function calls.
* Differentiating calls with pointer/array input.
* Differentiating user-defined types.
* Printing of error estimates.

Error Estimation Core Concepts
================================

A floating-point computation is not the computation it was written as: every
operation rounds its result, and those roundings accumulate. Clad estimates how
much they add up to in the final result, and it does so with the machinery it
already has.

The link is the adjoint. The adjoint of a value is how much the result changes
when that value changes, which is exactly what a reverse sweep computes, so an
adjoint also says how much of the result's error comes from the error in that
value. Clad therefore generates the gradient of the function and, at each
assignment, adds the magnitude of the adjoint of the value written times the
rounding error of that value into a running total. What the error of a value is
taken to be is decided by an error model, which is a parameter of the framework:
the built-in one uses a Taylor approximation, and a program can supply its own.

Written out, the estimate is

.. math::

   E = \sum_{i=1}^{k} \left| \bar{v}_i \, v_i \, \varepsilon \right|

The sum runs over the :math:`k` writes the reverse sweep passes -- every
assignment, every declaration with an initialiser, every increment, the
returned expression, and one term per floating-point parameter at the end -- so
a variable written three times contributes three terms. :math:`v_i` is the value
the :math:`i`-th write leaves behind, :math:`\bar{v}_i` is its adjoint at that
point, and :math:`\varepsilon` is the model's relative rounding bound. Clad
accumulates the terms into the ``double&`` parameter it appends to the gradient.

It is a first-order estimate rather than a bound: the model linearises, so the
higher-order terms are dropped. Taking each term's magnitude keeps the
contributions from cancelling, which errs high. What it buys is that a
whole-program estimate costs about what a gradient costs, rather than a rerun of
the computation in higher precision.

:doc:`Floating-point error estimation <FloatingPointErrorEstimation>` describes
the framework, its classes and how to write a custom model;
:cpp:func:`estimate_error` in the :doc:`API reference <reference>` describes the
interface.

.. _ad-further-reading:

Further Reading
================

Clad implements the standard forward and reverse modes described in the
automatic differentiation literature. These are good places to read more about
the subject itself, independently of Clad:

* A. G. Baydin, B. A. Pearlmutter, A. A. Radul and J. M. Siskind,
  *Automatic Differentiation in Machine Learning: a Survey*, Journal of Machine
  Learning Research 18(153), 2018,
  `jmlr.org/papers/v18/17-468.html <https://jmlr.org/papers/v18/17-468.html>`__.
  A short survey, and the best starting point if you read only one.

* C. C. Margossian, *A Review of Automatic Differentiation and its Efficient
  Implementation*, WIREs Data Mining and Knowledge Discovery, 2019,
  `doi:10.1002/widm.1305 <https://doi.org/10.1002/widm.1305>`__. Covers what
  implementations actually have to do, which is closer to what Clad does.

* A. Griewank and A. Walther, *Evaluating Derivatives: Principles and
  Techniques of Algorithmic Differentiation*, 2nd edition, SIAM, 2008,
  `doi:10.1137/1.9780898717761 <https://doi.org/10.1137/1.9780898717761>`__.
  The standard reference, and where the theory behind checkpointing and
  to-be-recorded analysis is worked out.

* `autodiff.org <https://www.autodiff.org>`__, the community portal: a list of
  tools, a bibliography, and the workshops in the field.

* V. Vassilev, M. Vassilev, A. Penev, L. Moneta and V. Ilieva,
  *Clad -- Automatic Differentiation Using Clang and LLVM*, Journal of Physics:
  Conference Series 608, 012055, 2015,
  `doi:10.1088/1742-6596/608/1/012055 <https://doi.org/10.1088/1742-6596/608/1/012055>`__.
  How Clad itself is put together.
