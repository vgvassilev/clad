Immediate mode
**************

The derivatives that Clad generates are valid C++ code, which could in theory
be executed at compile-time (or in an immediate context as the C++ standard
calls it). When a function is differentiated all specifiers, such as
`constexpr` and `consteval` are kept, so the derivative of a `constexpr`
function is itself `constexpr`.

Getting the derivative in time is the harder half. Clad normally waits until
the whole translation unit is parsed before generating anything, which is too
late for a compiler that is already working out the value of a call. So when
the call to `clad::differentiate` sits in the body of a `constexpr` or
`consteval` function, Clad plans and generates that derivative as soon as the
declaration carrying it arrives. There is nothing to ask for: Clad reads where
the call is and decides.

Usage of Clad's immediate mode
================================================

The following code snippet differentiates a function and evaluates the
derivative while the program compiles:

.. literalinclude:: ../../../../test/Documentation/Guide/ImmediateMode.cpp
   :language: cpp
   :start-after: docs-begin-immediate-mode
   :end-before: docs-end-immediate-mode

The example needs Clang 17 or later: the early-generation path in the plugin
is compiled only for those, and on an older Clang no derivative is generated
for a `constexpr` function at all.

Both the call to `clad::differentiate` and all its `.execute(...)` calls have
to stay in the same immediate context, as the C++ standard forbids holding a
function pointer to an immediate function outside of one. (It is not possible
to do the differentiation and the executions in `main`, as `dx` would hold
such a pointer while `main` is not and cannot be immediate.)

No variable the language requires to be initialised by a constant expression
can hold the call, for a different reason. Clad puts the derivative into a call
by rewriting it, and is handed a declaration only once the compiler has
finished with it, so such an initialiser is worked out before the derivative is
there. That covers a `constexpr` or `constinit` variable at namespace scope and
a `constexpr` local, even one inside a `constexpr` function. Clad says so, and
suggests writing it the way the example above does: call Clad from a
`constexpr` function, keep the result in an ordinary variable there, and
evaluate that function where the constant is wanted. An ordinary variable is
unaffected, because its value is worked out again after the rewrite.

When using `constexpr` there is no easy way to tell whether the functions are
actually being evaluated during translation, so it is a good idea to use
either `consteval` or an `if consteval` (in C++23 and newer) to check that the
immediate contexts behave as expected, or to assign the result to a variable
marked `constexpr`, which fails if the expression being assigned is not a
constant one.

Use cases supported by Clad's immediate mode
================================================

Forward mode (`clad::differentiate`) works throughout. Reverse mode
(`clad::gradient`) needs C++26: Clad calls the generated gradient through a
pointer whose adjoint parameters are ``void*`` while the function itself takes
a pointer to each argument's own type, and a cast from ``void*`` only became a
constant expression in C++26. Loops are a separate matter -- the tape Clad
uses to reverse one is not usable while the program compiles.

Both `constexpr` and `consteval` are supported as Clad doesn't actually rely on
these specific keywords for its support, but instead uses clang's API to
determine if the functions are immediate and should be differentiated eariler.
