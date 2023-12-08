Using Enzyme within Clad
*************************

Like Clad, `Enzyme <https://enzyme.mit.edu/>`_ is also a library for Automatic
Differentiation(AD). A major difference is that, Enzyme works at the LLVM
Intermediate Representation(IR) level, while Clad works at the Clang Abstract
Syntax Tree(AST) level. Clad analyses the Clang AST to generate the derivative
of a function, while Enzyme analyses the LLVM IR of a function to generate the
derivative of the same.

Languages such as Rust, C++, Julia and Swift output LLVM IR which makes Enzyme
more interoperable as it can differentiate functions originally written in
multiple languages. However, in Enzyme, interoperability comes at a cost. For
example, while basic functions are supported across various languages;
containers, classes and other language specific constructs require extra
scaffolding to be supported by Enzyme. Clad is more tightly coupled with the
Clang frontend and the C++ language. It has access to the high-level program
structure and compile-only constructs such as `consteval`. That allows more
coherency when differentiating C++. This includes first class support of Object
Oriented Programs written in C++.

The initial integration of Enzyme in Clad is to enable cross validation of the
produced code.

Currently, use of Enzyme for Reverse Mode AD is supported within Clad. This
section describes how Enzyme can be used within Clad.


Configuring Clad to use Enzyme
=================================
To enable the use of enzyme within Clad, one needs to configure Clad to use
Enzyme. This can be done by adding the flag ``-DCLAD_ENABLE_ENZYME_BACKEND=On``
to cmake while configuring Clad build. Thus the overall cmake command should
look something like this

.. code-block:: bash

   cmake ../clad -DClang_DIR=/usr/lib/llvm-11 -DLLVM_DIR=/usr/lib/llvm-11
    -DCMAKE_INSTALL_PREFIX=../inst -DLLVM_EXTERNAL_LIT="``which lit``"
    -DCLAD_ENABLE_ENZYME_BACKEND=On

This flag instructs the build system to download and build Enzyme. Then it is
linked as a static library to Clad.

Asking Clad to generate gradients with Enzyme
================================================

The following code snippet shows how one can request Clad to generate gradients
with Enzyme::

    #include "clad/Differentiator/Differentiator.h"

    double array_product(double* arr) { return arr[0] * arr[1]; }

    int main(){
        auto grad = clad::gradient<clad::opts::use_enzyme>(array_product);
        double v[2] = {3, 4};
        double g[2] = {0};
        grad.execute(v, g);
        printf("d_x = %.2f, d_y = %.2f\n", g[0], g[1]);
    }

Thus, the calling convention is to use
``clad::gradient<clad::opts::use_enzyme>(...)`` instead of the usual calling
convention, ``clad::gradient(...)``. Calling ``execute``, ``dump`` and other
functionalities remain same as that of Clad.

Extent of support for Enzyme within Clad
=========================================

Currently functions that take in arrays, pointers and primitive types(integer
and real) as parameters are supported for differentiation with Enzyme within
Clad. For more ideas on the type of functions supported, please have a look at
``test/Enzyme/ReverseMode.C`` for examples that can be differentiated within
Clad.
