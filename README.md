1. Description
In mathematics and computer algebra, automatic differentiation (AD) is a set of 
techniques to numerically evaluate the derivative of a function specified by a 
computer program. Automatic differentiation is an alternative technique to 
Symbolic differentiation and Numerical differentiation (the method of finite 
differences) that yields exact derivatives even of complicated functions.

The goal of the presented plugin is to extend the Cling functionality in order 
to make it possible for the tool to differentiate non-trivial functions and 
find partial derivatives for trivial cases. Our implementation approach is to 
employ source code transformation, which consists of explicitly building a 
new source code through a compiler-like process that includes parsing the 
original program, constructing an internal representation, and performing 
global analysis. This elegant but laborious process is greatly aided by 
Cling (http://cern.ch/cling) which does not only provide the necessary facilities
 for code transformation, but also serves as a basis for the plugin.


This is an example of how clang's AST can be used to synthesize automatically
derivatives of arbitrary C/C++ functions.

Build the plugin by running `make` in this directory.

Once the plugin is built, you can run it using:
--
Linux:
$ clang -cc1 -load ../../Debug+Asserts/lib/libAutoDiff.so -plugin print-fns some-input-file.c
$ clang -cc1 -load ../../Debug+Asserts/lib/libAutoDiff.so -plugin print-fns -plugin-arg-print-fns help -plugin-arg-print-fns --example-argument some-input-file.c
$ clang -cc1 -load ../../Debug+Asserts/lib/libAutoDiff.so -plugin print-fns -plugin-arg-print-fns -an-error some-input-file.c

Mac:
$ clang -cc1 -load ../../Debug+Asserts/lib/libAutoDiff.dylib -plugin print-fns some-input-file.c
$ clang -cc1 -load ../../Debug+Asserts/lib/libAutoDiff.dylib -plugin print-fns -plugin-arg-print-fns help -plugin-arg-print-fns --example-argument some-input-file.c
$ clang -cc1 -load ../../Debug+Asserts/lib/libAutoDiff.dylib -plugin print-fns -plugin-arg-print-fns -an-error some-input-file.c
