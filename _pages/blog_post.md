---
title: "Blog post draft"
layout: gridlay
excerpt: "Blog post draft"
sitemap: false
permalink: /blog_draft/
---

<!-----
NEW: Check the "Suppress top comment" option to remove this info from the output.

Conversion time: 5.296 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β29
* Thu Oct 29 2020 05:48:47 GMT-0700 (PDT)
* Source doc: Copy of Cling -- Beyond just Interpreting C++
* Tables are currently converted to HTML tables.
* This document has images: check for >>>>>  gd2md-html alert:  inline image link in generated source and store images to your server. NOTE: Images in exported zip file from Google Docs may not appear in  the same order as they do in your doc. Please check the images!


WARNING:
You have 12 H1 headings. You may want to use the "H1 -> H2" option to demote all headings by one level.

----->

# Interactive C++ with Cling

The C++ programming language is used for many numerically intensive scientific applications. A combination of performance and solid backward compatibility has led to its use for many research software codes over the past 20 years. Despite its power, C++ is often seen as difficult to learn and inconsistent with rapid application development. Exploration and prototyping is slowed down by the long edit-compile-run cycles during development.

[Cling](https://github.com/root-project/cling/) has emerged as a recognized capability that enables interactivity, dynamic interoperability and rapid prototyping capabilities to C++ developers. Cling supports the full C++ feature set including the use of templates, lambdas, and virtual inheritance. Cling is an interactive C++ interpreter, built on top of the Clang and [LLVM](llvm.org) compiler infrastructure. The interpreter enables interactive exploration and makes the C++ language more welcoming for research.

The main tool for storage, research and visualization of scientific data in the field of high energy physics (HEP) is the specialized software package [ROOT](root.cern/). ROOT is a set of interconnected components that assist scientists from data storage and research to their visualization when published in a scientific paper. ROOT has played a significant role in scientific discoveries such as gravitational waves, the great cavity in the Pyramid of Cheops, the discovery of the Higgs boson by the Large Hadron Collider. For the last 5 years, Cling has helped to analyze 1 EB physical data, serving as a basis for over 1000 scientific publications, and supports software run across a distributed million CPU core computing facility. 

Recently we started a project aiming to leverage our experience in interactive C++, just-in-time compilation technology (JIT), dynamic optimizations, and large scale software development to greatly reduce the impedance mismatch between C++ and Python. We will generalize Cling to offer a robust, sustainable and omnidisciplinary solution for C++ language interoperability.The scope of our objectives is to:



*   advance the interpretative technology to provide a state-of-the-art C++ execution environment,
*   enable functionality which can provide native-like, dynamic runtime interoperability between C++ and Python (and eventually other languages such as Julia and Swift)
*   allow seamless utilization of heterogeneous hardware (such as hardware accelerators)

Project results will be integrated with and into the widely used tools LLVM/Clang. The outcome of the proposed work is a platform which provides a C++ compiler as a service (CaaS) for both rapid application development and computational performance.

The rest of this post intends to demonstrate the design and several features of Cling 


# Interpreting C++

Exploratory programming (or Rapid Application Development) is an effective way to gain understanding of the requirements for a project; to reduce the complexity of the problem; and to provide an early validation of the system design and implementation. In particular, interactively probing data and interfaces makes complex libraries and complex data more accessible users. It is important in data science, computational science and debugging. It significantly reduces the time consumed by edit-run cycles during development. In practice only few programming languages offer both a compiler and an interpreter translating them into machine code, although whether a language to be interpreted or compiled is a property of the implementation

Languages which enable exploratory programming tend to have interpreters which shorten the compile-link cycle, which generally has a noticeable cost  in performance. Language developers who acknowledge the use case of exploratory programming may also put syntactic sugar, but that is mostly for convenience and terseness. The performance penalty is largely mitigated by using just-in-time (JIT) or ahead-of-time (AOT) compilation technology.

For the sake of this post series, interpreting C++ means enabling exploratory programming for C++ while mitigating the performance cost with JIT compilation. Figure 1 shows an illustrative example of exploratory programming. It becomes trivial to orient the shape, choose size and color or compare to previous settings. The invisible compile-link cycle aids interactive use which allows some qualitatively different approaches to program development and enhanced productivity.

<p align="center">
<img src="/images/blog/figure1.gif" width="70%"><br />

<!--- ![alt_text](/images/blog/figure1.gif "image_tooltip") --->


Figure 1. Interactive OpenGL Demo. </p>


## Design principles



*   Do not pay for what you do not use -- do not penalize users typing syntactically and semantically correct C++. This makes the implementability of error recovery much harder. The interactive C++ transformations are only done when necessary and can be disabled. 
*   Reuse Clang & LLVM at (almost) any cost -- reuse what is available in LLVM, exorbitantly. If a feature is not available, then try finding a minimalistic way to implement it and propose it for a review to the llvm community. Otherwise find the minimal patch, even at the cost of misusing API, which satisfies the requirements.
*   Continuous feature delivery -- focus on a minimal feature, its integration in the main use-case (ROOT), deployment in production, repeat. 
*   Library design -- allow to be used as a library from third party frameworks.
*   Learn and evolve -- experiment with user experience. There is no formal specification or consensus on overall user experience. Apply lessons learned from the legacy from CINT.


## Architecture

Cling accepts partial input and ensures that the compiler remains active. It includes an API providing access to the properties of recently compiled chunks of code that can be post-processed. Cling orchestrates the existing LLVM and Clang infrastructure following a data flow described in Figure 2.

<p align="center">
<img src="/images/blog/figure2.png" width="70%"><br />

Figure 2. Information flow in Cling </p>

The tool controls the input infrastructure by interactive prompt or by an interface allowing the incremental processing of input (➀). Then it sends the input to the underlying clang library for compilation (➁). Clang compiles the input, possibly wrapped into a function, into an AST (➂). When necessary the AST can be further transformed in order to attach specific behavior (➃). For example, reporting execution results, or other interpreter-related features. Once the high-level AST representation is ready, it is sent for lowering to an LLVM-specific assembly format, the LLVM IR (➄). The LLVM IR is the input format for LLVM’s just-in-time compilation infrastructure. Cling instructs the JIT to run specified functions ( ➅ ), translating them into machine code (MC) targeting the underlying device architecture (eg. Intel x86 or NVPTX) (➆,➇).

The C++ standard is developed towards compilers and does not cover interactive use well. Execution of statements on the global scope, reporting execution results, and entity redefinitions are the three most important features when it comes to user friendliness. Long running interpreter sessions are prone to typing errors and make flawless error recovery essential. More advanced use-cases require extra flexibility at runtime and lookup rules extensions aiding eval-style programming. Efficient watermark-based code removal is important when C++ is used as scripting language.


## Execution of statements

Cling processes C++ incrementally. Incremental input consists of one or multiple C++ statements. C++ does not allow expressions in the global scope.


```
[cling] std::vector<int> v = {1,2,3,4,5}; v[0]++;
[cling] std::cout << "v[0]=" << v[0] <<"\n";
v[0]=2
```


Instead, Cling moves each input into a unique wrapper function. Eg:


```
void __unique_1 () { std::vector<int> v = {1,2,3,4,5};v[0]++;; } // #1
void __unique_2 () { std::cout << "v[0]=" << v[0] <<"\n";; } // #2
```


After the clang AST is built, cling detects that wrapper #1 contains a declaration and moves the declaration's AST node to the global scope, such that `v` can be referenced by subsequent inputs. Wrapper #2 contains a statement and is executed as is. After transformations the example becomes:


```
std::vector<int> v = {1,2,3,4,5};
void __unique_1 () { v[0]++;; }
void __unique_2 () { std::cout << "v[0]=" << v[0] <<"\n";; }
```



## Reporting execution results

An integral part of interactivity is printing expression values. Typing printf each time is laborious and does not naturally include object type information. Instead, omitting the semicolon of the last statement of the input tells Cling to report the expression result. When wrapping the input, Cling textually attaches a semicolon to the end of it. If an execution report is requested the corresponding wrapper AST does not contain a _NullStmt_ (modelling extra semicolons).


```
[cling] std::vector<int> v = {1,2,3,4,5} // Note the missing semicolon
(std::vector<int> &) { 1, 2, 3, 4, 5 }
```


A transformation injects extra code depending on the properties of the particular entity such as if it is copyable, if it is a wrapper temporary or an array. Cling can report information about non-copyable or temporary objects by providing a ‘managed’ storage. The managed storage (_cling::Value_) is also used for exchanging values between interpreted and compiled code in embedded setup.


## Entity Redefinition

Variable redefinition is an important scripting feature. It is also essential for notebook-based C++ as each cell is a somewhat separate computation. 


```
[cling]$ #include <string>
[cling]$ std::string v
(std::string &) ""
[cling]$ #include <vector>
[cling]$ std::vector<int> v
input_line_7:2:19: error: redefinition of 'v' with a different type: 'std::vector<int>' vs 'std::string' (aka 'basic_string<char, char_traits<char>,
  	allocator<char> >')
 std::vector<int> v
              	^
input_line_4:2:14: note: previous definition is here
 std::string v
         	 ^
```


C++ does not support redefinitions of entities. Cling implements the feature using inline namespaces and rewires clang lookup rules to give higher priority to more recent declarations. The full description of this feature was published as a conference paper on CC 2020 (ACM conference on Compiler Construction). You can find the publication [here](https://doi.org/10.1145/3377555.3377901).


```
[cling]$ #include "cling/Interpreter/Interpreter.h"
[cling]$ gCling->allowRedefinition()
[cling]$ #include <vector>
[cling]$ std::vector<int> v
(std::vector<int> &) {}
[cling]$ #include <string>
[cling]$ std::string v
(std::string &) ""
```



## Invalid Code. Error Recovery

When used in interactive mode, invalid C++ does not terminate the session. Instead invalid code is discarded. Underlying clang keeps the invalid AST nodes in its internal data structures for better error diagnostics and recovery, expecting the process will end shortly after issuing the diagnostics. This particular example is more challenging because it first contains both valid and invalid constructs. The error recovery should undo a significant amount of changes in internal structures such as the name lookup and the AST. Cling is used in many high-performance environments; using checkpointing is not a viable option as it introduces overhead for correct code.


```
[cling]$ std::vector<int> v; v[0].error_here;
```


In order to handle the example, Cling models the incremental input into a _Transaction_. A transaction represents the delta of the changes of internal data structures of clang. Cling listens to events coming from various Clang callbacks such as declaration creation, deserialization and macro definition. This information is sufficient to undo the changes and continue with a valid state. The implementation is very intricate and in many cases requires extra work depending on the input declaration kinds.

Cling also protects against null pointer dereferences via a code transformation, avoiding a session crash.


```
[cling]$ int *p = nullptr; *p
input_line_3:2:21: warning: null passed to a callee that requires a non-null argument [-Wnonnull]
 int *p = nullptr; *p
                	  ^
[cling]$
```



## Code Removal

Incremental, interactive C++ assumes long lived sessions where not only syntax error can happen but also semantic ones. That poses one level of extra complexity if we want to re-execute the same code with minor adjustments.


```
[cling]$ .L Adder.h // #1, similar to #include "Adder.h"
[cling]$ Add(3, 1) // int Add(int a, int b) {return a - b; }
(int) 2
[cling]$ .U Adder.h // reverts the state prior to #1
[cling]$ .L Adder.h
[cling]$ Add(3, 1) // int Add(int a, int b) {return a + b; }
(int) 4
```


In the example, we include a header file with the _.L_ meta command; “uninclude” it with _.U_ and “reinclude” it with _.L_ to re-read the modified file. Unlike in the error recovery case, Cling cannot fence the machine code lowering infrastructure and needs to undo state changes in clang CodeGen and the llvm JIT and machine code infrastructure. The implementation of this feature requires expertise in a big portion of the LLVM toolchain.


# Conclusion

Cling has been one of the systems enabling interactive C++ for more than decade. Cling’s extensibility and fast prototyping features is of fundamental importance for researchers in  high-energy physics, and an enabler for many of the technologies that they rely on. Cling has several unique features tailored to the challenges which comes with incremental C++. The implementation of error recovery and code unloading still has rough edges and it is being improved constantly.

In the next blog post we will focus on interactive C++ for Data Science; Eval-Style Programming; Interactive CUDA; and C++ in notebooks.

You can find out more about our activities at https://root.cern and https://compiler-research.org


# Interactive C++ for Data Science

In our previous blog post “Interactive C++ with Cling” we mentioned that exploratory programming is an effective way to reduce the complexity of the problem. In particular, interactively probing data and interfaces makes complex libraries and complex data more accessible users.

The main tool for storage, research and visualization of scientific data in the field of high energy physics (HEP) is the specialized software package [ROOT](http://root.cern/). ROOT is a set of interconnected components that assist scientists from data storage and research to their visualization when published in a scientific paper. ROOT has played a significant role in scientific discoveries such as gravitational waves, the great cavity in the Pyramid of Cheops, the discovery of the Higgs boson by the Large Hadron Collider. For the last 5 years, Cling has helped to analyze 1 EB physical data, serving as a basis for over 1000 scientific publications, and supports software run across a distributed million CPU core computing facility.

ROOT uses Cling as a reflection information service for data serialization. The C++ objects are stored in a binary format, vertically. The content of a loaded data file is made available to the users and C++ objects become a first class citizen.

FIXME: Make a smoother transition

[Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html) became the standard way for data analysts to explore ideas. The notebook extends the console-based approach to interactive computing in a qualitatively new direction, providing a web-based application suitable for capturing the whole computation process: developing, documenting, and executing code, as well as communicating the results.

This rest of this blog post intends to demonstrate Cling’s features at scale; projects related to Cling; and show interactive C++/CUDA.


## Eval-style Programming

A Cling object can access itself through its runtime. The example creates a `cling::Value` to store the execution result of the incremented variable _i_. That mechanism can be used further to support dynamic scopes extending the name lookup at runtime.


```
[cling]$ int i = 1;
[cling]$ cling::Value V;
[cling]$ gCling->evaluate("++i", V);
[cling]$ ++i
(int) 3
[cling]$ V
(cling::Value &) boxes [(int) 2]
```


We use this in HEP to make it easy to inspect and use C++ objects stored by ROOT. Cling enables ROOT to inject available object names into the name lookup when a file is opened:


```
[root] ntuple->GetTitle()
error: use of undeclared identifier 'ntuple'
[root] TFile::Open("tutorials/hsimple.root"); ntuple->GetTitle() // #1
(const char *) "Demo ntuple"
[root] gFile->ls();
TFile**   	 tutorials/hsimple.root    Demo ROOT file with histograms
 TFile*   	 tutorials/hsimple.root    Demo ROOT file with histograms
  OBJ: TH1F    hpx    This is the px distribution : 0 at: 0x7fadbb84e390
  OBJ: TNtuple    ntuple    Demo ntuple : 0 at: 0x7fadbb93a890
  KEY: TH1F    hpx;1    This is the px distribution
  [...]
  KEY: TNtuple    ntuple;1    Demo ntuple
[root] hpx->Draw()
```


Figure 3. Interactive plot of the _px_ distribution read from a root file.

The ROOT framework injects additional names to the name lookup on two stages. First, it builds an invalid AST by marking the occurrence of _ntuple_ (#1), then it is transformed into <code>gCling->EvaluateT&lt;/*<strong>return</strong> type*/<strong>void</strong>>("ntuple->GetTitle()", <em>/*context*/</em>); </code>On the next stage, at runtime, ROOT opens the file, reads its preambule and injects the names via the external name lookup facility in clang. The transformation becomes more complex if <em>ntuple->GetTitle()</em> takes arguments. 


# C++ in Notebooks

**_Author: Sylvain Corlay, QuantStack_**

The Jupyter Notebook technology allows users to create and share documents that contain live code, equations, visualizations and narrative text. It enables data scientists to easily exchange ideas or collaborate by sharing their analyses in a straight-forward and reproducible way. Language agnosticism is a key design principle for the Jupyter project, and the Jupyter frontend communicates with the kernel (the part of the infrastructure that runs the code) through a well-specified protocol. Kernels have been developed for dozens of programming languages, such as R, Julia, Python, Fortran (through the LLVM-based LFortran project).

Jupyter's official C++ kernel relies on [Xeus](https://github.com/jupyter-xeus/xeus), a C++ implementation of the kernel protocol, and Cling. An advantage of using a reference implementation for the kernel protocol is that a lot of features come for free, such as rich mime type display, interactive widgets, auto-complete, and much more.

Rich mime-type rendering for user-defined types can be specified by providing an overload of mime_bundle_repr for the said type, which is picked up by argument dependent lookup.




![alt_text](/images/blog/figure3.png "image_tooltip")
 \
Inline rendering of images in JupyterLab for a user-defined image type.

Possibilities with rich mime type rendering are endless, such as rich display of dataframes with HTML tables, or even mime types that are rendered in the front-end with JavaScript extensions. 

Xeus-cling comes along with an implementation of the Jupyter widgets protocol which makes an advanced use of such rich mime type rendering and enables bidirectional communication with the backend.


![alt_text](/images/blog/figure4.png "image_tooltip")


Interactive widgets in the JupyterLab with the C++ kernel.

More complex widget libraries have been enabled through this framework like [xleaflet](https://github.com/jupyter-xeus/xleaflet).



![alt_text](/images/blog/figure5.gif "image_tooltip")


Interactive GIS in C++ in JupyterLab with xleaflet

Other features include rich HTML help for the standard library and third-party packages:





![alt_text](/images/blog/figure6.gif "image_tooltip")
 \
Accessing cppreference for std::vector from JupyterLab by typing `?std::vector`

The Xeus and Xeus-cling kernels were recently incorporated as subprojects to Jupyter, and are governed by its code of conduct and general governance.

Planned future developments for the xeus-cling kernel include



*   Adding support for the Jupyter console interface, through an implementation of the Jupyter “is_complete” message, currently lacking.
*   Adding support for cling “dot commands” as Jupyter magics.
*   Supporting the new debugger protocol that was recently added to the Jupyter kernel protocol, which will enable the use of the JupyterLab visual debugger with the C++ kernel.

Another tool that brings interactive plotting features to xeus-cling is _xvega_. Xvega, which is at an early stage of development, produces vega charts that can be displayed in the notebook.





![alt_text](/images/blog/figure7.png "image_tooltip")
 \
The xvega plotting library in the xeus-cling kernel


# CUDA C++

**_Author: Simeon Ehrig, HZDR_**

The Cling CUDA extension brings the workflows of interactive C++ to GPUs without losing performance and compatibility to existing software. To execute CUDA C++ Code, Cling activates an extension in the compiler frontend to understand the CUDA C++ dialect and creates a second compiler instance that compiles the code for the GPU.





![alt_text](/images/blog/figure8.png "image_tooltip")


Like the normal C++ mode, the CUDA C++ mode uses AST transformation to enable interactive CUDA C++ or special features as the Cling print system. In contrast to the normal Cling compiler pipeline used for the host code, the device compiler pipeline does not use all the transformations of the host pipeline. Therefore, the device pipeline has some special transformation.


```
[cling] #include <iostream>
[cling] #include <cublas_v2.h>
[cling] #pragma cling(load "libcublas.so") // link a shared library
// set parameters
// allocate memory
// ...
[cling] __global__ void init(float *matrix, int size){
[cling] ?   int x = blockIdx.x * blockDim.x + threadIdx.x;
[cling] ?   if (x < size)
[cling] ? 	matrix[x] = x;
[cling] ?   }
[cling]
[cling] // launching a function direct in the global space
[cling] init<<<blocks, threads>>>(d_A, dim*dim);
[cling] init<<<blocks, threads>>>(d_B, dim*dim);
[cling]
[cling] cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, dim, d_B, dim, &beta, d_C, dim);
[cling] cublasGetVector(dim*dim, sizeof(h_C[0]), d_C, 1, h_C, 1);
[cling] cudaGetLastError()
(cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
```


Like the normal C++ mode, the CUDA mode can be used in a Jupyter Notebook.




![alt_text](/images/blog/figure9.png "image_tooltip")


A special property of Cling in CUDA mode is that the Cling application becomes a normal CUDA application at the time of the first CUDA API call. This enables the CUDA SDK with Cling. For example, you can use the CUDA profiler `nvprof ./cling -xcuda` to profile your interactive application.

Planned future developments for the CUDA mode include



*   Supporting of the complete current CUDA API
*   Redefining CUDA Kernels
*   Supporting other GPU SDK's like HIP (AMD) and SYCL (Intel)


# Conclusion

TODO:


# Cling -- Beyond just Interpreting C++


# Template Instantiation on Demand

Cling implements a facility, LookupHelper, takes a (perhaps qualified) C++ code and checks if a declaration with that qualified name already exists. For instance:


```
[cling] struct S{};
[cling] cling::LookupHelper& LH = gCling->getLookupHelper()
(cling::LookupHelper &) @0x7fcba3c0bfc0
[cling] auto D = LH.findScope("std::vector<S>",
				 cling::LookupHelper::DiagSetting::NoDiagnostics)
(const clang::Decl *) 0x1216bdcd8
[cling] D->getDeclKindName()
(const char *) "ClassTemplateSpecialization"
```


In the particular case, _findScope_ instantiates the template and returns its clang AST representation. Template instantiation on demand addresses the common library problem of template combinatorial explosion. Template instantiation on demand and conversion of textual qualified C++ names into entity meta information has proven to be a very powerful mechanism aiding data serialization and language interoperability.


# Language Interop on Demand

**_Author: Wim Lavrijsen, LBL_**

An example is cppyy ([https://cppyy.readthedocs.io/](https://cppyy.readthedocs.io/)), which provides automatic Python bindings, at run-time, to C++ code through Cling. Python is itself a dynamic language executed by an interpreter, thus making the interaction with C++ code more natural when intermediated by Cling. Examples include run-time template instantiations, function (pointer) callbacks, cross-language inheritance, automatic downcasting, and exception mapping. Many advanced C++ features such as placement new, multiple virtual inheritance, variadic templates, etc., etc., are naturally resolved by the LookupHelper.

cppyy achieves high performance through an all-lazy approach to run-time bindings construction and specializations of common cases through run-time reflection. As such, it has a much lower call overhead than e.g. pybind11, and looping over a std::vector through cppyy is faster than looping over a numpy array of the same type. Taking it a step further, its implementation for PyPy, a fully compatible Python interpreter sporting at tracing JIT (https://pypy.org), can in many cases provide native access to C++ code in PyPy’s JIT, including overload resolution and JIT hints that allow for aggressive optimizations.

Thanks to Cling’s run-time reflection, cppyy makes maintaining a large software stack simpler: except for cppyy's own python-interpreter binding, it does not have any compiled code that is Python-dependent. I.e., cppyy-based extension modules require no recompilation when switching Python versions (or even when switching between the CPython and PyPy interpreters, say).

cppyy is used in several large code bases in physics, chemistry, mathematics, and biology. It is readily installable through pip from PyPI ([https://pypi.org/project/cppyy/](https://pypi.org/project/cppyy/)) and through conda (https://anaconda.org/conda-forge/cppyy).


# Interpreter/Compiler as a Service

The design of Cling, just like Clang, allows it to be used as a library. In the next example we show how to incorporate libCling in a C++ program. Cling can be used on-demand, as a service, to compile, modify or describe C++ code. The example program shows several ways how compiled and interpreted C++ can interact:



*   _callCompiledFn_ --  The cling-demo.cpp defines an in global variable, _aGlobal_; a static float variable _anotherGlobal_; and its accessors. The _interp_ argument is an earlier created instance of the Cling interpreter. Just like in standard C++, it is sufficient to forward declare the compiled entities to the interpreter to be able to use them. Then the execution information from the different calls to _process_ is stored in a generic Cling _Value_ object which is used to exchange information between compiled and interpreted code.
*   _callInterpretedFn_ -- Complementing _callCompiledFn_, compiled code can call an interpreted function by asking Cling to form a function pointer from a given mangled name. Then the call uses the standard C++ syntax.
*   _modifyCompiledValue_ -- Cling has full understanding of C++ and thus we can support complex low-level operations on stack-allocated memory. In the example we ask the compiler for the memory address of the local variable _loc_ and ask the interpreter, at runtime, to square its value.

    ```
// cling-demo.cpp
// g++ ... cling-demo.cpp; ./cling-demo
#include <cling/Interpreter/Interpreter.h>
#include <cling/Interpreter/Value.h>
#include <cling/Utils/Casting.h>
#include <iostream>
#include <string>
#include <sstream>

/// Definitions of declarations injected also into cling.
/// NOTE: this could also stay in a header #included here and into cling, but
/// for the sake of simplicity we just redeclare them here.
int aGlobal = 42;
static float anotherGlobal = 3.141;
float getAnotherGlobal() { return anotherGlobal; }
void setAnotherGlobal(float val) { anotherGlobal = val; }

///\brief Call compiled functions from the interpreter.
void callCompiledFn(cling::Interpreter& interp) {
  // We could use a header, too...
  interp.declare("int aGlobal;\n"
                 "float getAnotherGlobal();\n"
                 "void setAnotherGlobal(float val);\n");

  cling::Value res; // Will hold the result of the expression evaluation.
  interp.process("aGlobal;", &res);
  std::cout << "aGlobal is " << res.getAs<long long>() << '\n';
  interp.process("getAnotherGlobal();", &res);
  std::cout << "getAnotherGlobal() returned " << res.getAs<float>() << '\n';

  setAnotherGlobal(1.); // We modify the compiled value,
  interp.process("getAnotherGlobal();", &res); // does the interpreter see it?
  std::cout << "getAnotherGlobal() returned " << res.getAs<float>() << '\n';

  // We modify using the interpreter, now the binary sees the new value.
  interp.process("setAnotherGlobal(7.777); getAnotherGlobal();");
  std::cout << "getAnotherGlobal() returned " << getAnotherGlobal() << '\n';
}

/// Call an interpreted function using its symbol address.
void callInterpretedFn(cling::Interpreter& interp) {
  // Declare a function to the interpreter. Make it extern "C" to remove
  // mangling from the game.
  interp.declare("extern \"C\" int plutification(int siss, int sat) "
             	"{ return siss * sat; }");
  void* addr = interp.getAddressOfGlobal("plutification");
  using func_t = int(int, int);
  func_t* pFunc = cling::utils::VoidToFunctionPtr<func_t*>(addr);
  std::cout << "7 * 8 = " << pFunc(7, 8) << '\n';
}

/// Pass a pointer into cling as a string.
void modifyCompiledValue(cling::Interpreter& interp) {
  int loc = 17; // The value that will be modified

  // Update the value of loc by passing it to the interpreter.
  std::ostringstream sstr;
  // on Windows, to prefix the hexadecimal value of a pointer with '0x',
  // one need to write: std::hex << std::showbase << (size_t)pointer
  sstr << "int& ref = *(int*)" << std::hex << std::showbase << (size_t)&loc << ';';
  sstr << "ref = ref * ref;";
  interp.process(sstr.str());
  std::cout << "The square of 17 is " << loc << '\n';
}

int main(int argc, const char* const* argv) {
  // Create the Interpreter. LLVMDIR is provided as -D during compilation.
  cling::Interpreter interp(argc, argv, LLVMDIR);

  callCompiledFn(interp);
  callInterpretedFn(interp);
  modifyCompiledValue(interp);

  return 0;
}
```



Output:


```
./cling-demo

aGlobal is 42
getAnotherGlobal() returned 3.141
getAnotherGlobal() returned 1
getAnotherGlobal() returned 7.777
7 * 8 = 56
The square of 17 is 289
```


Crossing the compiled-interpreted boundary relies on the stability of Clang’s application binary interface (ABI) logic. Over the years it has been very reliable for both Itanium and Windows however, Cling usually was among the first to discover ABI incompatibilities between GCC and Clang with respect to the Itanium ABI specification.


## Extensions

Just like Clang, Cling can be extended by plugins. The next example demonstrates embedded use of Cling’s extension for automatic differentiation, _Clad_. When creating the Cling instance we specify _-fplugin_ and the path to the plugin itself. Then we define a target function, _pow2_, and ask for its derivative with respect to its first argument.


```
#include <cling/Interpreter/Interpreter.h>
#include <cling/Interpreter/Value.h>

// Derivatives as a service.

void gimme_pow2dx(cling::Interpreter &interp) {
  // Definitions of declarations injected also into cling.
  interp.declare("double pow2(double x) { return x*x; }");
  interp.declare("#include <clad/Differentiator/Differentiator.h>");
  interp.declare("auto dfdx = clad::differentiate(pow2, 0);");

  cling::Value res; // Will hold the evaluation result.
  interp.process("dfdx.getFunctionPtr();", &res);

  using func_t = double(double);
  func_t* pFunc = res.getAs<func_t*>();
  printf("dfdx at 1 = %f\n", pFunc(1));
}

int main(int argc, const char* const* argv) {
 std::vector<const char*> argvExt(argv, argv+argc);
  argvExt.push_back("-fplugin=etc/cling/plugins/lib/clad.dylib");
  // Create cling. LLVMDIR is provided as -D during compilation.
  cling::Interpreter interp(argvExt.size(), &argvExt[0], LLVMDIR);
  gimme_pow2dx(interp);
  return 0;
}
```


Output:


```
./clad-demo
dfdx at 1 = 2.000000
```



# 


# C++ Modules in Interpretative Context

Initial design goal of C++ Modules feature is to enable scalable compilation for C/C++ code. As noted in the original ISOCPP [proposal](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2015/n4465.pdf), the feature targets improving the compilation model inherited from the C language. C compilation model introduces a notion of independent compilation. A program can consist of multiple independent compilations, _translation units_. Each translation unit is translated independently on the rest, with no knowledge of any details of the way it is used in the program. The communication between translation units is done via _name linkage_. A translation unit can reference, by name, entities defined elsewhere by qualifying them as _external_:


<table>
  <tr>
   <td><code><em>// A.cpp</em> \
<strong>int</strong> <strong>pow2</strong>(<strong>int</strong> x) { \
  <strong>return</strong> x * x; \
}</code>
   </td>
   <td><code><em>// B.cpp</em> \
<strong>extern</strong> <strong>int</strong> <strong>pow2</strong>(<strong>int</strong> x); \
<strong>int</strong> <strong>main</strong>() { \
  <strong>return</strong> pow2(42); \
}</code>
   </td>
  </tr>
</table>


A.cpp defines _pow2_ and B.cpp resolves _pow2_ via name linkage. 

The _linker_ resolves the communication problem between translation units. This brittle low level technology is the backbone of the C/C++ compilation model.

A common organization practice of C/C++ codebases is to declare names in _header files_. This can minimize the errors and give an illusion for uniform view of the declared entities in a program. From compilers' point of view, however, those files have to be textually expanded for each translation unit which includes them. This simple concept, served well for C/C++ for decades, has a few practical drawbacks. The textual expansion of invariant header files in the including translation unit causes significant increase of compile times and memory usage.

The linear compile times and memory usage is hardly a problem for the majority of the C++ codebases. It mostly results in longer build times and does not affect execution time. However, in interpretative C++ linear compile times affect performance and memory usage at runtime of the entire system. Efficient C++ modules implementation is essential for interpreted C++.

Auto Overlays

Performance comparison

TODO:


# In summary

Interactive C++ is popular in the data science domains but also gains popularity in other domains. Cling has been one of the systems enabling interactive C++ for more than decade. Cling’s extensibility and fast prototyping features is of fundamental importance for High Energy Physics and an enabler for many of the technologies it relies on.

Recently we started looking into opportunities to make big portions of the generic Cling-related infrastructure to the wider LLVM community.


# Acknowledgements

The author would like to thank David Lange, Axel Naumann, Sylvain Corlay, Wim Lavrijsen, Simeon Ehrig, Richard Smith, Alexander Penev, Martin Vassilev, Xavier Valls Pla who contributed to early versions of this post.
