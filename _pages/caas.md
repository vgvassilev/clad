---
title: "CaaS Project"
layout: textlay
excerpt: "CaaS Project"
sitemap: false
permalink: /caas/
---

# Incremental Compilation Support in Clang (CaaS) Project

### Project Goals
Incremental compilation aims to support clients that need to keep a single compiler instance active across multiple compile requests. Our work focuses on:
 * Enhancement – upstream incremental compilation extensions available in forks;
 * Generalization – make building tools using incremental compilation easier;
 * Sustainability – move incremental use cases upstream.

[Cling](/cling) currently requires around 100 patches to clang’s incremental compilation facilities.
For example, CodeGen was modified to work with multiple llvm::Module instances, and to finalize per each end-of-translation unit (cling has more than one).
Tweaks to the FileManager’s caching mechanism, and improvements to the SourceManager virtual and overridden files (code reached mostly from within cling’s setup) were necessary.
Our research shows that the clang infrastructure works amazingly well to support something which was not its main use case. The grand total of our diffs against clang-9 is: 62 files changed, 1294 insertions(+), 231 deletions(-). A major weakness of cling’s infrastructure is that it does not work with the clang Action infrastructure due to the lack of an IncrementalAction. An incremental action should enable the incremental compilation mode in clang (eg., in the preprocessor) and does not terminate at end of the main source file.

Our *Incremental Action* allows constant compilation of partial inputs and ensures that the compiler remains active. It includes an API to access attributes of recently compiled chunks of code that can be post-processed. The REPL orchestrates existing LLVM and Clang infrastructure with a similar data flow:

<img src="/images/caas_diagram1.png" width="100%">

The tool enabling incremental compilation (eg, Clang-Repl or cling) controls the input infrastructure by interactive prompt
or by an interface allowing the incremental processing of input (1). Then it sends the input to the underlying incremental
facilities in clang, for simplicity libIncremental, infrastructure (2). Clang compiles the input into an AST representation (3).
When required the AST can be further transformed in order to attach specific behavior (4).
The AST representation is then lowered to LLVM IR (5). The LLVM IR is the input format for LLVM’s just-in-time compilation
infrastructure. The tool will instruct the JIT to run specified functions ( 6 ), translating them into machine code
targeting the underlying device architecture (eg. Intel x86 or NVPTX). This embeddable design (7, 8) offers a compiler as a service (CaaS) capability.
A CaaS can support various language interoperability services. For example libInterOp can aid a Python program unable to resolve an entity
via last resort lookup request to the proposed layer (1). It performs a name lookup through for the requested entity (2). The REPL,
run as a service, finds a suitable candidate(s) and returns it. Then the layer wraps the candidate into a meta object and
returns to the Python interpreter as C++ entity bound to Python.

In the end we aim to enable a very interactive programming experience: 
<br />
<img src="/images/caas_prog_model.png" width="700">
