//===--- Analyses.td - what clad's analyses look for ------*- tablegen -*-===//
//
// The one place clad's analyses are described: which analyses there are, what
// construct each one can act on, every way an input can miss that construct,
// the wording of every message they emit, and the documentation a reader gets
// sent to.
//
// Five files are rendered from it, none of them committed. The three headers
// are built before anything that expands them, into the build directory:
//
//   include/clad/Differentiator/Analyses.def   the analyses
//   lib/Differentiator/AnalysisDescs.def       the constructs and their misses
//   lib/Differentiator/Diagnostics.def         the messages
//
// and the two pages where the documentation is built:
//
//   docs/userDocs/source/user/Analyses.rst     the page reports link to
//   docs/userDocs/source/user/Diagnostics.rst  the messages, as a reference
//
//===----------------------------------------------------------------------===//

// An analysis clad can run. Every one of them is optional: it may change the
// code clad generates, never the values that code computes. With all of them
// off clad must still produce a correct derivative, only a more conservative
// one.
// To add one: write the def and name it in Clad's Analyses below. The next
// build renders the tables again, and the switches, the help screen, what a
// request carries and the reference page all come from the def.
class Analysis<string name, string legacy, bit onByDefault, string summary> {
  // How the analysis is named to users, as -fdisable-analysis takes it.
  string Name = name;
  // Suffix of the original -enable-<legacy>/-disable-<legacy> pair.
  string Legacy = legacy;
  // Whether it runs when no switch selects it.
  bit Default = onByDefault;
  // One line for -help.
  string Summary = summary;
}

def TBR : Analysis<"tbr", "tbr", 1,
                   "skips storing a value whose reverse sweep can recover it">;
def Varied : Analysis<"activity", "va", 0,
                      "skips the adjoint of a value no input can vary">;
def Useful : Analysis<"useful", "ua", 0,
                      "skips the adjoint of a value no output depends on">;
def Loop : Analysis<"loop", "loop", 1,
                    "proves what a function's counted loops allow: a trip "
                    "count the reverse sweep recomputes, and the range of a "
                    "pointer parameter the body writes">;

// One way an input can miss a construct. Detail says what the code does instead,
// in one short sentence about that code -- not about the analysis, which the
// reader does not have to know anything about to act on it. It is printed as
// the first note of a report and listed on the construct's documentation page.
class Miss<string detail> {
  string Detail = detail;
}

def IndexNotCounted : Miss<"no counted loop steps the index of this write">;
def IndexNotUnderstood : Miss<"the index is neither a constant nor a variable">;
def WritesDisagree : Miss<"two writes here cover ranges that are not one range">;
def BoundNotUsable : Miss<"the bound is neither a parameter nor a constant, so "
                          "a caller cannot work it out">;
def OpaqueWrite : Miss<"this writes through a pointer that cannot be traced "
                       "back to a parameter">;
def NoDefinition : Miss<"this function has no body in this file">;

def CondNotComparison : Miss<"the condition is not a comparison">;
def CondNotBelow : Miss<"the comparison is not < or <=">;
def IndexNotVariable : Miss<"the left side of the comparison is not a variable">;
def IndexNotInteger : Miss<"the index is not an integer">;
def NoStart : Miss<"the header gives the index no start value">;
def StepNotOne : Miss<"the increment does not step the index by one">;
def FunctionReturnsEarly : Miss<"the function can return before reaching this "
                                "loop">;
def BodyExitsEarly : Miss<"the body can leave the loop early">;
def BodyMovesIndex : Miss<"the body assigns the index too">;
def StartNotStable : Miss<"the start value is written elsewhere in the "
                          "function">;
def BoundNotStable : Miss<"the bound is written elsewhere in the function">;

def ElementIsNotANumber : Miss<"one element of this is not a number, and a "
                               "register holds no more than one">;
def SubscriptsDiffer : Miss<"this is read at more than one index in the loop">;
def ArrayIsWritten : Miss<"the loop writes this too">;
def ArrayIsLocal : Miss<"this is declared inside the loop, so each iteration "
                        "gets a fresh one and its adjoint starts again">;
def ArrayEscapes : Miss<"this is used somewhere other than as the base of a "
                        "subscript, so something else may reach its adjoint">;
def IndexIsWritten : Miss<"the index reads a variable the loop writes, so it "
                          "is not the same element every iteration">;

def CalleeHasNoBody : Miss<"this function has no body in this file, so there "
                           "is nothing to read to find out which arguments it "
                           "reads">;

// One message clad emits. The wording is user-facing: worth reviewing on its
// own, worth quoting in the documentation without drifting, and worth finding
// when someone asks what clad says about a subject. Text takes clang's %0
// placeholders, filled in at the site that emits it.
//
// Only the messages the analyses emit are here so far. The rest are still
// literals at their sites and move over as they are touched. A message gets
// no code when it moves: a CLAD number is the anchor of a documentation
// section, and it is the construct that has one to send a reader to, not the
// sentence announcing it. Clang numbers none of its own diagnostics either.
class Diagnostic<string severity, string text> {
  // Remark, Note, Warning or Error, spelled as clang::DiagnosticsEngine does.
  string Severity = severity;
  string Text = text;
}

// What an analysis left in the derivative, and why.
def remark_construct_cost : Diagnostic<"Remark", "%0">;
def remark_construct_cost_for : Diagnostic<"Remark", "'%0': %1">;
def note_construct_miss : Diagnostic<"Note", "%0">;
def note_construct_fix :
    Diagnostic<"Note", "to avoid this, make it %0 (CLAD%1)">;
def note_analysis_off :
    Diagnostic<"Note", "the %0 analysis is off (-fdisable-analysis=%0)">;

// A value the reverse sweep cannot recompute, so the forward sweep stores it.
def remark_value_kept :
    Diagnostic<"Remark", "clad keeps this value for the reverse sweep">;
def note_tbr_could_not_prove :
    Diagnostic<"Note", "to-be-recorded analysis could not show it unused">;
def note_tbr_disabled :
    Diagnostic<"Note", "to-be-recorded analysis is disabled">;
def note_derivative_requested :
    Diagnostic<"Note", "in the derivative of '%0' requested here">;
def note_kept_value_is :
    Diagnostic<"Note", "the value kept is the one this expression had">;

// A construct an analysis can act on.
class Desc<Analysis analysis, int number, string name, string cost> {
  // Which analysis acts on it.
  Analysis Owner = analysis;
  // The number a report prints for it, as CLAD1001 and up, and the anchor of
  // its section in the generated documentation. A code is never reused and never
  // changes: it is what a reader searches for and what a build log keeps.
  int Code = number;
  // How the construct is named to users, article included: a report reads it as
  // "make it <Name>".
  string Name = name;
  // What clad emits where the construct is missed, in the code's terms: the extra
  // work the reader is paying for. It is the report's first line.
  string Cost = cost;
  // Every way an input can miss it, in the order the page lists them.
  list<Miss> Misses = [];
  // The section of the documentation about it, in reStructuredText. What to
  // write for the construct, and why clad wants it, belong here, where there is
  // room for an example; a diagnostic has none. The ways to miss it are
  // appended to this from Misses.
  string Doc = "";
}

def CountedLoop : Desc<Loop, 1001, "a counted loop",
                        "clad adds a counter to this loop and increments it "
                        "every iteration"> {
  let Misses = [CondNotComparison, CondNotBelow, IndexNotVariable,
                IndexNotInteger, NoStart, StepNotOne, FunctionReturnsEarly,
                BodyExitsEarly, BodyMovesIndex, StartNotStable, BoundNotStable];
  let Doc = [{
An integer index stepped by one, from a start up to a bound, where nothing
else in the function writes the start, the bound or the index::

  for (int i = 0; i < n; i++)     // counted, as long as the body leaves n and i alone
    s += x[i] * y[i];

The reverse sweep of a loop has to run as many times as the loop did. For a
counted loop clad works that number out from the bounds, so the forward
sweep carries nothing. Otherwise it adds a counter, increments it every
iteration, and for a nested loop pushes each count onto a tape, which
allocates::

  int k = n;
  for (int i = 0; i < k; i++) {   // not counted: the body writes k
    s += x[i];
    k = k - 1;
  }

Rewriting the second loop to run to a bound nothing else touches is what
makes the counter go away.
}];
}

def BroadcastRead : Desc<Loop, 1002, "a broadcast read",
                          "clad adds to this adjoint in memory on every "
                          "iteration, which keeps the loop from vectorising"> {
  let Misses = [ElementIsNotANumber, SubscriptsDiffer, ArrayIsWritten,
                ArrayIsLocal, ArrayEscapes, IndexIsWritten];
  let Doc = [{
An element read on every iteration at an index the loop never moves, and
read nowhere else in the loop::

  for (int j = 0; j < n; j++)
    s += x[i] * w[j];             // x[i] is a broadcast read of the j loop

The adjoint of a broadcast read is a sum over the loop. Clad keeps that sum
in a register and adds it to memory once after the loop. Accumulated in
place instead, it is a store to an address that does not change, and that
one store is what stops the loop vectorising: LLVM reports it as a *write to
a loop invariant address*::

  for (int j = 0; j < n; j++)
    s += x[0] * x[j];             // x is read at two indices, so neither moves

A read the loop's own index moves is not a broadcast read at all, and clad
says nothing about it.
}];
}

def BoundedWrite : Desc<Loop, 1003, "a bounded write",
                         "clad watches every address written here instead of "
                         "copying one range"> {
  let Misses = [IndexNotCounted, IndexNotUnderstood, WritesDisagree,
                BoundNotUsable, OpaqueWrite, NoDefinition];
  let Doc = [{
A write through a pointer parameter whose range clad can express in that
function's own parameters::

  void subtract(int d, const double* x, const double* y, double* out) {
    for (int i = 0; i < d; i++)
      out[i] = x[i] - y[i];       // out is written over [0, d)
  }

A call site that knows the range copies it once before the call and restores
it after. Without it, clad falls back to watching every address the callee
writes, which costs a record per element and a scan per restore.

The range has to be one range, over an index some counted loop steps, up to
a bound the caller can work out: another parameter, or a constant.
}];
}

def TransparentCall : Desc<TBR, 1004, "a call clad can look inside",
                            "clad saves every argument of this call for the "
                            "reverse sweep"> {
  let Misses = [CalleeHasNoBody];
  let Doc = [{
A direct call to a function with a body in this file.

The to-be-recorded analysis decides which values the reverse sweep will read
back, so that the forward sweep saves only those. At a call it needs to know
which arguments the callee reads and which it writes, and it learns that by
analysing the callee. Where it cannot, every argument is saved.

Moving the definition into a header, or calling the function directly rather
than through a pointer, is what lets the analysis see it.
}];
}

// What is generated, and in which order.
def Clad {
  list<Analysis> Analyses = [TBR, Varied, Useful, Loop];
  list<Desc> Descs = [CountedLoop, BroadcastRead, BoundedWrite,
                      TransparentCall];
  list<Diagnostic> Diagnostics = [remark_construct_cost,
                                  remark_construct_cost_for,
                                  note_construct_miss, note_construct_fix,
                                  note_analysis_off, remark_value_kept,
                                  note_tbr_could_not_prove, note_tbr_disabled,
                                  note_derivative_requested,
                                  note_kept_value_is];
  string Title = "What the analyses look for";
  string Overview = [{
Clad generates a correct derivative for any code it can differentiate. How
fast that derivative is depends on what its analyses can prove about the
code, and each analysis proves it by recognising a *construct*: a pattern it
knows how to exploit. Code that fits a construct gets the cheaper derivative.
Code that does not still gets a correct one, built the conservative way.

This page lists the constructs, what each one buys, and what clad emits instead
when your code misses it. Each construct has a code, ``CLAD1001`` and up, which
its reports print and which never changes; one section here per code.

Seeing what clad declined
=========================

Ask an analysis to report what it looked for and did not find::

  clang -fplugin=clad.so -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=loop ...

The name is one of ``tbr``, ``activity``, ``useful`` and ``loop``, the same
names ``-fdisable-analysis=`` takes. Each report reads the same way: what
clad had to emit, what in the code forced it, and what to write instead.

.. code-block:: none

  gmm.cpp:53:3: remark: clad adds a counter to this loop and increments it every iteration
  gmm.cpp:53:23: note: the bound is written elsewhere in the function
  gmm.cpp:53:3: note: to avoid this, make it a counted loop (CLAD1001)

The caret of the first note is on what caused it, which is not always where
the cost lands. Above, the cost is the loop; the cause is its bound.

To check whether an analysis is behind a wrong derivative, turn it off with
``-fdisable-analysis=<name>``, or turn off all of them with
``-fdisable-analysis=all``. Every analysis is optional by construction: with
all of them off clad still computes the same values, only more slowly. If a
derivative changes when you disable one, that is a bug in clad, and a good
bug report.
}];
}
