//===--- Options.td - Options clad takes ---------------------*- tablegen -*-===//
//
// Every option clad accepts, written down once: its spelling, what -help says
// about it, and which field of clad::Options it sets. The header clad compiles
// against, the help screen and the reference page are all rendered from here.
//
// The grammar is LLVM's, the same one clang's Options.td is written in, so an
// entry reads the way a clang option does. What clad does not reuse is
// llvm::opt::OptTable at run time: the table it generates has changed contract
// four times across the LLVM releases clad supports, while the records below
// are read identically by every llvm-tblgen from 14 to 22.
//
// The one pair not written here is -enable-<x>/-disable-<x>, which is a
// switch per analysis rather than an option in its own right and is spelled
// where the analyses themselves are.
//
//===----------------------------------------------------------------------===//

include "llvm/Option/OptParser.td"

// Where a marshalled option writes. clad's options are a flat struct, so
// there is no key-path prefix, and no -cc1 database to name.
class CladOpt<string base> : KeyPathAndMacro<"", base, ""> {}

// Prose for the reference page and for the field's doc comment, where one line
// of help is not enough. Spelled as clang spells it.
class DocBrief<code text> { code DocBrief = text; }

// A directory that is absent when the option is not given and may be empty
// when it is. LLVM ships no marshalling class for an optional, and the
// difference is what the option means.
class MarshallingInfoOptionalString<KeyPathAndMacro kpm>
  : MarshallingInfo<kpm, "std::nullopt"> {
  code Normalizer = "normalizeOptionalString";
}

//===----------------------------------------------------------------------===//
// Showing what clad read and what it produced
//===----------------------------------------------------------------------===//

def fdump_source_fn : Flag<["-"], "fdump-source-fn">,
  HelpText<"Prints out the source code of the function.">,
  MarshallingInfoFlag<CladOpt<"DumpSourceFn">>;

def fdump_source_fn_ast : Flag<["-"], "fdump-source-fn-ast">,
  HelpText<"Prints out the AST of the function.">,
  MarshallingInfoFlag<CladOpt<"DumpSourceFnAST">>;

def fdump_derived_fn : Flag<["-"], "fdump-derived-fn">,
  HelpText<"Prints out the source code of the derivative.">,
  MarshallingInfoFlag<CladOpt<"DumpDerivedFn">>;

def fdump_derived_fn_ast : Flag<["-"], "fdump-derived-fn-ast">,
  HelpText<"Prints out the AST of the derivative.">,
  MarshallingInfoFlag<CladOpt<"DumpDerivedAST">>;

def fdump_generated_source : Flag<["-"], "fdump-generated-source">,
  HelpText<"Prints the text clad renders each derivative as, and where every "
           "statement lands in it.">,
  DocBrief<[{For each derivative, prints the text clad renders it as and, for
every statement in it, the line and column that text occupies. Diagnostics
about generated code point into that rendering.}]>,
  MarshallingInfoFlag<CladOpt<"DumpGeneratedSource">>;

//===----------------------------------------------------------------------===//
// Putting the generated code where a debugger can find it
//===----------------------------------------------------------------------===//

def fgenerated_source_dir_EQ : Joined<["-"], "fgenerated-source-dir=">,
  MetaVarName<"<dir>">,
  HelpText<"Writes the code clad generates into <dir> and names it in the "
           "debug line table.">,
  DocBrief<[{Writes the code clad generates into <dir>, one file per translation
unit, and names it in the debug line table. A debugger that cannot read source
embedded in the object -- gdb cannot -- opens that file instead. Given with no
directory, nothing is written and clad stops advising that nothing can be
read.}]>,
  MarshallingInfoOptionalString<CladOpt<"GeneratedSourceDir">>;

def fgenerate_source_file : Flag<["-"], "fgenerate-source-file">,
  HelpText<"Produces a file containing the derivatives.">,
  MarshallingInfoFlag<CladOpt<"GenerateSourceFile">>;

//===----------------------------------------------------------------------===//
// The rest
//===----------------------------------------------------------------------===//

def fno_validate_clang_version : Flag<["-"], "fno-validate-clang-version">,
  HelpText<"Disables the validation of the clang version.">,
  MarshallingInfoNegativeFlag<CladOpt<"ValidateClangVersion">>;

def fprint_num_diff_errors : Flag<["-"], "fprint-num-diff-errors">,
  HelpText<"Prints the calculated numerical diff errors; overridden by "
           "-DCLAD_NO_NUM_DIFF.">,
  MarshallingInfoFlag<CladOpt<"PrintNumDiffErrorInfo">>;

def fclad_porting_hints : Flag<["-"], "fclad-porting-hints">,
  HelpText<"Names the custom derivative clad would have used, where it fell "
           "back to differentiating a definition.">,
  DocBrief<[{When clad has no custom derivative for a function defined outside
the main source file and falls back to differentiating its definition, emit a
remark naming the expected custom-derivative signature and the
non-differentiable marker. Useful when teaching clad about a new library.}]>,
  MarshallingInfoFlag<CladOpt<"EmitPortingHints">>;

//===----------------------------------------------------------------------===//
// Options that name an analysis
//
// One option taking a name, not an option per analysis: clad has several and
// will grow more. Which names they take is settled where the analyses are,
// no field of their own.
//===----------------------------------------------------------------------===//

def fenable_analysis_EQ : Joined<["-"], "fenable-analysis=">,
  MetaVarName<"<name>">,
  HelpText<"Turns the named analysis on; the last such option decides.">;

def fdisable_analysis_EQ : Joined<["-"], "fdisable-analysis=">,
  MetaVarName<"<name>">,
  HelpText<"Turns the named analysis off; 'all' turns off every one.">,
  DocBrief<[{Turns the named analysis off; the last such option on the command
line decides. -fdisable-analysis=all turns off every analysis clad has, asking
for the most conservative derivative it can produce.}]>;

def Rclad_analysis_EQ : Joined<["-"], "Rclad-analysis=">,
  MetaVarName<"<name>">,
  HelpText<"Reports what the named analysis left in the derivative.">,
  DocBrief<[{Reports what the named analysis left in the derivative, as remarks
pointed at the generated code itself. The analogue of -Rpass-missed, which
cannot reach an AST-level plugin.}]>;

def fdump_analysis_EQ : Joined<["-"], "fdump-analysis=">,
  MetaVarName<"<name>">,
  HelpText<"Prints what the named analysis concluded, per function "
           "differentiated.">;

//===----------------------------------------------------------------------===//
// Options that do something rather than set something
//===----------------------------------------------------------------------===//

def fcustom_estimation_model : Flag<["-"], "fcustom-estimation-model">,
  HelpText<"Deprecated; sending in a shared object as the estimation model is "
           "no longer supported.">;

def help : Flag<["-"], "help">,
  HelpText<"Prints out this screen.">;

def version : Flag<["-"], "version">,
  HelpText<"Prints out clad's version.">;

def v : Flag<["-"], "v">,
  HelpText<"Alias for -version.">;
