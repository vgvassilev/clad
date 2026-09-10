#!/usr/bin/env python3
"""HistFactory-style likelihood benchmark for clad hessians.

Generates a scaled-down proxy of a real 1470-parameter ATLAS likelihood
with the same computational structure as the code RooFit generates for it,
compiles it with the clad plugin, and runs it. The generated .cpp is a
transient artifact; the compilation time is reported in the same format as
the timing output of the benchmark executable, so a full run reads as one
timing table.

Structure mirrored from the full benchmark:
- N channel functions, each: weight-sum loop; straight-line per-sample
  yields (flexibleInterp over a nuisance subset x product-of-4 free norms x
  lumi); a bin loop with per-sample uniformBinNumber shape lookups, a
  yield.shape dot product and a binned nll term.
- A top-level function summing the channels plus a constraint block:
  gaussian/gaussianIntegral ratios per nuisance, poisson/poissonIntegral
  ratios per gamma parameter, one tight (1e-4) lumi constraint, all fed
  into constraintSum (which carries clad's loop-checkpointing pragma, like
  in ROOT).

Scale: 91 parameters (36 gaussian nuisances, 6 gammas, lumi, 5 free norms,
36+6+1 global observables), 4 channels with {8,10,12,6} samples and
{1,2,4,1} bins.

Usage:
  histfactory.py                       # generate + compile + run against ../build
  histfactory.py --clad-build <dir>    # use another clad build tree
  histfactory.py --cpp out.cpp         # also keep the generated source
  histfactory.py --gen-only --cpp out.cpp
"""

import argparse
import math
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO = BENCHMARK_DIR.parent

# ---- parameter layout -------------------------------------------------------
N_NUIS = 36          # gaussian-constrained interpolation nuisances
N_GAMMA = 6          # poisson-constrained (gamma) parameters
NUIS = list(range(0, N_NUIS))                    # 0..35
GAMMA = list(range(N_NUIS, N_NUIS + N_GAMMA))    # 36..41
LUMI = 42                                        # in every yield, 1e-4 constr.
FREE = [43, 44, 45, 46, 47]                      # unconstrained norm factors
GOBS0 = 48                                       # 48..83 gaussian glob. obs
POBS0 = 84                                       # 84..89 poisson glob. obs
LUMIOBS = 90
N_PARAMS = 91

# ---- channels ---------------------------------------------------------------
CH_SAMPLES = [8, 10, 12, 6]
CH_BINS = [1, 2, 4, 1]
INTERP_SIZES = [24, 3, 28, 16, 8, 20, 32, 12,
                26, 22, 2, 30, 18, 25, 6, 23, 27, 10,
                21, 29, 4, 24, 19, 31, 14, 26, 22, 5, 33, 17,
                28, 7, 25, 15, 34, 11]
assert len(INTERP_SIZES) == sum(CH_SAMPLES)

TAU = 50.0

# The timing/validation driver appended to the generated model code.
DRIVER = r'''
namespace {

double funcEval(std::vector<double> &params)
{
   return roo_mini_0(params.data(), observablesVec.data(), auxConstantsVec.data());
}

template <typename Fn>
double timeIt(int reps, Fn &&fn)
{
   auto start = std::chrono::steady_clock::now();
   for (int r = 0; r < reps; ++r) {
      fn();
   }
   auto stop = std::chrono::steady_clock::now();
   return std::chrono::duration<double, std::milli>(stop - start).count() / reps;
}

} // namespace

int main(int argc, char **argv)
{
   const bool validate = !(argc > 1 && std::strcmp(argv[1], "--no-validation") == 0);

   const std::size_t n = parametersVec.size();

   auto grad = clad::gradient(roo_mini_0, "params");
   auto hess = clad::hessian(roo_mini_0, "params[0:{maxparam}]");

   std::vector<double> gradientVec(n);
   std::vector<double> hessianVec(n * n);

   auto evalGradient = [&](std::vector<double> &params, std::vector<double> &out) {
      std::fill(out.begin(), out.end(), 0.0);
      grad.execute(params.data(), observablesVec.data(), auxConstantsVec.data(), out.data());
   };
   auto evalHessian = [&](std::vector<double> &params, std::vector<double> &out) {
      std::fill(out.begin(), out.end(), 0.0);
      hess.execute(params.data(), observablesVec.data(), auxConstantsVec.data(), out.data());
   };

   const double nllVal = funcEval(parametersVec);
   std::printf("n params      : %zu\n", n);
   std::printf("function value: %.15g\n", nllVal);

   // ---------- Timings ----------
   const double tFunc = timeIt(2000, [&] { funcEval(parametersVec); });
   std::printf("primal        : %10.3f ms/call\n", tFunc);

   const double tGrad = timeIt(500, [&] { evalGradient(parametersVec, gradientVec); });
   std::printf("gradient      : %10.3f ms/call  (%.1fx primal)\n", tGrad, tGrad / tFunc);

   const double tHess = timeIt(20, [&] { evalHessian(parametersVec, hessianVec); });
   std::printf("hessian       : %10.3f ms/call  (%.1fx gradient)\n", tHess, tHess / tGrad);

   if (!validate) {
      return 0;
   }

   int numBadGrad = 0;
   int numBadHess = 0;

   // ---------- Validate the gradient against central differences ----------
   {
      std::vector<double> p = parametersVec;
      double worst = 0.0;
      std::size_t worstIdx = 0;
      for (std::size_t i = 0; i < n; ++i) {
         const double eps = 1e-6;
         p[i] = parametersVec[i] - eps;
         const double funcValDown = funcEval(p);
         p[i] = parametersVec[i] + eps;
         const double funcValUp = funcEval(p);
         p[i] = parametersVec[i];
         const double num = (funcValUp - funcValDown) / (2 * eps);

         // The primal is O(1e6), so central differences carry ~1e-4 of
         // cancellation noise on top of the truncation error.
         const double err = std::abs(gradientVec[i] - num) / std::max(1.0, std::abs(num));
         if (err > worst) {
            worst = err;
            worstIdx = i;
         }
         if (err > 1e-3) {
            if (++numBadGrad <= 10) {
               std::printf("  grad[%zu]: clad=%.10g num=%.10g\n", i, gradientVec[i], num);
            }
         }
      }
      std::printf("gradient vs numerical: worst rel. deviation %.2e (at index %zu), %d outside tolerance\n", worst,
                  worstIdx, numBadGrad);
   }

   // ---------- Validate sampled hessian rows against central differences of the gradient ----------
   {
      std::vector<double> p = parametersVec;
      std::vector<double> gradUp(n);
      std::vector<double> gradDown(n);
      double worst = 0.0;
      std::size_t worstRow = 0;
      std::size_t worstCol = 0;
      unsigned int lcg = 12345;
      for (int sample = 0; sample < 8; ++sample) {
         lcg = 1664525 * lcg + 1013904223;
         const std::size_t i = lcg % n;
         const double eps = 1e-5;
         p[i] = parametersVec[i] - eps;
         evalGradient(p, gradDown);
         p[i] = parametersVec[i] + eps;
         evalGradient(p, gradUp);
         p[i] = parametersVec[i];
         for (std::size_t j = 0; j < n; ++j) {
            const double num = (gradUp[j] - gradDown[j]) / (2 * eps);
            const double cladH = hessianVec[i * n + j];
            const double err = std::abs(cladH - num) / std::max(1.0, std::abs(num));
            if (err > worst) {
               worst = err;
               worstRow = i;
               worstCol = j;
            }
            if (err > 1e-3) {
               if (++numBadHess <= 10) {
                  std::printf("  hess[%zu,%zu]: clad=%.10g num=%.10g\n", i, j, cladH, num);
               }
            }
         }
      }
      std::printf("hessian vs numerical (8 sampled rows): worst rel. deviation %.2e (at [%zu,%zu]), %d outside "
                  "tolerance\n",
                  worst, worstRow, worstCol, numBadHess);
   }

   // ---------- Hessian symmetry ----------
   {
      double worst = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
         for (std::size_t j = 0; j < i; ++j) {
            const double a = hessianVec[i * n + j];
            const double b = hessianVec[j * n + i];
            worst = std::max(worst, std::abs(a - b) / std::max(1.0, std::abs(a)));
         }
      }
      std::printf("hessian asymmetry: worst rel. deviation %.2e\n", worst);
   }

   if (numBadGrad + numBadHess > 0) {
      std::printf("VALIDATION FAILED\n");
      return 1;
   }
   std::printf("validation passed\n");
   return 0;
}
'''


def fmt(v):
    return f"{v:.9g}"


def generate():
    """Return the full source text of the mini benchmark."""
    rng = random.Random(20260821)
    xl = []          # xlArr contents, allocated sequentially
    lines = []       # generated code lines

    def alloc_xl(values):
        off = len(xl)
        xl.extend(values)
        return off

    # parameter values --------------------------------------------------------
    par = [0.0] * N_PARAMS
    for i in NUIS:
        par[i] = 0.35 * math.sin(3.1 * i + 0.4)          # interior |alpha| < 1
    for j, g in enumerate(GAMMA):
        par[g] = 1.0 + 0.06 * math.sin(2.3 * j)
    par[LUMI] = 1.0
    for k, f in enumerate(FREE):
        par[f] = 1.0 + 0.08 * math.cos(1.7 * k)
    for i in NUIS:                                        # global observables
        par[GOBS0 + i] = par[i] + 0.12 * math.sin(5.7 * i)
    for j in range(N_GAMMA):
        par[POBS0 + j] = round(TAU * par[GAMMA[j]] + 3 * math.sin(7.1 * j))
    par[LUMIOBS] = 1.0

    def interp_expr(t, subset, size):
        """Emit a flexibleInterp over `size` nuisances like the codegen does."""
        los = [1.0 - (0.04 + 0.03 * abs(math.sin(2.9 * (len(xl) + i)))) for i in range(size)]
        his = [1.0 + (0.04 + 0.03 * abs(math.cos(3.7 * (len(xl) + i)))) for i in range(size)]
        lo_off = alloc_xl(los)
        hi_off = alloc_xl(his)
        plist = ", ".join(f"params[{p}]" for p in subset)
        lines.append(f"    double t{t}[]{{{plist}}};")
        lines.append(
            f"    const double t{t + 1} = RooFit::Detail::MathFuncs::flexibleInterp"
            f"(5, t{t}, {size}, xlArr + {lo_off}, xlArr + {hi_off}, 1, 1, 1);")
        return t + 2

    # channel functions -------------------------------------------------------
    obs_data = []        # payload appended after the 12 header slots
    channel_exp = []     # per channel: expected yields per bin (for weights)

    for c, (n_samples, n_bins) in enumerate(zip(CH_SAMPLES, CH_BINS)):
        t = 1
        lines.append(f"double roo_mini_{c + 1}(double *params, const double *obs, const double *xlArr)")
        lines.append("{")
        lines.append("    const double t0 = 1;")
        hdr = 3 * c
        lines.append(f"    double wsum_{c} = 0.;")
        lines.append(f"    double res_{c} = 0.;")
        lines.append(f"    for (int loopIdx1 = 0; loopIdx1 < obs[{hdr + 2}]; loopIdx1++) {{")
        lines.append(f"        wsum_{c} += obs[static_cast<int>(obs[{hdr + 1}]) + loopIdx1];")
        lines.append("    }")
        lines.append(f"    res_{c} += wsum_{c} * std::log({TAU:.1f});")
        # gamma parameter as 1-element lookup table, like `double t4[]{params[404]}`
        gpar = GAMMA[c % N_GAMMA]
        lines.append(f"    double tg[]{{params[{gpar}]}};")
        # two product-of-4 free-norm factors shared between samples
        f4 = []
        for _ in range(2):
            quad = rng.sample(FREE, 4)
            qlist = ", ".join(f"params[{q}]" for q in quad)
            lines.append(f"    double t{t}[]{{{qlist}}};")
            lines.append(f"    const double t{t + 1} = RooFit::Detail::MathFuncs::product(t{t}, 4);")
            f4.append(t + 1)
            t += 2
        # straight-line sample yields
        yields = []
        sizes = INTERP_SIZES[sum(CH_SAMPLES[:c]):sum(CH_SAMPLES[:c]) + n_samples]
        for s, size in enumerate(sizes):
            subset = rng.sample(NUIS, min(size, N_NUIS))
            t2 = interp_expr(t, subset, len(subset))
            interp_res = t2 - 1
            t = t2
            if s % 4 == 3:  # some yields have no product-of-4 factor
                lines.append(f"    double t{t}[]{{t{interp_res}, params[{LUMI}]}};")
                lines.append(f"    const double t{t + 1} = RooFit::Detail::MathFuncs::product(t{t}, 2);")
            else:
                lines.append(f"    double t{t}[]{{t{interp_res}, t{f4[s % 2]}, params[{LUMI}]}};")
                lines.append(f"    const double t{t + 1} = RooFit::Detail::MathFuncs::product(t{t}, 3);")
            yields.append(t + 1)
            t += 2
        ylist = ", ".join(f"t{y}" for y in yields)
        lines.append(f"    double ty[]{{{ylist}}};")
        # per-sample shape tables and expected bin contents
        shape_offs = []
        exp_bins = [0.0] * n_bins
        for s in range(n_samples):
            shapes = [2.0 + 18.0 * abs(math.sin(1.3 * (len(xl) + b + 7 * s))) for b in range(n_bins)]
            shape_offs.append(alloc_xl(shapes))
            for b in range(n_bins):
                exp_bins[b] += shapes[b]  # yields are ~1 at nominal
        channel_exp.append(exp_bins)
        # bin loop
        lines.append(f"    for (int loopIdx1 = 0; loopIdx1 < obs[{hdr + 2}]; loopIdx1++) {{")
        bl = []
        tb = t
        for s in range(n_samples):
            off = shape_offs[s]
            shape = (f"(xlArr + {off})[RooFit::Detail::MathFuncs::uniformBinNumber"
                     f"(75, 150, obs[static_cast<int>(obs[{hdr}]) + loopIdx1], 1, 1)]")
            lines.append(f"        const double t{tb} = {shape};")
            if s % 3 == 0:  # some samples carry the gamma shape factor
                gshape = (f"tg[RooFit::Detail::MathFuncs::uniformBinNumber"
                          f"(75, 150, obs[static_cast<int>(obs[{hdr}]) + loopIdx1], 1, 1)]")
                lines.append(f"        const double t{tb + 1} = {gshape};")
                lines.append(f"        double t{tb + 2}[]{{t{tb}, t{tb + 1}, t0}};")
                lines.append(f"        const double t{tb + 3} = RooFit::Detail::MathFuncs::product(t{tb + 2}, 3);")
                bl.append(tb + 3)
                tb += 4
            else:
                lines.append(f"        double t{tb + 1}[]{{t{tb}, t0}};")
                lines.append(f"        const double t{tb + 2} = RooFit::Detail::MathFuncs::product(t{tb + 1}, 2);")
                bl.append(tb + 2)
                tb += 3
        bllist = ", ".join(f"t{b}" for b in bl)
        lines.append(f"        double tb[]{{{bllist}}};")
        lines.append("        double tsum = 0;")
        lines.append("        double tnorm = 0;")
        lines.append(f"        for (int i = 0; i < {n_samples}; i++) {{")
        lines.append("            tsum += tb[i] * ty[i];")
        lines.append("            tnorm += ty[i];")
        lines.append("        }")
        lines.append(
            f"        res_{c} += RooFit::Detail::MathFuncs::nll(tsum, "
            f"obs[static_cast<int>(obs[{hdr + 1}]) + loopIdx1], 1, 0);")
        lines.append("    }")
        lines.append(f"    return res_{c};")
        lines.append("}")
        lines.append("")

    # obs vector: 12 header slots, then per channel x-values and weights
    obs = [0.0] * 12
    for c, n_bins in enumerate(CH_BINS):
        x_off = 12 + len(obs_data)
        for b in range(n_bins):
            obs_data.append(75.0 + (150.0 - 75.0) * (b + 0.5) / n_bins)
        w_off = 12 + len(obs_data)
        for b in range(n_bins):
            # data close to the nominal expectation, with a small offset
            obs_data.append(round(channel_exp[c][b] * (1.0 + 0.05 * math.sin(4.9 * (c + b))), 1))
        obs[3 * c] = x_off
        obs[3 * c + 1] = w_off
        obs[3 * c + 2] = n_bins
    obs += obs_data

    # top-level function ------------------------------------------------------
    lines.append("double roo_mini_0(double *params, const double *obs, const double *xlArr)")
    lines.append("{")
    calls = " + ".join(f"roo_mini_{c + 1}(params, obs, xlArr)" for c in range(len(CH_SAMPLES)))
    lines.append(f"    const double t0 = ({calls});")
    t = 1
    cons = []
    # tight lumi constraint
    lines.append(f"    const double t{t} = 1.0E-4;")
    lines.append(f"    const double t{t + 1} = RooFit::Detail::MathFuncs::gaussian(params[{LUMI}], params[{LUMIOBS}], t{t});")
    cons.append(t + 1)
    lines.append(f"    const double t{t + 2} = 1;")
    tone = t + 2
    t += 3
    for i in NUIS:
        lines.append(f"    const double t{t} = RooFit::Detail::MathFuncs::gaussian(params[{i}], params[{GOBS0 + i}], t{tone});")
        lines.append(f"    const double t{t + 1} = RooFit::Detail::MathFuncs::gaussianIntegral(-10, 10, params[{i}], t{tone});")
        lines.append(f"    const double t{t + 2} = t{t} / t{t + 1};")
        cons.append(t + 2)
        t += 3
    for j in range(N_GAMMA):
        lines.append(f"    const double t{t} = {fmt(TAU)};")
        lines.append(f"    double t{t + 1}[]{{params[{GAMMA[j]}], t{t}}};")
        lines.append(f"    const double t{t + 2} = RooFit::Detail::MathFuncs::product(t{t + 1}, 2);")
        lines.append(f"    const double t{t + 3} = RooFit::Detail::MathFuncs::poisson(params[{POBS0 + j}], t{t + 2});")
        lines.append(f"    const double t{t + 4} = RooFit::Detail::MathFuncs::poissonIntegral(1, t{t + 2}, 0, 0, 1.0E+30, 1);")
        lines.append(f"    const double t{t + 5} = t{t + 3} / t{t + 4};")
        cons.append(t + 5)
        t += 6
    clist = ", ".join(f"t{cn}" for cn in cons)
    lines.append(f"    double tc[]{{{clist}}};")
    lines.append(f"    const double t{t} = RooFit::Detail::MathFuncs::constraintSum(tc, {len(cons)});")
    lines.append(f"    return t0 + t{t};")
    lines.append("}")

    # assemble the file -------------------------------------------------------
    sep = "// " + "-" * 75 + "\n"
    shared = (
        "// Scaled-down clad benchmark: hessian of a HistFactory-style likelihood.\n"
        "//\n"
        "// Transient file, generated by histfactory.py (see there for the\n"
        "// structure); a small proxy of the code RooFit generates for a real\n"
        "// 1470-parameter ATLAS likelihood.\n\n"
        '#include "HistFactoryMathFuncs.h"\n\n'
        "#include <algorithm>\n"
        "#include <chrono>\n"
        "#include <cmath>\n"
        "#include <cstddef>\n"
        "#include <cstdio>\n"
        "#include <cstring>\n"
        "#include <vector>\n\n"
        # constraintSum sits here rather than in the header: clad currently
        # mis-attributes a checkpoint pragma that lives in an included header
        # (the planner selects pragmas by raw SourceLocation order, which does
        # not match translation-unit order across files).
        "namespace RooFit {\n"
        "namespace Detail {\n"
        "namespace MathFuncs {\n"
        "\n"
        "template <typename DoubleArray>\n"
        "double constraintSum(DoubleArray comp, unsigned int compSize)\n"
        "{\n"
        "   double sum = 0;\n"
        "#ifndef CLAD_HISTFACTORY_NO_CHECKPOINT\n"
        "#pragma clad checkpoint loop\n"
        "#endif\n"
        "   for (unsigned int i = 0; i < compSize; i++) {\n"
        "      sum -= std::log(comp[i]);\n"
        "   }\n"
        "   return sum;\n"
        "}\n"
        "\n"
        "} // namespace MathFuncs\n"
        "} // namespace Detail\n"
        "} // namespace RooFit\n\n"
    )

    def literal_block(name, values):
        out = [f"std::vector<double> {name} = {{"]
        for i in range(0, len(values), 10):
            chunk = ", ".join(fmt(v) for v in values[i:i + 10])
            tail = "," if i + 10 < len(values) else ""
            out.append("    " + chunk + tail)
        out.append("};")
        return "\n".join(out)

    data = "\n\n".join([
        "// clang-format off",
        literal_block("parametersVec", par),
        literal_block("observablesVec", obs),
        literal_block("auxConstantsVec", xl),
        "// clang-format on",
    ])

    driver = DRIVER.replace("{maxparam}", str(N_PARAMS - 1))

    return "".join([
        shared, sep,
        "// Generated model code: HistFactory-style channels, mini scale.\n",
        sep, "\n",
        "\n".join(lines), "\n\n",
        data, "\n",
        driver,
    ])


def cmake_cache_get(build_dir, key):
    cache = build_dir / "CMakeCache.txt"
    if cache.exists():
        m = re.search(rf"^{re.escape(key)}:\w+=(.*)$", cache.read_text(), re.M)
        if m:
            return m.group(1).strip()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clad-build", type=Path, default=REPO / "build",
                    help="clad build tree containing lib/clad.so (default: ./build)")
    ap.add_argument("--compiler", default=None,
                    help="C++ compiler to use; must be the clang the clad "
                         "build targets (default: CMAKE_CXX_COMPILER from the "
                         "build tree's CMakeCache.txt)")
    ap.add_argument("--cpp", type=Path, default=None,
                    help="write (and keep) the generated source here instead of a temp dir")
    ap.add_argument("--gen-only", action="store_true",
                    help="only generate the source, do not compile or run")
    ap.add_argument("--run-args", nargs=argparse.REMAINDER, default=[],
                    help="arguments passed to the benchmark executable (e.g. --no-validation)")
    args = ap.parse_args()

    if args.gen_only and args.cpp is None:
        ap.error("--gen-only needs --cpp to name the output file")

    source = generate()

    tmp = None
    if args.cpp is None:
        tmp = tempfile.TemporaryDirectory(prefix="histfactory_")
        cpp = Path(tmp.name) / "histfactory_model.cpp"
    else:
        cpp = args.cpp
    cpp.write_text(source)

    if args.gen_only:
        print(f"generated     : {cpp} ({len(source.splitlines())} lines)")
        return 0

    build = args.clad_build.resolve()
    clad_so = build / "lib" / "clad.so"
    if not clad_so.exists():
        sys.exit(f"error: {clad_so} not found (pass --clad-build)")
    compiler = (args.compiler or
                cmake_cache_get(build, "CMAKE_CXX_COMPILER") or "clang++")
    include = cmake_cache_get(build, "CMAKE_HOME_DIRECTORY")
    include = (Path(include) if include else REPO) / "include"
    exe = cpp.with_suffix("")

    cmd = [compiler, "-std=c++17", "-O2", "-I", str(include),
           "-I", str(BENCHMARK_DIR),
           "-Xclang", "-add-plugin", "-Xclang", "clad",
           "-Xclang", "-load", "-Xclang", str(clad_so),
           "-DCLAD_NO_NUM_DIFF", str(cpp), "-o", str(exe)]

    print(f"clad build    : {build}")
    start = time.monotonic()
    res = subprocess.run(cmd)
    seconds = time.monotonic() - start
    if res.returncode != 0:
        sys.exit(res.returncode)
    print(f"compilation   : {seconds:10.3f} s")
    sys.stdout.flush()

    res = subprocess.run([str(exe)] + args.run_args)
    if tmp is not None:
        tmp.cleanup()
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
