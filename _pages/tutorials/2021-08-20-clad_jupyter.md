---
title: "Interactive Automatic Differentiation With Clad and Jupyter Notebooks"
layout: post
excerpt: "This tutorial demonstrates the use of automatic differentiation via
clad interactively within Jupyter Notebooks."
sitemap: false
permalink: /tutorials/clad_jupyter/
date: 20-08-2021
author: Ioana Ifrim
custom_css: jupyter
custom_jss: jupyter
---


*Tutorial level: Intro*

{::nomarkdown}

<br /> <br /> <br />


<div tabindex="-1" id="notebook" class="border-box-sizing">
  <div class="container" id="notebook-container">
    <div class="cell border-box-sizing text_cell rendered">
      <div class="prompt input_prompt"></div>
      <div class="inner_cell">
        <div class="text_cell_render border-box-sizing rendered_html">
          <h1 id="Game-of-Life-on-GPU---Interactive-&amp;-Extensible">Game of Life on GPU - Interactive &amp; Extensible<a class="anchor-link" href="#Game-of-Life-on-GPU---Interactive-&amp;-Extensible">&#182;</a></h1>
          <p>
            xeus-cling provides a Jupyter kernel for C++ with the help of the C++
            interpreter cling and the native implementation of the Jupyter protocol xeus.
          </p>

          <p>
            Within the xeus-cling framework, Clad can enable automatic differentiation (AD)
            such that users can automatically generate C++ code for their computation of
            derivatives of their functions.
          </p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing text_cell rendered">
      <div class="prompt input_prompt"></div>
      <div class="inner_cell">
        <div class="text_cell_render border-box-sizing rendered_html">
          <h2 id="Rosenbrock-Function">Rosenbrock Function<a class="anchor-link" href="#Rosenbrock-Function">&para;</a></h2>
          <p>In mathematical optimization, the Rosenbrock function is a non-convex function
             used as a performance test problem for optimization problems. The function is
             defined as:
          </p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[1]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre>
<span class="cp">#include</span> <span class=
"cpf">"clad/Differentiator/Differentiator.h"</span>
<span class="cp">#include</span> <span class="cpf">&lt;chrono&gt;</span>
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[2]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
             <pre>
<span class="c1">// Rosenbrock function declaration</span>
<span class="kt">double</span> <span class="nf">rosenbrock_func</span><span class=
"p">(</span><span class="kt">double</span> <span class="n">x</span><span class=
"p">,</span> <span class="kt">double</span> <span class="n">y</span><span class=
"p">)</span> <span class="p">{</span>
<span class="k">return</span> <span class="p">(</span><span class=
"n">x</span> <span class="o">-</span> <span class="mi">1</span><span class=
"p">)</span> <span class="o">*</span> <span class="p">(</span><span class=
"n">x</span> <span class="o">-</span> <span class="mi">1</span><span class=
"p">)</span> <span class="o">+</span> <span class="mi">100</span> <span class=
"o">*</span> <span class="p">(</span><span class="n">y</span> <span class=
"o">-</span> <span class="n">x</span> <span class="o">*</span> <span class=
"n">x</span><span class="p">)</span> <span class="o">*</span> <span class=
"p">(</span><span class="n">y</span> <span class="o">-</span> <span class=
"n">x</span> <span class="o">*</span> <span class="n">x</span><span class="p">);</span>
<span class="p">}</span>
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing text_cell rendered">
      <div class="prompt input_prompt"></div>
      <div class="inner_cell">
        <div class="text_cell_render border-box-sizing rendered_html">
          <p>In order to compute the function&rsquo;s derivatives, we can employ both
      Clad&rsquo;s Forward Mode or Reverse Mode as detailed below:</p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing text_cell rendered">
      <div class="prompt input_prompt"></div>
      <div class="inner_cell">
        <div class="text_cell_render border-box-sizing rendered_html">
          <h2 id="Forward-Mode-AD">Forward Mode AD<a class="anchor-link" href="#Forward-Mode-AD">&para;</a></h2>
          <p>
            For a function <em>f</em> of several inputs and single (scalar) output, forward
            mode AD can be used to compute (or, in case of Clad, create a function) computing a
            directional derivative of <em>f</em> with respect to a single specified input
            variable. Moreover, the generated derivative function has the same signature as the
            original function <em>f</em>, however its return value is the value of the
            derivative.
          </p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[3]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre>
<span class="kt">double</span> <span class="nf">rosenbrock_forward</span><span class=
"p">(</span><span class="kt">double</span> <span class="n">x</span><span class=
"p">[],</span> <span class="kt">int</span> <span class="n">size</span><span class=
"p">)</span> <span class="p">{</span>
    <span class="kt">double</span> <span class="n">sum</span> <span class=
"o">=</span> <span class="mi">0</span><span class="p">;</span>
    <span class="k">auto</span> <span class="n">rosenbrockX</span> <span class=
"o">=</span> <span class="n">clad</span><span class="o">::</span><span class=
"n">differentiate</span><span class="p">(</span><span class=
"n">rosenbrock_func</span><span class="p">,</span> <span class="mi">0</span><span class=
"p">);</span>
    <span class="k">auto</span> <span class="n">rosenbrockY</span> <span class=
"o">=</span> <span class="n">clad</span><span class="o">::</span><span class=
"n">differentiate</span><span class="p">(</span><span class=
"n">rosenbrock_func</span><span class="p">,</span> <span class="mi">1</span><span class=
"p">);</span>
    <span class="k">for</span> <span class="p">(</span><span class=
"kt">int</span> <span class="n">i</span> <span class="o">=</span> <span class=
"mi">0</span><span class="p">;</span> <span class="n">i</span> <span class=
"o">&lt;</span> <span class="n">size</span><span class="mi">-1</span><span class=
"p">;</span> <span class="n">i</span><span class="o">++</span><span class=
"p">)</span> <span class="p">{</span>
        <span class="kt">double</span> <span class="n">one</span> <span class=
"o">=</span> <span class="n">rosenbrockX</span><span class="p">.</span><span class=
"n">execute</span><span class="p">(</span><span class="n">x</span><span class=
"p">[</span><span class="n">i</span><span class="p">],</span> <span class=
"n">x</span><span class="p">[</span><span class="n">i</span> <span class=
"o">+</span> <span class="mi">1</span><span class="p">]);</span>
        <span class="kt">double</span> <span class="n">two</span> <span class=
"o">=</span> <span class="n">rosenbrockY</span><span class="p">.</span><span class=
"n">execute</span><span class="p">(</span><span class="n">x</span><span class=
"p">[</span><span class="n">i</span><span class="p">],</span> <span class=
"n">x</span><span class="p">[</span><span class="n">i</span> <span class=
"o">+</span> <span class="mi">1</span><span class="p">]);</span>
            <span class="n">sum</span> <span class="o">=</span> <span class=
"n">sum</span> <span class="o">+</span> <span class="n">one</span> <span class=
"o">+</span> <span class="n">two</span><span class="p">;</span>
    <span class="p">}</span>
    <span class="k">return</span> <span class="n">sum</span><span class="p">;</span>
<span class="p">}</span>
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[4]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre>
<span class="k">const</span> <span class="kt">int</span> <span class=
"n">size</span> <span class="o">=</span> <span class="mi">100000000</span><span class=
"p">;</span>
<span class="kt">double</span> <span class="n">Xarray</span><span class=
"p">[</span><span class="n">size</span><span class="p">];</span>
<span class="k">for</span><span class="p">(</span><span class=
"kt">int</span> <span class="n">i</span><span class="o">=</span><span class=
"mi">0</span><span class="p">;</span><span class="n">i</span><span class=
"o">&lt;</span><span class="n">size</span><span class="p">;</span><span class=
"n">i</span><span class="o">++</span><span class="p">)</span>
  <span class="n">Xarray</span><span class="p">[</span><span class=
"n">i</span><span class="p">]</span><span class="o">=</span><span class=
"p">((</span><span class="kt">double</span><span class="p">)</span><span class=
"n">rand</span><span class="p">()</span><span class="o">/</span><span class=
"n">RAND_MAX</span><span class="p">);</span>
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[5]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre>
<span class="kt">double</span> <span class="n">forward_time</span> <span class=
"o">=</span> <span class="n">timeForwardMode</span><span class="p">();</span>
</pre>
            </div>
          </div>
        </div>
      </div>
      <div class="output_wrapper">
        <div class="output">
          <div class="output_area">
            <div class="prompt output_prompt">Out[5]:</div>
            <div class="output_text output_subarea output_execute_result">
              <pre>
Elapsed time for rosenbrock_forward: 3.038005 s
The result of the function is 3232877463.475859.
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing text_cell rendered">
      <div class="prompt input_prompt"></div>
      <div class="inner_cell">
        <div class="text_cell_render border-box-sizing rendered_html">
         <h2 id="Reverse-Mode-AD">Reverse Mode AD<a class="anchor-link" href="#Reverse-Mode-AD">&para;</a></h2>
           <p>
             Reverse-mode AD enables the gradient computation within a single pass of the
             computation graph of <em>f</em> using at most a constant factor (around 4) more
             arithmetical operations compared to the original function. While its constant
             factor and memory overhead is higher than that of the forward-mode, it is
             independent of the number of inputs.</p>

           <p>
             Moreover, the generated function has void return type and same input arguments.
             The function has an additional argument of type T*, where T is the return type of
             <em>f</em>. This is the &ldquo;result&rdquo; argument which has to point to the
             beginning of the vector where the gradient will be stored.
           </p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[6]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre>
<span class="kt">double</span> <span class="nf">rosenbrock_reverse</span><span class=
"p">(</span><span class="kt">double</span> <span class="n">x</span><span class=
"p">[],</span> <span class="kt">int</span> <span class="n">size</span><span class=
"p">)</span> <span class="p">{</span>
    <span class="kt">double</span> <span class="n">sum</span> <span class=
"o">=</span> <span class="mi">0</span><span class="p">;</span>
    <span class="k">auto</span> <span class="n">rosenbrock_dX_dY</span> <span class=
"o">=</span> <span class="n">clad</span><span class="o">::</span><span class=
"n">gradient</span><span class="p">(</span><span class=
"n">rosenbrock_func</span><span class="p">);</span>
    <span class="k">for</span> <span class="p">(</span><span class=
"kt">int</span> <span class="n">i</span> <span class="o">=</span> <span class=
"mi">0</span><span class="p">;</span> <span class="n">i</span> <span class=
"o">&lt;</span> <span class="n">size</span><span class="mi">-1</span><span class=
"p">;</span> <span class="n">i</span><span class="o">++</span><span class=
"p">)</span> <span class="p">{</span>
        <span class="kt">double</span> <span class="n">result</span><span class=
"p">[</span><span class="mi">2</span><span class="p">]</span> <span class=
"o">=</span> <span class="p">{};</span>
        <span class="n">rosenbrock_dX_dY</span><span class="p">.</span><span class=
"n">execute</span><span class="p">(</span><span class="n">x</span><span class=
"p">[</span><span class="n">i</span><span class="p">],</span><span class=
"n">x</span><span class="p">[</span><span class="n">i</span><span class=
"o">+</span><span class="mi">1</span><span class="p">],</span> <span class=
"n">result</span><span class="p">);</span>
        <span class="n">sum</span> <span class="o">=</span> <span class=
"n">sum</span> <span class="o">+</span> <span class="n">result</span><span class=
"p">[</span><span class="mi">0</span><span class="p">]</span> <span class=
"o">+</span> <span class="n">result</span><span class="p">[</span><span class=
"mi">1</span><span class="p">];</span>
    <span class="p">}</span>
    <span class="k">return</span> <span class="n">sum</span><span class="p">;</span>
<span class="p">}</span>
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[7]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre>
<span class="kt">double</span> <span class="n">forward_time</span> <span class=
"o">=</span> <span class="n">timeForwardMode</span><span class="p">();</span>
</pre>
            </div>
          </div>
        </div>
      </div>
      <div class="output_wrapper">
        <div class="output">
          <div class="output_area">
            <div class="prompt output_prompt">Out[7]:</div>
            <div class="output_text output_subarea output_execute_result">
              <pre>
Elapsed time for rosenbrock_forward: 3.038005 s
The result of the function is 3232877463.475859.
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing text_cell rendered">
      <div class="prompt input_prompt"></div>
      <div class="inner_cell">
        <div class="text_cell_render border-box-sizing rendered_html">
          <h2 id="Performance-Comparison">Performance Comparison<a class="anchor-link" href="#Performance-Comparison">&para;</a></h2>
          <p>
            The derivative function created by the forward-mode AD is guaranteed to have at
            most a constant factor (around 2-3) more arithmetical operations compared to the
            original function. Whilst for the reverse-mode AD for a function having N inputs
            and consisting of T arithmetical operations, computing its gradient takes a single
            execution of the reverse-mode AD and around 4T operations. In comparison, it would
            take N executions of the forward-mode, this requiring up to N3*T operations.
          </p>
        </div>
      </div>
    </div>

    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[8]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre>
<span class="kt">double</span> <span class="n">difference</span> <span class=
"o">=</span> <span class="n">forward_time</span> <span class="o">-</span>  <span class=
"n">reverse_time</span><span class="p">;</span>
<span class="n">printf</span><span class="p">(</span><span class=
"s">"Forward - Reverse timing for an array of size: %d is: %fs</span><span class=
"se">\n</span><span class="s">"</span><span class="p">,</span> <span class=
"n">size</span><span class="p">,</span> <span class="n">difference</span><span class=
"p">);</span>
</pre>
            </div>
          </div>
        </div>
      </div>
      <div class="output_wrapper">
        <div class="output">
          <div class="output_area">
            <div class="prompt output_prompt">Out[8]:</div>
            <div class="output_text output_subarea output_execute_result">
              <pre>
Forward - Reverse timing for an array of size: 100000000 is: 0.125806s
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="cell border-box-sizing text_cell rendered">
      <div class="prompt input_prompt"></div>
      <div class="inner_cell">
        <div class="text_cell_render border-box-sizing rendered_html">
          <h2 id="Clad-Produced-Code">Clad Produced Code<a class="anchor-link" href="#Clad-Produced-Code">&para;</a></h2>
          <p>
            We can now call <code>rosenbrockX</code> / <code>rosenbrockY</code> /
            <code>rosenbrock_dX_dY.dump()</code> to obtain a print out of the Clad&rsquo;s
            generated code. As an illustration, the reverse-mode produced code is:
          </p>
        </div>
      </div>
    </div>

    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[9]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre>
<span class="k">auto</span> <span class="n">rosenbrock_dX_dY</span> <span class=
"o">=</span> <span class="n">clad</span><span class="o">::</span><span class=
"n">gradient</span><span class="p">(</span><span class=
"n">rosenbrock_func</span><span class="p">);</span>
<span class="n">rosenbrock_dX_dY</span><span class="p">.</span><span class=
"n">dump</span><span class="p">();</span>
</pre>
            </div>
          </div>
        </div>
      </div>
      <div class="output_wrapper">
        <div class="output">
          <div class="output_area">
            <div class="prompt output_prompt">Out[9]:</div>
            <div class="output_text output_subarea output_execute_result">
            <pre>
The code is: void rosenbrock_func_grad(double x, double y, double *_result) {
    double _t2;
    double _t3;
    double _t4;
    double _t5;
    double _t6;
    double _t7;
    double _t8;
    double _t9;
    double _t10;
    _t3 = (x - 1);
    _t2 = (x - 1);
    _t7 = x;
    _t6 = x;
    _t5 = (y - _t7 * _t6);
    _t8 = 100 * _t5;
    _t10 = x;
    _t9 = x;
    _t4 = (y - _t10 * _t9);
    double rosenbrock_func_return = _t3 * _t2 + _t8 * _t4;
    goto _label0;
  _label0:
    {
        double _r0 = 1 * _t2;
        _result[0UL] += _r0;
        double _r1 = _t3 * 1;
        _result[0UL] += _r1;
        double _r2 = 1 * _t4;
        double _r3 = _r2 * _t5;
        double _r4 = 100 * _r2;
        _result[1UL] += _r4;
        double _r5 = -_r4 * _t6;
        _result[0UL] += _r5;
        double _r6 = _t7 * -_r4;
        _result[0UL] += _r6;
        double _r7 = _t8 * 1;
        _result[1UL] += _r7;
        double _r8 = -_r7 * _t9;
        _result[0UL] += _r8;
        double _r9 = _t10 * -_r7;
        _result[0UL] += _r9;
    }
}

</pre>
            </div>
          </div>
        </div>
      </div>
    </div>



  </div>
</div>


<br /> <br /> <br />

{:/}
