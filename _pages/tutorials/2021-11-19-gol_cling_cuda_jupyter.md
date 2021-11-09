---
title: "Game of Life on GPU Using Cling-CUDA"
layout: post
excerpt: "This tutorial demonstrates some functions of Cling-CUDA and Jupyter Notebooks
and gives an idea what you can do with C++ in a web browser."
sitemap: false
permalink: /tutorials/gol_cling_cuda_jupyter/
date: 09-11-2021
author: Simeon Ehrig
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
          <p>This notebook demonstrates some functions of Cling-CUDA and Jupyter Notebooks and gives an idea what you can do with C++ in a web browser. The example shows the usual workflow of simulation and analysis. The simulation runs <a href="https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life">Conway's Game of Life</a> on a GPU.</p>
          <p>You can jump directly to the functions and look at them, but independent execution of the cells is not possible because there are some dependencies. Just start with the first cell and run (<code>Shift + Enter</code>) the cells with one exception(see nonlinear program flow) downwards.</p>
          <p>The following functions can be found in the notebook:</p>
          <ul>
            <li><a href="#Include-and-Link-Files">reuse code via header files and shared libraries</a></li>
            <li><a href="#CUDA-Kernels">interactive definition of CUDA kernels</a></li>
            <li><a href="#Interactive-Input">write a cell directly in a file (magic command)</a></li>
            <li><a href="#Cling-I%2FO-System">the simple I/O system of Cling</a></li>
            <li><a href="#Display-Simulation-Images">display images directly in the notebook</a></li>
            <li><a href="#Nonlinear-Program-Flow">nonlinear program flow</a></li>
            <li><a href="#In-Situ-Data-Analysis">In-Situ data analysis (zero copy)</a></li>
            <li><a href="#Dynamic-Extension-Without-Loss-of-State">dynamic extension of the program without loss of state</a></li>
            <li><a href="#Continue-Simulation-Analysis-Loop">continue the simulation analysis loop without additional calculation or memory copies</a></li>
          </ul>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing text_cell rendered">
      <div class="prompt input_prompt"></div>
      <div class="inner_cell">
        <div class="text_cell_render border-box-sizing rendered_html">
          <h2 id="Include-and-Link-Files">Include and Link Files<a class="anchor-link" href="#Include-and-Link-Files">&#182;</a></h2>
          <p>Cling allows to include and link existing code. For this project some help functions were written, which are loaded and compiled at runtime. Additionally, the external library <code>pngwriter</code> is used to create images from the data. The source code for this example was downloaded from git and compiled as a shared library.</p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing text_cell rendered">
      <div class="prompt input_prompt"></div>
      <div class="inner_cell">
        <div class="text_cell_render border-box-sizing rendered_html">
          <p>Commands beginning with a dot are Cling metacommands. They make it possible to manipulate the state of the cling instance. For example, <code>.I</code> allows you to specify an include path at runtime.</p>
          <p>set include path</p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[1]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="p">.</span><span class="n">I</span> <span class="n">pngwriter</span><span class="o">/</span><span class="n">include</span>
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
              <pre><span></span><span class="cp">#include</span> <span class="cpf">&lt;fstream&gt;</span><span class="cp"></span>
<span class="cp">#include</span> <span class="cpf">&lt;vector&gt;</span><span class="cp"></span>
<span class="cp">#include</span> <span class="cpf">&lt;sstream&gt;</span><span class="cp"></span>
<span class="cp">#include</span> <span class="cpf">&lt;chrono&gt;</span><span class="cp"></span>
<span class="cp">#include</span> <span class="cpf">&lt;thread&gt;</span><span class="cp"></span>

<span class="c1">// lib pngwriter</span>
<span class="cp">#define NO_FREETYPE</span>
<span class="cp">#include</span> <span class="cpf">&lt;pngwriter.h&gt;</span><span class="cp"></span>

<span class="c1">// self-defined help functions</span>
<span class="cp">#include</span> <span class="cpf">&quot;color_maps.hpp&quot;</span><span class="cp"></span>
<span class="cp">#include</span> <span class="cpf">&quot;input_reader.hpp&quot;</span><span class="cp"></span>
<span class="cp">#include</span> <span class="cpf">&quot;png_generator.hpp&quot;</span><span class="cp"></span>

<span class="c1">// help functions for additional notebook functions</span>
<span class="cp">#include</span> <span class="cpf">&quot;xtl/xbase64.hpp&quot;</span><span class="cp"></span>
<span class="cp">#include</span> <span class="cpf">&quot;xeus/xjson.hpp&quot;</span><span class="cp"></span>
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
          <p>load precompield shared libary (<a href="https://github.com/pngwriter/pngwriter">pngwriter</a>)</p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[3]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="p">.</span><span class="n">L</span> <span class="n">pngwriter</span><span class="o">/</span><span class="n">lib</span><span class="o">/</span><span class="n">libPNGwriter</span><span class="p">.</span><span class="n">so</span>
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
          <h2 id="Help-functions">Help functions<a class="anchor-link" href="#Help-functions">&#182;</a></h2>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[4]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// checks whether the return value was successful</span>
<span class="c1">// If not, print an error message</span>
<span class="kr">inline</span> <span class="kt">void</span> <span class="nf">cuCheck</span><span class="p">(</span><span class="n">cudaError_t</span> <span class="n">code</span><span class="p">){</span>
    <span class="k">if</span><span class="p">(</span><span class="n">code</span> <span class="o">!=</span> <span class="n">cudaSuccess</span><span class="p">){</span>
        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">&quot;Error code: &quot;</span> <span class="o">&lt;&lt;</span> <span class="n">code</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span> <span class="o">&lt;&lt;</span> <span class="n">cudaGetErrorString</span><span class="p">(</span><span class="n">code</span><span class="p">)</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span>
    <span class="p">}</span>
<span class="p">}</span>
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
              <pre><span></span><span class="c1">// display image in the notebook</span>
<span class="kt">void</span> <span class="nf">display_image</span><span class="p">(</span><span class="n">std</span><span class="o">::</span><span class="n">vector</span><span class="o">&lt;</span> <span class="kt">unsigned</span> <span class="kt">char</span><span class="o">&gt;</span> <span class="o">&amp;</span> <span class="n">image</span><span class="p">,</span> <span class="kt">bool</span> <span class="n">clear_ouput</span><span class="p">){</span>
    <span class="c1">// memory objects for output in the web browser</span>
    <span class="n">std</span><span class="o">::</span><span class="n">stringstream</span> <span class="n">buffer</span><span class="p">;</span>
    <span class="n">xeus</span><span class="o">::</span><span class="n">xjson</span> <span class="n">mine</span><span class="p">;</span>

    <span class="k">if</span><span class="p">(</span><span class="n">clear_ouput</span><span class="p">)</span>
        <span class="n">xeus</span><span class="o">::</span><span class="n">get_interpreter</span><span class="p">().</span><span class="n">clear_output</span><span class="p">(</span><span class="nb">true</span><span class="p">);</span>

    <span class="n">buffer</span><span class="p">.</span><span class="n">str</span><span class="p">(</span><span class="s">&quot;&quot;</span><span class="p">);</span>
    <span class="k">for</span><span class="p">(</span><span class="k">auto</span> <span class="nl">c</span> <span class="p">:</span> <span class="n">image</span><span class="p">){</span>
        <span class="n">buffer</span> <span class="o">&lt;&lt;</span> <span class="n">c</span><span class="p">;</span>
    <span class="p">}</span>

    <span class="n">mine</span><span class="p">[</span><span class="s">&quot;image/png&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">xtl</span><span class="o">::</span><span class="n">base64encode</span><span class="p">(</span><span class="n">buffer</span><span class="p">.</span><span class="n">str</span><span class="p">());</span>
    <span class="n">xeus</span><span class="o">::</span><span class="n">get_interpreter</span><span class="p">().</span><span class="n">display_data</span><span class="p">(</span>
        <span class="n">std</span><span class="o">::</span><span class="n">move</span><span class="p">(</span><span class="n">mine</span><span class="p">),</span>
        <span class="n">xeus</span><span class="o">::</span><span class="n">xjson</span><span class="o">::</span><span class="n">object</span><span class="p">(),</span>
        <span class="n">xeus</span><span class="o">::</span><span class="n">xjson</span><span class="o">::</span><span class="n">object</span><span class="p">());</span>
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
          <h2 id="CUDA-Kernels">CUDA Kernels<a class="anchor-link" href="#CUDA-Kernels">&#182;</a></h2>
          <ul>
            <li>define kernels at runtime</li>
            <li>unfortunately C++ does not allow redefinition</li>
          </ul>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[6]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// Improve the memory access (reduce the memory control actions)</span>
<span class="c1">// The kernel appends the first row to the last row and the last before the first</span>
<span class="c1">// The top line is also appended below the last line and vice versa</span>

<span class="n">__global__</span> <span class="kt">void</span> <span class="nf">copy_ghostcells</span><span class="p">(</span><span class="kt">int</span> <span class="n">dim</span><span class="p">,</span> <span class="kt">int</span> <span class="o">*</span><span class="n">world</span><span class="p">)</span> <span class="p">{</span>
  <span class="kt">int</span> <span class="n">col</span> <span class="o">=</span> <span class="n">blockIdx</span><span class="p">.</span><span class="n">x</span> <span class="o">*</span> <span class="n">blockDim</span><span class="p">.</span><span class="n">x</span> <span class="o">+</span> <span class="n">threadIdx</span><span class="p">.</span><span class="n">x</span><span class="p">;</span>
  <span class="kt">int</span> <span class="n">row</span> <span class="o">=</span> <span class="n">blockIdx</span><span class="p">.</span><span class="n">y</span> <span class="o">*</span> <span class="n">blockDim</span><span class="p">.</span><span class="n">y</span> <span class="o">+</span> <span class="n">threadIdx</span><span class="p">.</span><span class="n">y</span><span class="p">;</span>

  <span class="c1">// ignore the first two threads: only needed to copy ghost cells for columns </span>
  <span class="k">if</span><span class="p">(</span><span class="n">col</span> <span class="o">&gt;</span> <span class="mi">1</span><span class="p">)</span> <span class="p">{</span>
    <span class="k">if</span><span class="p">(</span><span class="n">row</span> <span class="o">==</span> <span class="mi">0</span><span class="p">)</span> <span class="p">{</span>
      <span class="c1">// Copy first real row to bottom ghost row</span>
      <span class="n">world</span><span class="p">[</span><span class="n">col</span><span class="o">-</span><span class="mi">1</span> <span class="o">+</span> <span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">)</span><span class="o">*</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">1</span><span class="p">)]</span> <span class="o">=</span> <span class="n">world</span><span class="p">[(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">)</span>     <span class="o">+</span> <span class="n">col</span><span class="o">-</span><span class="mi">1</span><span class="p">];</span>
    <span class="p">}</span><span class="k">else</span><span class="p">{</span>
      <span class="c1">// Copy last real row to top ghost row</span>
      <span class="n">world</span><span class="p">[</span><span class="n">col</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>                   <span class="o">=</span> <span class="n">world</span><span class="p">[(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">)</span><span class="o">*</span><span class="n">dim</span> <span class="o">+</span> <span class="n">col</span><span class="o">-</span><span class="mi">1</span><span class="p">];</span>
    <span class="p">}</span>
  <span class="p">}</span>
  <span class="n">__syncthreads</span><span class="p">();</span>

  <span class="k">if</span><span class="p">(</span><span class="n">row</span> <span class="o">==</span> <span class="mi">0</span><span class="p">)</span> <span class="p">{</span>
    <span class="c1">// Copy first real column to right most ghost column</span>
    <span class="n">world</span><span class="p">[</span><span class="n">col</span><span class="o">*</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">)</span><span class="o">+</span><span class="n">dim</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="n">world</span><span class="p">[</span><span class="n">col</span><span class="o">*</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">)</span> <span class="o">+</span> <span class="mi">1</span><span class="p">];</span>
  <span class="p">}</span> <span class="k">else</span> <span class="p">{</span>
    <span class="c1">// Copy last real column to left most ghost column</span>
    <span class="n">world</span><span class="p">[</span><span class="n">col</span><span class="o">*</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">)</span>      <span class="p">]</span> <span class="o">=</span> <span class="n">world</span><span class="p">[</span><span class="n">col</span><span class="o">*</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">)</span> <span class="o">+</span> <span class="n">dim</span><span class="p">];</span>
  <span class="p">}</span>

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
              <pre><span></span><span class="c1">// main kernel which calculates an iteration of game of life</span>
<span class="n">__global__</span> <span class="kt">void</span> <span class="nf">GOL_GPU</span><span class="p">(</span><span class="kt">int</span> <span class="n">dim</span><span class="p">,</span> <span class="kt">int</span> <span class="o">*</span><span class="n">world</span><span class="p">,</span> <span class="kt">int</span> <span class="o">*</span><span class="n">newWorld</span><span class="p">)</span> <span class="p">{</span>
   <span class="kt">int</span> <span class="n">row</span> <span class="o">=</span> <span class="n">blockIdx</span><span class="p">.</span><span class="n">y</span> <span class="o">*</span> <span class="n">blockDim</span><span class="p">.</span><span class="n">y</span> <span class="o">+</span> <span class="n">threadIdx</span><span class="p">.</span><span class="n">y</span> <span class="o">+</span> <span class="mi">1</span><span class="p">;</span>
   <span class="kt">int</span> <span class="n">col</span> <span class="o">=</span> <span class="n">blockIdx</span><span class="p">.</span><span class="n">x</span> <span class="o">*</span> <span class="n">blockDim</span><span class="p">.</span><span class="n">x</span> <span class="o">+</span> <span class="n">threadIdx</span><span class="p">.</span><span class="n">x</span> <span class="o">+</span> <span class="mi">1</span><span class="p">;</span>
   <span class="kt">int</span> <span class="n">id</span> <span class="o">=</span> <span class="n">row</span><span class="o">*</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">)</span> <span class="o">+</span> <span class="n">col</span><span class="p">;</span>

   <span class="kt">int</span> <span class="n">numNeighbors</span><span class="p">;</span>
   <span class="kt">int</span> <span class="n">cell</span> <span class="o">=</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="p">];</span>

   <span class="n">numNeighbors</span> <span class="o">=</span>   <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">+</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">)]</span>   <span class="c1">// lower</span>
      <span class="o">+</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">-</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">)]</span>               <span class="c1">// upper</span>
      <span class="o">+</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span>                     <span class="c1">// right</span>
      <span class="o">+</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>                     <span class="c1">// left</span>

      <span class="o">+</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">+</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">3</span><span class="p">)]</span>   <span class="c1">// diagonal lower right</span>
      <span class="o">+</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">-</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">3</span><span class="p">)]</span>   <span class="c1">// diagonal upper left</span>
      <span class="o">+</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">-</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">1</span><span class="p">)]</span>   <span class="c1">// diagonal upper right</span>
      <span class="o">+</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">+</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">1</span><span class="p">)];</span>  <span class="c1">// diagonal lower left</span>

   <span class="k">if</span> <span class="p">(</span><span class="n">cell</span> <span class="o">==</span> <span class="mi">1</span> <span class="o">&amp;&amp;</span> <span class="n">numNeighbors</span> <span class="o">&lt;</span> <span class="mi">2</span><span class="p">)</span>
      <span class="n">newWorld</span><span class="p">[</span><span class="n">id</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span>

    <span class="c1">// 2) Any live cell with two or three live neighbours lives</span>
    <span class="k">else</span> <span class="k">if</span> <span class="p">(</span><span class="n">cell</span> <span class="o">==</span> <span class="mi">1</span> <span class="o">&amp;&amp;</span> <span class="p">(</span><span class="n">numNeighbors</span> <span class="o">==</span> <span class="mi">2</span> <span class="o">||</span> <span class="n">numNeighbors</span> <span class="o">==</span> <span class="mi">3</span><span class="p">))</span>
      <span class="n">newWorld</span><span class="p">[</span><span class="n">id</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span><span class="p">;</span>

    <span class="c1">// 3) Any live cell with more than three live neighbours dies</span>
    <span class="k">else</span> <span class="k">if</span> <span class="p">(</span><span class="n">cell</span> <span class="o">==</span> <span class="mi">1</span> <span class="o">&amp;&amp;</span> <span class="n">numNeighbors</span> <span class="o">&gt;</span> <span class="mi">3</span><span class="p">)</span>
      <span class="n">newWorld</span><span class="p">[</span><span class="n">id</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span>

    <span class="c1">// 4) Any dead cell with exactly three live neighbours becomes a live cell</span>
    <span class="k">else</span> <span class="k">if</span> <span class="p">(</span><span class="n">cell</span> <span class="o">==</span> <span class="mi">0</span> <span class="o">&amp;&amp;</span> <span class="n">numNeighbors</span> <span class="o">==</span> <span class="mi">3</span><span class="p">)</span>
      <span class="n">newWorld</span><span class="p">[</span><span class="n">id</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span><span class="p">;</span>

    <span class="k">else</span>
      <span class="n">newWorld</span><span class="p">[</span><span class="n">id</span><span class="p">]</span> <span class="o">=</span> <span class="n">cell</span><span class="p">;</span>
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
          <h2 id="Game-of-Life-Setup">Game of Life Setup<a class="anchor-link" href="#Game-of-Life-Setup">&#182;</a></h2>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[8]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// size of the world</span>
<span class="k">const</span> <span class="kt">unsigned</span> <span class="kt">int</span> <span class="n">dim</span> <span class="o">=</span> <span class="mi">10</span><span class="p">;</span>
<span class="c1">// two extra columns and rows for ghostcells</span>
<span class="k">const</span> <span class="kt">unsigned</span> <span class="kt">int</span> <span class="n">world_size</span> <span class="o">=</span> <span class="n">dim</span> <span class="o">+</span> <span class="mi">2</span><span class="p">;</span>
<span class="kt">unsigned</span> <span class="kt">int</span> <span class="n">iterations</span> <span class="o">=</span> <span class="mi">5</span><span class="p">;</span>

<span class="c1">// pointer for host and device memory </span>
<span class="kt">int</span> <span class="o">*</span> <span class="n">sim_world</span><span class="p">;</span>
<span class="kt">int</span> <span class="o">*</span> <span class="n">d_sim_world</span><span class="p">;</span>
<span class="kt">int</span> <span class="o">*</span> <span class="n">d_new_sim_world</span><span class="p">;</span>
<span class="kt">int</span> <span class="o">*</span> <span class="n">d_swap</span><span class="p">;</span>

<span class="c1">// saves the images of each simulation step</span>
<span class="n">std</span><span class="o">::</span><span class="n">vector</span><span class="o">&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">vector</span><span class="o">&lt;</span> <span class="kt">unsigned</span> <span class="kt">char</span> <span class="o">&gt;</span> <span class="o">&gt;</span> <span class="n">sim_pngs</span><span class="p">;</span>
<span class="c1">// describe the color of a living or dead cell</span>
<span class="n">BlackWhiteMap</span><span class="o">&lt;</span><span class="kt">int</span><span class="o">&gt;</span> <span class="n">bw_map</span><span class="p">;</span>
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[9]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// allocate memory on CPU and GPU</span>
<span class="n">sim_world</span> <span class="o">=</span> <span class="k">new</span> <span class="kt">int</span><span class="p">[</span> <span class="n">world_size</span> <span class="o">*</span> <span class="n">world_size</span> <span class="p">];</span>
<span class="n">cuCheck</span><span class="p">(</span><span class="n">cudaMalloc</span><span class="p">(</span> <span class="p">(</span><span class="kt">void</span> <span class="o">**</span><span class="p">)</span> <span class="o">&amp;</span><span class="n">d_sim_world</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="kt">int</span><span class="p">)</span><span class="o">*</span><span class="n">world_size</span><span class="o">*</span><span class="n">world_size</span><span class="p">));</span>
<span class="n">cuCheck</span><span class="p">(</span><span class="n">cudaMalloc</span><span class="p">(</span> <span class="p">(</span><span class="kt">void</span> <span class="o">**</span><span class="p">)</span> <span class="o">&amp;</span><span class="n">d_new_sim_world</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="kt">int</span><span class="p">)</span><span class="o">*</span><span class="n">world_size</span><span class="o">*</span><span class="n">world_size</span><span class="p">));</span>
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
          <h2 id="Interactive-Input">Interactive Input<a class="anchor-link" href="#Interactive-Input">&#182;</a></h2>
          <ul>
            <li>Jupyter Notebook offers "magic" commands that provide language-independent functions</li>
            <li>magic commands starts with <code>%%</code></li>
            <li>
              <code>%%file [name]</code> writes the contents of a cell to a file
              <ul>
                <li>the file is stored in the same folder as the notebook and can be loaded via C/C++ functions </li>
              </ul>
            </li>
            <li>depends on the language kernel</li>
          </ul>
          <p>Define the initial world for the Game-of-Life simulation. <code>X</code> are living cells and <code>0</code> are dead.</p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[10]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="o">%%</span><span class="n">file</span> <span class="n">input</span><span class="p">.</span><span class="n">txt</span>
<span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span>
<span class="mi">0</span> <span class="mi">0</span> <span class="n">X</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span>
<span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="n">X</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span>
<span class="mi">0</span> <span class="n">X</span> <span class="n">X</span> <span class="n">X</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span>
<span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span>
<span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span>
<span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span>
<span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span>
<span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span>
<span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span>
</pre>
            </div>
          </div>
        </div>
      </div>
      <div class="output_wrapper">
        <div class="output">
          <div class="output_area">
            <div class="prompt"></div>
            <div class="output_subarea output_stream output_stdout output_text">
              <pre>Overwriting input.txt
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
          <h2 id="Cling-I/O-System">Cling I/O-System<a class="anchor-link" href="#Cling-I/O-System">&#182;</a></h2>
          <ul>
            <li>read_input() reads the initial world from a file and returns an error code </li>
            <li>
              Return values:
              <ul>
                <li>0 = success</li>
                <li>-1 = file cannot be opened</li>
                <li>-2 = too many elements in file -&gt; extra elements are ignored</li>
              </ul>
            </li>
          </ul>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[11]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="n">read_input</span><span class="p">(</span><span class="s">&quot;input.txt&quot;</span><span class="p">,</span> <span class="n">sim_world</span><span class="p">,</span> <span class="n">dim</span><span class="p">,</span> <span class="n">dim</span><span class="p">,</span> <span class="nb">true</span><span class="p">)</span>
</pre>
            </div>
          </div>
        </div>
      </div>
      <div class="output_wrapper">
        <div class="output">
          <div class="output_area">
            <div class="prompt output_prompt">Out[11]:</div>
            <div class="output_text output_subarea output_execute_result">
              <pre>0</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[12]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="n">cuCheck</span><span class="p">(</span><span class="n">cudaMemcpy</span><span class="p">(</span><span class="n">d_sim_world</span><span class="p">,</span> <span class="n">sim_world</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="kt">int</span><span class="p">)</span><span class="o">*</span><span class="n">world_size</span><span class="o">*</span><span class="n">world_size</span><span class="p">,</span> <span class="n">cudaMemcpyHostToDevice</span><span class="p">));</span>
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[13]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// create an image of the initial world</span>
<span class="n">sim_pngs</span><span class="p">.</span><span class="n">push_back</span><span class="p">(</span><span class="n">generate_png</span><span class="o">&lt;</span><span class="kt">int</span><span class="o">&gt;</span><span class="p">(</span><span class="n">sim_world</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">bw_map</span><span class="p">,</span> <span class="nb">true</span><span class="p">,</span> <span class="mi">20</span><span class="p">));</span>
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
          <h2 id="Interactive-Simulation:-Main-Loop">Interactive Simulation: Main Loop<a class="anchor-link" href="#Interactive-Simulation:-Main-Loop">&#182;</a></h2>
          <ul>
            <li>calculate the game of life</li>
          </ul>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[17]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// main loop</span>
<span class="k">for</span><span class="p">(</span><span class="kt">unsigned</span> <span class="kt">int</span> <span class="n">i</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span> <span class="n">i</span> <span class="o">&lt;</span> <span class="n">iterations</span><span class="p">;</span> <span class="o">++</span><span class="n">i</span><span class="p">)</span> <span class="p">{</span>

    <span class="n">copy_ghostcells</span><span class="o">&lt;&lt;&lt;</span><span class="mi">1</span><span class="p">,</span> <span class="n">dim3</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span><span class="o">&gt;&gt;&gt;</span><span class="p">(</span><span class="n">dim</span><span class="p">,</span> <span class="n">d_sim_world</span><span class="p">);</span>
    <span class="n">GOL_GPU</span><span class="o">&lt;&lt;&lt;</span><span class="mi">1</span><span class="p">,</span> <span class="n">dim3</span><span class="p">(</span><span class="n">dim</span><span class="p">,</span> <span class="n">dim</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span><span class="o">&gt;&gt;&gt;</span><span class="p">(</span><span class="n">dim</span><span class="p">,</span> <span class="n">d_sim_world</span><span class="p">,</span> <span class="n">d_new_sim_world</span><span class="p">);</span>
    <span class="n">cuCheck</span><span class="p">(</span><span class="n">cudaDeviceSynchronize</span><span class="p">());</span>

    <span class="n">d_swap</span> <span class="o">=</span> <span class="n">d_new_sim_world</span><span class="p">;</span>
    <span class="n">d_new_sim_world</span> <span class="o">=</span> <span class="n">d_sim_world</span><span class="p">;</span>
    <span class="n">d_sim_world</span> <span class="o">=</span> <span class="n">d_swap</span><span class="p">;</span>

    <span class="n">cuCheck</span><span class="p">(</span><span class="n">cudaMemcpy</span><span class="p">(</span><span class="n">sim_world</span><span class="p">,</span> <span class="n">d_sim_world</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="kt">int</span><span class="p">)</span><span class="o">*</span><span class="n">world_size</span><span class="o">*</span><span class="n">world_size</span><span class="p">,</span> <span class="n">cudaMemcpyDeviceToHost</span><span class="p">));</span>
    <span class="n">sim_pngs</span><span class="p">.</span><span class="n">push_back</span><span class="p">(</span><span class="n">generate_png</span><span class="o">&lt;</span><span class="kt">int</span><span class="o">&gt;</span><span class="p">(</span><span class="n">sim_world</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">bw_map</span><span class="p">,</span> <span class="nb">true</span><span class="p">,</span> <span class="mi">20</span><span class="p">));</span>
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
          <h2 id="Display-Simulation-Images">Display Simulation Images<a class="anchor-link" href="#Display-Simulation-Images">&#182;</a></h2>
          <ul>
            <li>xeus-cling offers a built-in C++ library for displaying media content in notebooks</li>
            <li>see xeus-cling <a href="https://github.com/QuantStack/xeus-cling#trying-it-online">example</a></li>
          </ul>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[18]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="k">for</span><span class="p">(</span><span class="kt">int</span> <span class="n">i</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span> <span class="n">i</span> <span class="o">&lt;</span> <span class="n">sim_pngs</span><span class="p">.</span><span class="n">size</span><span class="p">();</span> <span class="o">++</span><span class="n">i</span><span class="p">)</span> <span class="p">{</span>
    <span class="n">display_image</span><span class="p">(</span><span class="n">sim_pngs</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="nb">true</span><span class="p">);</span>
    <span class="n">std</span><span class="o">::</span><span class="n">cout</span> <span class="o">&lt;&lt;</span> <span class="s">&quot;iteration = &quot;</span> <span class="o">&lt;&lt;</span> <span class="n">i</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span>
    <span class="n">std</span><span class="o">::</span><span class="n">this_thread</span><span class="o">::</span><span class="n">sleep_for</span><span class="p">(</span><span class="n">std</span><span class="o">::</span><span class="n">chrono</span><span class="o">::</span><span class="n">milliseconds</span><span class="p">(</span><span class="mi">800</span><span class="p">));</span>
<span class="p">}</span>
</pre>
            </div>
          </div>
        </div>
      </div>
      <div class="output_wrapper">
        <div class="output">
          <div class="output_area">
            <div class="prompt"></div>
            <div class="output_png output_subarea ">
              <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADIEAIAAAByquWKAAAABGdBTUEAAOpgYgYacwAAAAd0SU1FB+ULCAkXBDUGQm8AAAANdEVYdFRpdGxlAHRtcC5wbme+kCfgAAAAJ3RFWHRBdXRob3IAUE5Hd3JpdGVyIEF1dGhvcjogUGF1bCBCbGFja2J1cm692N2jAAAAMnRFWHREZXNjcmlwdGlvbgBodHRwczovL2dpdGh1Yi5jb20vcG5nd3JpdGVyL3BuZ3dyaXRlcto6+2MAAAA0dEVYdFNvZnR3YXJlAFBOR3dyaXRlcjogQW4gZWFzeSB0byB1c2UgZ3JhcGhpY3MgbGlicmFyeS5FhlbUAAAAJ3RFWHRDcmVhdGlvbiBUaW1lADggTm92IDIwMjEgMDk6MjM6MDQgKzAwMDArn5JqAAABp0lEQVR4nO3csQ3DMAwAQTPI/isrfaoUVoQ37hYQC+HBitcFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA3GlOD8Aua6211v/fnZkZ/4otXqcHAPiVYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZLi9zc3ckmcfGxaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARnv0wM836kb5/A8NiwgQ7CADMECMgQLyBAsIEOwgAzBAjIEC8gQLCBDsIAMwQIyBAvIECwgQ7CADMECMgQLyBAsIEOwgAzBAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPjyAc9hEnPC90PkAAAAAElFTkSuQmCC"
                >
            </div>
          </div>
          <div class="output_area">
            <div class="prompt"></div>
            <div class="output_subarea output_stream output_stdout output_text">
              <pre>iteration = 8
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
          <h2 id="Nonlinear-Program-Flow">Nonlinear Program Flow<a class="anchor-link" href="#Nonlinear-Program-Flow">&#182;</a></h2>
          <p>Jupyter Notebook enables nonlinear program execution. You can execute a cell again. The result may vary depending on the source code and the state of the runtime.</p>
          <p>For example, if you repeat the main loop of the simulation, the simulation continues because the state of the simulation is in the 5th iteration. If you run the cell again, you calculate step 6 to 10 of the simulation. You can also change cell variables. Simply set the <code>iterations</code> variable to <code>3</code>, run the <a href="#Interactive-Simulation%3A-Main-Loop">main loop</a> and the visualization cell again and see what happens.</p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[16]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="n">iterations</span> <span class="o">=</span> <span class="mi">3</span><span class="p">;</span>
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
          <h1 id="In-Situ-Data-Analysis">In-Situ Data Analysis<a class="anchor-link" href="#In-Situ-Data-Analysis">&#182;</a></h1>
          <p>After the simulation, the results must be analyzed. Often it is processed by an additional process, which means that you have to write your data to disk and reload it. Depending on the simulation, it can take a long time. Alternatively, you can integrate your analysis into the simulation. Then you don't need the time to save and load the data, but you need to know what you want to analyze before running the simulation. If you want to perform another analysis, e.g. because you get new insights from a previous analysis, you have to run the simulation again.</p>
          <p>Cling can combine the advantages of both methods. You can add an analysis at runtime and analyze the simulation data without copying it.</p>
          <p><strong>Add a new analysis</strong></p>
          <p>Count the neighbors of a cell and display them as a heat map.</p>
          <ul>
            <li>persistent simulation data <em>on the GPU</em></li>
            <li>add analysis <em>on-the-fly</em> and <em>in-memory</em></li>
          </ul>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing text_cell rendered">
      <div class="prompt input_prompt"></div>
      <div class="inner_cell">
        <div class="text_cell_render border-box-sizing rendered_html">
          <h2 id="Dynamic-Extension-Without-Loss-of-State">Dynamic Extension Without Loss of State<a class="anchor-link" href="#Dynamic-Extension-Without-Loss-of-State">&#182;</a></h2>
          <p>Define an analysis kernel on-the-fly.</p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[19]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// counts the neighbors of a cell</span>
<span class="n">__global__</span> <span class="kt">void</span> <span class="nf">get_num_neighbors</span><span class="p">(</span><span class="kt">int</span> <span class="n">dim</span><span class="p">,</span> <span class="kt">int</span> <span class="o">*</span><span class="n">world</span><span class="p">,</span> <span class="kt">int</span> <span class="o">*</span><span class="n">newWorld</span><span class="p">)</span> <span class="p">{</span>
   <span class="kt">int</span> <span class="n">row</span> <span class="o">=</span> <span class="n">blockIdx</span><span class="p">.</span><span class="n">y</span> <span class="o">*</span> <span class="n">blockDim</span><span class="p">.</span><span class="n">y</span> <span class="o">+</span> <span class="n">threadIdx</span><span class="p">.</span><span class="n">y</span> <span class="o">+</span> <span class="mi">1</span><span class="p">;</span>
   <span class="kt">int</span> <span class="n">col</span> <span class="o">=</span> <span class="n">blockIdx</span><span class="p">.</span><span class="n">x</span> <span class="o">*</span> <span class="n">blockDim</span><span class="p">.</span><span class="n">x</span> <span class="o">+</span> <span class="n">threadIdx</span><span class="p">.</span><span class="n">x</span> <span class="o">+</span> <span class="mi">1</span><span class="p">;</span>
   <span class="kt">int</span> <span class="n">id</span> <span class="o">=</span> <span class="n">row</span><span class="o">*</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">)</span> <span class="o">+</span> <span class="n">col</span><span class="p">;</span>

   <span class="n">newWorld</span><span class="p">[</span><span class="n">id</span><span class="p">]</span> <span class="o">=</span>   <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">+</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">)]</span>   <span class="c1">// lower</span>
      <span class="o">+</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">-</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">)]</span>               <span class="c1">// upper</span>
      <span class="o">+</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span>                     <span class="c1">// right</span>
      <span class="o">+</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>                     <span class="c1">// left</span>

      <span class="o">+</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">+</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">3</span><span class="p">)]</span>   <span class="c1">// diagonal lower right</span>
      <span class="o">+</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">-</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">3</span><span class="p">)]</span>   <span class="c1">// diagonal upper left</span>
      <span class="o">+</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">-</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">1</span><span class="p">)]</span>   <span class="c1">// diagonal upper right</span>
      <span class="o">+</span> <span class="n">world</span><span class="p">[</span><span class="n">id</span><span class="o">+</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">1</span><span class="p">)];</span>  <span class="c1">// diagonal lower left</span>
<span class="p">}</span>
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[20]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// allocate extra memory on the GPU to ouput the analysis</span>
<span class="kt">int</span> <span class="o">*</span> <span class="n">d_ana_world</span><span class="p">;</span>
<span class="n">cuCheck</span><span class="p">(</span><span class="n">cudaMalloc</span><span class="p">(</span> <span class="p">(</span><span class="kt">void</span> <span class="o">**</span><span class="p">)</span> <span class="o">&amp;</span><span class="n">d_ana_world</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="kt">int</span><span class="p">)</span><span class="o">*</span><span class="n">world_size</span><span class="o">*</span><span class="n">world_size</span><span class="p">));</span>

<span class="c1">// allocate memory on CPU to generate an image</span>
<span class="n">std</span><span class="o">::</span><span class="n">vector</span><span class="o">&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">vector</span><span class="o">&lt;</span> <span class="kt">unsigned</span> <span class="kt">char</span> <span class="o">&gt;</span> <span class="o">&gt;</span> <span class="n">ana_pngs</span><span class="p">;</span>
<span class="kt">int</span> <span class="o">*</span> <span class="n">ana_world</span> <span class="o">=</span> <span class="k">new</span> <span class="kt">int</span><span class="p">[</span><span class="n">world_size</span><span class="o">*</span><span class="n">world_size</span><span class="p">];</span>
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[21]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// run the analysis</span>
<span class="c1">// uuse the simulation data as input and write the result into an extra memory</span>
<span class="n">get_num_neighbors</span><span class="o">&lt;&lt;&lt;</span><span class="mi">1</span><span class="p">,</span><span class="n">dim3</span><span class="p">(</span><span class="n">dim</span><span class="p">,</span> <span class="n">dim</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span><span class="o">&gt;&gt;&gt;</span><span class="p">(</span><span class="n">dim</span><span class="p">,</span> <span class="n">d_sim_world</span><span class="p">,</span> <span class="n">d_ana_world</span><span class="p">);</span>
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[22]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// copy analysis data to the CPU</span>
<span class="n">cuCheck</span><span class="p">(</span><span class="n">cudaMemcpy</span><span class="p">(</span><span class="n">ana_world</span><span class="p">,</span> <span class="n">d_ana_world</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="kt">int</span><span class="p">)</span><span class="o">*</span><span class="n">world_size</span><span class="o">*</span><span class="n">world_size</span><span class="p">,</span> <span class="n">cudaMemcpyDeviceToHost</span><span class="p">));</span>
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[23]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// define a color map for the heat map</span>
<span class="c1">// use the same code which has generated images of the game of life world</span>
<span class="k">template</span> <span class="o">&lt;</span><span class="k">typename</span> <span class="n">varTyp</span><span class="o">&gt;</span>
<span class="k">struct</span> <span class="nl">HeatMap</span> <span class="p">:</span> <span class="n">ColorMap</span><span class="o">&lt;</span><span class="n">varTyp</span><span class="o">&gt;</span>
<span class="p">{</span>
    <span class="kt">int</span> <span class="n">r</span><span class="p">(</span><span class="n">varTyp</span> <span class="n">value</span><span class="p">){</span><span class="k">return</span> <span class="n">value</span> <span class="o">*</span> <span class="mi">65535</span><span class="o">/</span><span class="mi">8</span><span class="p">;}</span>
    <span class="kt">int</span> <span class="n">g</span><span class="p">(</span><span class="n">varTyp</span> <span class="n">value</span><span class="p">){</span><span class="k">return</span> <span class="mi">0</span><span class="p">;}</span>
    <span class="kt">int</span> <span class="n">b</span><span class="p">(</span><span class="n">varTyp</span> <span class="n">value</span><span class="p">){</span><span class="k">return</span> <span class="mi">0</span><span class="p">;}</span>
<span class="p">};</span>
<span class="n">HeatMap</span><span class="o">&lt;</span><span class="kt">int</span><span class="o">&gt;</span> <span class="n">h_map</span><span class="p">;</span>
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[25]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// generate a heat map image</span>
<span class="n">ana_pngs</span><span class="p">.</span><span class="n">push_back</span><span class="p">(</span><span class="n">generate_png</span><span class="o">&lt;</span><span class="kt">int</span><span class="o">&gt;</span><span class="p">(</span><span class="n">ana_world</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">h_map</span><span class="p">,</span> <span class="nb">true</span><span class="p">,</span> <span class="mi">20</span><span class="p">));</span>
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[18]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="k">for</span><span class="p">(</span><span class="kt">int</span> <span class="n">i</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span> <span class="n">i</span> <span class="o">&lt;</span> <span class="n">sim_pngs</span><span class="p">.</span><span class="n">size</span><span class="p">();</span> <span class="o">++</span><span class="n">i</span><span class="p">)</span> <span class="p">{</span>
    <span class="n">display_image</span><span class="p">(</span><span class="n">sim_pngs</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="nb">true</span><span class="p">);</span>
    <span class="n">std</span><span class="o">::</span><span class="n">cout</span> <span class="o">&lt;&lt;</span> <span class="s">&quot;iteration = &quot;</span> <span class="o">&lt;&lt;</span> <span class="n">i</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span>
    <span class="n">std</span><span class="o">::</span><span class="n">this_thread</span><span class="o">::</span><span class="n">sleep_for</span><span class="p">(</span><span class="n">std</span><span class="o">::</span><span class="n">chrono</span><span class="o">::</span><span class="n">milliseconds</span><span class="p">(</span><span class="mi">800</span><span class="p">));</span>
<span class="p">}</span>
</pre>
            </div>
          </div>
        </div>
      </div>
      <div class="output_wrapper">
        <div class="output">
          <div class="output_area">
            <div class="prompt"></div>
            <div class="output_png output_subarea ">
              <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADIEAIAAAByquWKAAAABGdBTUEAAOpgYgYacwAAAAd0SU1FB+ULCAkXBDUGQm8AAAANdEVYdFRpdGxlAHRtcC5wbme+kCfgAAAAJ3RFWHRBdXRob3IAUE5Hd3JpdGVyIEF1dGhvcjogUGF1bCBCbGFja2J1cm692N2jAAAAMnRFWHREZXNjcmlwdGlvbgBodHRwczovL2dpdGh1Yi5jb20vcG5nd3JpdGVyL3BuZ3dyaXRlcto6+2MAAAA0dEVYdFNvZnR3YXJlAFBOR3dyaXRlcjogQW4gZWFzeSB0byB1c2UgZ3JhcGhpY3MgbGlicmFyeS5FhlbUAAAAJ3RFWHRDcmVhdGlvbiBUaW1lADggTm92IDIwMjEgMDk6MjM6MDQgKzAwMDArn5JqAAABp0lEQVR4nO3csQ3DMAwAQTPI/isrfaoUVoQ37hYQC+HBitcFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA3GlOD8Aua6211v/fnZkZ/4otXqcHAPiVYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZLi9zc3ckmcfGxaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARnv0wM836kb5/A8NiwgQ7CADMECMgQLyBAsIEOwgAzBAjIEC8gQLCBDsIAMwQIyBAvIECwgQ7CADMECMgQLyBAsIEOwgAzBAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPjyAc9hEnPC90PkAAAAAElFTkSuQmCC"
                >
            </div>
          </div>
          <div class="output_area">
            <div class="prompt"></div>
            <div class="output_subarea output_stream output_stdout output_text">
              <pre>iteration = 8
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[26]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="k">for</span><span class="p">(</span><span class="kt">int</span> <span class="n">i</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span> <span class="n">i</span> <span class="o">&lt;</span> <span class="n">ana_pngs</span><span class="p">.</span><span class="n">size</span><span class="p">();</span> <span class="o">++</span><span class="n">i</span><span class="p">)</span> <span class="p">{</span>
    <span class="n">display_image</span><span class="p">(</span><span class="n">ana_pngs</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="nb">true</span><span class="p">);</span>
    <span class="n">std</span><span class="o">::</span><span class="n">cout</span> <span class="o">&lt;&lt;</span> <span class="s">&quot;iteration = &quot;</span> <span class="o">&lt;&lt;</span> <span class="n">i</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span>
    <span class="n">std</span><span class="o">::</span><span class="n">this_thread</span><span class="o">::</span><span class="n">sleep_for</span><span class="p">(</span><span class="n">std</span><span class="o">::</span><span class="n">chrono</span><span class="o">::</span><span class="n">milliseconds</span><span class="p">(</span><span class="mi">800</span><span class="p">));</span>
<span class="p">}</span>
</pre>
            </div>
          </div>
        </div>
      </div>
      <div class="output_wrapper">
        <div class="output">
          <div class="output_area">
            <div class="prompt"></div>
            <div class="output_png output_subarea ">
              <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADIEAIAAAByquWKAAAABGdBTUEAAOpgYgYacwAAAAd0SU1FB+ULCAkXDtXTq3EAAAANdEVYdFRpdGxlAHRtcC5wbme+kCfgAAAAJ3RFWHRBdXRob3IAUE5Hd3JpdGVyIEF1dGhvcjogUGF1bCBCbGFja2J1cm692N2jAAAAMnRFWHREZXNjcmlwdGlvbgBodHRwczovL2dpdGh1Yi5jb20vcG5nd3JpdGVyL3BuZ3dyaXRlcto6+2MAAAA0dEVYdFNvZnR3YXJlAFBOR3dyaXRlcjogQW4gZWFzeSB0byB1c2UgZ3JhcGhpY3MgbGlicmFyeS5FhlbUAAAAJ3RFWHRDcmVhdGlvbiBUaW1lADggTm92IDIwMjEgMDk6MjM6MTQgKzAwMDDnNZL0AAACJElEQVR4nO3csQ0CQQwAQUD0AaVQ+pcClUALEPDW/s00YAen1UU+nQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgd+fpBY7v9p7eYA0vb3kBl+kFAL4lWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGdfpBTia+9Tgodv5bsnvyQ8LyBAsIEOwgAzBAjIEC8gQLCBDsIAMwQIyBAvIECwgQ7CADMECMgQLyBAsIEOwgAzBAjIEC8gQLCBDsICMhW6634Zufj9mxo7N3YbmTnlNL7AUPywgQ7CADMECMgQLyBAsIEOwgAzBAjIEC8gQLCBDsIAMwQIyBAvIECwgQ7CADMECMgQLyBAsIEOwgAzBAjIWuuk+ZZteYGfP6QU4MD8sIEOwgAzBAjIEC8gQLCBDsIAMwQIyBAvIECwgQ7CADMECMgQLyBAsIEOwgAzBAjIEC8gQLCBDsIAMwQIy3HT/u/v0Aot4Ti/ADvywgAzBAjIEC8gQLCBDsIAMwQIyBAvIECwgQ7CADMECMgQLyBAsIEOwgAzBAjIEC8gQLCBDsIAMwQIyBAsAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFvUBrzIKsP5U2JsAAAAASUVORK5CYII="
                >
            </div>
          </div>
          <div class="output_area">
            <div class="prompt"></div>
            <div class="output_subarea output_stream output_stdout output_text">
              <pre>iteration = 0
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
          <h1 id="Continue-Simulation-Analysis-Loop">Continue Simulation-Analysis-Loop<a class="anchor-link" href="#Continue-Simulation-Analysis-Loop">&#182;</a></h1>
          <p>You have completed your first iteration of the analysis. Now you can continue the simulation with</p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="n">iterations</span> <span class="o">=</span> <span class="mi">3</span><span class="p">;</span>
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
          <p>steps and run the main loop and the neighborhood analysis again</p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// copy together the content of different cells, to avoid extra navigation</span>

<span class="c1">// main loop</span>
<span class="k">for</span><span class="p">(</span><span class="kt">unsigned</span> <span class="kt">int</span> <span class="n">i</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span> <span class="n">i</span> <span class="o">&lt;</span> <span class="n">iterations</span><span class="p">;</span> <span class="o">++</span><span class="n">i</span><span class="p">)</span> <span class="p">{</span>

    <span class="n">copy_ghostcells</span><span class="o">&lt;&lt;&lt;</span><span class="mi">1</span><span class="p">,</span> <span class="n">dim3</span><span class="p">(</span><span class="n">dim</span><span class="o">+</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span><span class="o">&gt;&gt;&gt;</span><span class="p">(</span><span class="n">dim</span><span class="p">,</span> <span class="n">d_sim_world</span><span class="p">);</span>
    <span class="n">GOL_GPU</span><span class="o">&lt;&lt;&lt;</span><span class="mi">1</span><span class="p">,</span> <span class="n">dim3</span><span class="p">(</span><span class="n">dim</span><span class="p">,</span> <span class="n">dim</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span><span class="o">&gt;&gt;&gt;</span><span class="p">(</span><span class="n">dim</span><span class="p">,</span> <span class="n">d_sim_world</span><span class="p">,</span> <span class="n">d_new_sim_world</span><span class="p">);</span>
    <span class="n">cuCheck</span><span class="p">(</span><span class="n">cudaDeviceSynchronize</span><span class="p">());</span>

    <span class="n">d_swap</span> <span class="o">=</span> <span class="n">d_new_sim_world</span><span class="p">;</span>
    <span class="n">d_new_sim_world</span> <span class="o">=</span> <span class="n">d_sim_world</span><span class="p">;</span>
    <span class="n">d_sim_world</span> <span class="o">=</span> <span class="n">d_swap</span><span class="p">;</span>

    <span class="n">cuCheck</span><span class="p">(</span><span class="n">cudaMemcpy</span><span class="p">(</span><span class="n">sim_world</span><span class="p">,</span> <span class="n">d_sim_world</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="kt">int</span><span class="p">)</span><span class="o">*</span><span class="n">world_size</span><span class="o">*</span><span class="n">world_size</span><span class="p">,</span> <span class="n">cudaMemcpyDeviceToHost</span><span class="p">));</span>
    <span class="n">sim_pngs</span><span class="p">.</span><span class="n">push_back</span><span class="p">(</span><span class="n">generate_png</span><span class="o">&lt;</span><span class="kt">int</span><span class="o">&gt;</span><span class="p">(</span><span class="n">sim_world</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">bw_map</span><span class="p">,</span> <span class="nb">true</span><span class="p">,</span> <span class="mi">20</span><span class="p">));</span>
<span class="p">}</span>

<span class="c1">// run the analysis</span>
<span class="c1">// use the simulation data as input and write the result to an extra chuck of memory</span>
<span class="n">get_num_neighbors</span><span class="o">&lt;&lt;&lt;</span><span class="mi">1</span><span class="p">,</span><span class="n">dim3</span><span class="p">(</span><span class="n">dim</span><span class="p">,</span> <span class="n">dim</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span><span class="o">&gt;&gt;&gt;</span><span class="p">(</span><span class="n">dim</span><span class="p">,</span> <span class="n">d_sim_world</span><span class="p">,</span> <span class="n">d_ana_world</span><span class="p">);</span>

<span class="c1">// copy analysis data to CPU</span>
<span class="n">cuCheck</span><span class="p">(</span><span class="n">cudaMemcpy</span><span class="p">(</span><span class="n">ana_world</span><span class="p">,</span> <span class="n">d_ana_world</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="kt">int</span><span class="p">)</span><span class="o">*</span><span class="n">world_size</span><span class="o">*</span><span class="n">world_size</span><span class="p">,</span> <span class="n">cudaMemcpyDeviceToHost</span><span class="p">));</span>

<span class="n">ana_pngs</span><span class="p">.</span><span class="n">clear</span><span class="p">();</span>

<span class="c1">// generate heat map image</span>
<span class="n">ana_pngs</span><span class="p">.</span><span class="n">push_back</span><span class="p">(</span><span class="n">generate_png</span><span class="o">&lt;</span><span class="kt">int</span><span class="o">&gt;</span><span class="p">(</span><span class="n">ana_world</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">h_map</span><span class="p">,</span> <span class="nb">true</span><span class="p">,</span> <span class="mi">20</span><span class="p">));</span>

<span class="k">for</span><span class="p">(</span><span class="kt">int</span> <span class="n">i</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span> <span class="n">i</span> <span class="o">&lt;</span> <span class="n">sim_pngs</span><span class="p">.</span><span class="n">size</span><span class="p">();</span> <span class="o">++</span><span class="n">i</span><span class="p">)</span> <span class="p">{</span>
    <span class="n">display_image</span><span class="p">(</span><span class="n">sim_pngs</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="nb">true</span><span class="p">);</span>
    <span class="n">std</span><span class="o">::</span><span class="n">cout</span> <span class="o">&lt;&lt;</span> <span class="s">&quot;iteration = &quot;</span> <span class="o">&lt;&lt;</span> <span class="n">i</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span>
    <span class="n">std</span><span class="o">::</span><span class="n">this_thread</span><span class="o">::</span><span class="n">sleep_for</span><span class="p">(</span><span class="n">std</span><span class="o">::</span><span class="n">chrono</span><span class="o">::</span><span class="n">milliseconds</span><span class="p">(</span><span class="mi">800</span><span class="p">));</span>
<span class="p">}</span>

<span class="n">display_image</span><span class="p">(</span><span class="n">ana_pngs</span><span class="p">.</span><span class="n">back</span><span class="p">(),</span> <span class="nb">false</span><span class="p">);</span>
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
          <p>or develop a new analysis:</p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// time for new code</span>
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
          <p>Click <a href="#Display-Simulation-Images">here</a> to jump to the visualization cell of the simulation and display all simulation steps.</p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing text_cell rendered">
      <div class="prompt input_prompt"></div>
      <div class="inner_cell">
        <div class="text_cell_render border-box-sizing rendered_html">
          <h1 id="Resetting-the-simulation-without-restarting-the-kernel">Resetting the simulation without restarting the kernel<a class="anchor-link" href="#Resetting-the-simulation-without-restarting-the-kernel">&#182;</a></h1>
          <p>If you want to calculate the simulation with a new initial world without restarting the kernel, you must reset the following variables.</p>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span><span class="c1">// load a new inital world in the host memory</span>
<span class="n">read_input</span><span class="p">(</span><span class="s">&quot;input.txt&quot;</span><span class="p">,</span> <span class="n">sim_world</span><span class="p">,</span> <span class="n">dim</span><span class="p">,</span> <span class="n">dim</span><span class="p">,</span> <span class="nb">true</span><span class="p">);</span>
<span class="c1">// copy the world to the device</span>
<span class="n">cuCheck</span><span class="p">(</span><span class="n">cudaMemcpy</span><span class="p">(</span><span class="n">d_sim_world</span><span class="p">,</span> <span class="n">sim_world</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="kt">int</span><span class="p">)</span><span class="o">*</span><span class="n">world_size</span><span class="o">*</span><span class="n">world_size</span><span class="p">,</span> <span class="n">cudaMemcpyHostToDevice</span><span class="p">));</span>
<span class="c1">// delete the old images</span>
<span class="n">sim_pngs</span><span class="p">.</span><span class="n">clear</span><span class="p">();</span>
<span class="c1">// create an image of the initial world</span>
<span class="n">sim_pngs</span><span class="p">.</span><span class="n">push_back</span><span class="p">(</span><span class="n">generate_png</span><span class="o">&lt;</span><span class="kt">int</span><span class="o">&gt;</span><span class="p">(</span><span class="n">sim_world</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="n">world_size</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">bw_map</span><span class="p">,</span> <span class="nb">true</span><span class="p">,</span> <span class="mi">20</span><span class="p">));</span>
</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="cell border-box-sizing code_cell rendered">
      <div class="input">
        <div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
        <div class="inner_cell">
          <div class="input_area">
            <div class=" highlight hl-c++">
              <pre><span></span>
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
