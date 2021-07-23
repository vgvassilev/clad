---
title: "Compiler Research - Interactive C++"
layout: textlay
excerpt: "Compiler Research"
sitemap: false
permalink: /interactive_cpp
---

# Interactive C++

The C++ programming language is used for many numerically intensive scientific
applications. A combination of performance and solid backward compatibility has
led to its use for many research software codes over the past 20 years. Despite
its power, C++ is often seen as difficult to learn and inconsistent with rapid
application development. Exploration and prototyping is slowed down by the long
edit-compile-run cycles during development.

[Cling](https://github.com/root-project/cling/) has emerged as a recognized
capability that enables interactivity, dynamic interoperability and rapid
prototyping capabilities to C++ developers. Cling supports the full C++ feature
set including the use of templates, lambdas, and virtual inheritance. Cling is
an interactive C++ interpreter, built on top of the Clang and LLVM compiler
infrastructure. The interpreter enables interactive exploration and makes the
C++ language more welcoming for research.

<div align=center style="max-width:1095px; margin:0 auto;">
  <img src="https://blog.llvm.org/img/cling-2020-11-30-figure1.gif" style="max-width:90%;"><br />
  <!--- ![alt_text](https://blog.llvm.org/img/cling-2020-11-30-figure1.gif "image_tooltip") --->
 <p align="center">
  Figure 1. Interactive OpenGL Demo, adapted from
  <a href="https://www.youtube.com/watch?v=eoIuqLNvzFs">here</a>.
  </p>
</div>

A key enabler of innovation and discovery for many scientific researchers is the
ability to explore data and express ideas quickly as software prototypes. Tools
and techniques that reduce the “time to insight” are essential to the
productivity of researchers. At the same time massive increases in data volumes
and computational needs require a continual focus on maximizing code
performance. To manage these competing requirements, today’s researchers often
find themselves using a heterogeneous and complex mix of programming languages,
development tools, and hardware.

Over the last decade, together with collaborators,
we have developed an interactive, interpretative C++ (aka REPL) based on LLVM and clang.
Amongst our research goals are to
 * Advance the interpretative technology to provide a state-of-the-art C++ execution environment,
 * Enable functionality which can provide native-like, dynamic runtime interoperability between
C++ and Python (and eventually other languages), and
 * Allow seamless utilization of heterogeneous hardware (such as hardware accelerators)


## Contributors

{% assign people = "" | split: ',' %}
{% assign names = "" | split: ',' %}
{% for repo in site.data.repos %}
{% for user in repo[1] %}
{% unless names contains user.login %}
{% assign people = people | push: user %}
{% assign names = names | push: user.login %}
{% endunless %}
{% endfor %}
{% endfor %}

{% assign sorted_people = people | sort: "login" %}


<div class="grid-container grid-container--fill">
{% for user in sorted_people %}
  <div class="grid-element">
  [{{ user.login }}]({{user.html_url}}) <br />
  [<img src="{{user.avatar_url}}" width="100" style="float: center" />]({{user.html_url}})
  </div>
{% endfor %}
</div>

