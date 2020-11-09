---
title: "Compiler Research - Driven by enabling scientists to program for speed, Interoperability, Interactivity, Flexibility, and Reproducibility"
layout: homelay
excerpt: "Compiler Research."
sitemap: false
permalink: /
---


A key enabler of innovation and discovery for many scientific researchers is the ability to explore data and express ideas quickly as software prototypes. Tools and techniques that reduce the “time to insight” are essential to the productivity of researchers. At the same time massive increases in data volumes and computational needs require a continual focus on maximizing code performance. To manage these competing requirements, today’s researchers often find themselves using a heterogeneous and complex mix of programming languages, development tools, and hardware.

The C++ programming language is used for many numerically intensive scientific applications. A combination of performance and solid backward compatibility has led to its use for many research software codes over the past 20 years. Despite its power, C++ is often seen as difficult to learn and inconsistent with rapid application development. Exploration and prototyping is slowed down by the long edit-compile-run cycles during development.

Over the last decade, toegher with collaborators,
we have developed an interactive, interpretative C++ (aka REPL) based on LLVM and clang.
Amongst our research goals are to 
 * Advance the interpretative technology to provide a state-of-the-art C++ execution environment,
 * Enable functionality which can provide native-like, dynamic runtime interoperability between
C++ and Python (and eventually other languages), and
 * Allow seamless utilization of heterogeneous hardware (such as hardware accelerators)


Interested in joining the development to to use our work? Join our [cppaas-announce google groups forum](https://groups.google.com/forum/#!forum/cppaas-announce) [here](https://groups.google.com/forum/#!forum/cppaas-announce/join).

We are looking for interested and passionate undergrad and graduate students. Fellowships (and open projects) currently available via [IRIS-HEP](https://iris-hep.org/fellows.html).

## Contributors
{% for user in site.github.contributors %}
<div class="col-sm-6 clearfix">
  [{{ user.login }}]({{user.html_url}})
  [<img src="{{user.avatar_url}}" class="img-responsive" width="100" style="float: center" />]({{user.html_url}})
</div>
  
{% endfor %}

## Collaborators and Related Projects

{% assign number_printed = 0 %}
{% for coll in site.data.collabs %}
{% assign even_odd = number_printed | modulo: 4 %}
{% if even_odd == 0 %}
<div class="row">
{% endif %}

<div class="col-sm-3 clearfix">
  [<img src="/assets/collab_logos/{{coll.logo}}" class="img-responsive" width="75%" style="float: center" />]({{coll.url}})
</div>

{% assign number_printed = number_printed | plus: 1 %}

{% if even_odd == 3 %}
</div>
{% endif %}

{% endfor %}

{% assign even_odd = number_printed | modulo: 4 %}
{% if even_odd > 0 %}
</div>
{% endif %}
