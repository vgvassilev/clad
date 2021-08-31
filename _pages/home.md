---
title: "Compiler Research - Welcome to the Compiler Research Group"
layout: homelay
excerpt: "Compiler Research"
sitemap: false
permalink: /
---

We are a group of programming languages enthusiasts located at the Princeton
University and CERN. Our primary goal is research into foundational software
tools helping scientists to program for speed, interoperability, interactivity,
flexibility, and reproducibility.

Our current research focus is primarily in [interpretative C/C++/CUDA](interactive_cpp),
automatic differentiation tools, and C++ language interoperability with Python
and D.

## Recent Content

{% include carousel.html %}



Interested in joining the development or to use our work? Join our
[compiler-research-announce google groups forum](https://groups.google.com/g/compiler-research-announce).


**We are looking for and passionate undergrad and graduate students. Please visit
our [vacancies page](/vacancies).**

## Related Projects

<div class="grid-container grid-container--fill">
{% for coll in site.data.collabs %}
  <div class="grid-element">
  [<img src="/assets/collab_logos/{{coll.logo}}" class="img-responsive" width="85%" style="float: center" />]({{coll.url}})
  </div>
{% endfor %}
</div>

{% include open-embed.html %}
