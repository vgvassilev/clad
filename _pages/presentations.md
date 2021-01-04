---
title: "Presentations"
layout: gridlay
excerpt: "Presentations"
sitemap: false
permalink: /presentations/
---


# Related Presentations

{% assign sorted_pres = site.data.preslist | sort: "date" | reverse %}

<div style="padding-left: 40px;">

{% for pres in sorted_pres %}
  <b>{{ pres.title }}</b> <br />
  <em>{{ pres.speaker }} </em> at the {{pres.location}} ({{ pres.date | date: '%-d %B %Y' }}) {{ pres.artifacts }}
{% endfor %}

</div>
