---
title: "Presentations"
layout: gridlay
excerpt: "Presentations"
sitemap: false
permalink: /presentations/
---


# Presentations

{% assign sorted_pres = site.data.preslist | sort: "date" | reverse %}

{% for pres in sorted_pres %}
  <b>{{ pres.title }}</b> <br />
  <em>{{ pres.speaker }} </em> at the {{pres.location}} ({{ pres.date | date: '%-d %B %Y' }}) (<a href="{{ pres.link.url }}">{{ pres.link.display }}</a>)

{% endfor %}
