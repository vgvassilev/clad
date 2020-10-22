---
title: "Presentations"
layout: gridlay
excerpt: "Presentations"
sitemap: false
permalink: /presentations/
---


# Presentations


{% for pres in site.data.preslist %}

  {{ pres.title }} <br />
  <em>{{ pres.speaker }} </em> at the {{pres.location}} ({{pres.date}}) (<a href="{{ pres.link.url }}">{{ pres.link.display }}</a>)

{% endfor %}
