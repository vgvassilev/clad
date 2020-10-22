---
title: "Presentations"
layout: gridlay
excerpt: "Presentations"
sitemap: false
permalink: /presentations/
---


# Presentations


{% for pres in site.data.preslist %}

  <b>{{ pres.title }}</b> <br />
  <em>{{ pres.speaker }} </em> at the {{pres.location}} ({{pres.date}}) (<a href="{{ pres.link.url }}">{{ pres.link.display }}</a>)

{% endfor %}
